// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/render/ReferenceRenderer.h"

#include <algorithm>
#include <array>

#include "FrameParams.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::render {
namespace {

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

}   // namespace

Result<ReferenceRenderer> ReferenceRenderer::create(gpu::ShaderLibrary& library) {
    ReferenceRenderer r;
    r.device_ = &library.device();
    auto project = gpu::ComputeKernel::create(library, "lrt/reference/reference_project", "referenceProject");
    if (!project) return std::move(project).error();
    auto points = gpu::ComputeKernel::create(library, "lrt/reference/reference_points_project",
                                             "referencePointsProject");
    if (!points) return std::move(points).error();
    r.pointsProject_ = std::move(*points);
    auto blend = gpu::ComputeKernel::create(library, "lrt/reference/reference_blend", "referenceBlend");
    if (!blend) return std::move(blend).error();
    auto count = gpu::ComputeKernel::create(library, "lrt/reference/count_nonzero", "countNonzero");
    if (!count) return std::move(count).error();
    r.project_ = std::move(*project);
    r.blend_ = std::move(*blend);
    r.count_ = std::move(*count);
    auto peakProject = gpu::ComputeKernel::create(library, "lrt/reference/reference_peak_project",
                                                  "referencePeakProject");
    if (!peakProject) return std::move(peakProject).error();
    auto peakBlend = gpu::ComputeKernel::create(library, "lrt/reference/reference_peak_blend",
                                                "referencePeakBlend");
    if (!peakBlend) return std::move(peakBlend).error();
    r.peakProject_ = std::move(*peakProject);
    r.peakBlend_ = std::move(*peakBlend);
    return r;
}

Result<uint32_t> ReferenceRenderer::render(const Camera& camera,
                                           std::span<const SplatInstance> instances,
                                           const RenderSettings& settings, RenderTargets& targets,
                                           std::span<const PointInstance> points) {
    const Projection projection = projectionFor(camera, settings.width, settings.height);
    FrameCommon common{settings.width, settings.height, (settings.width + 15) / 16,
                       (settings.height + 15) / 16, &projection, &settings};
    uint32_t total = 0;
    for (const SplatInstance& instance : instances) {
        if (instance.splats != nullptr) {
            total += instance.splats->count;
        }
    }
    for (const PointInstance& instance : points) {
        if (instance.points != nullptr) {
            total += instance.points->count;
        }
    }
    const uint64_t pixels = uint64_t{settings.width} * settings.height;
    auto screen = buffer(*device_, total, 16, "ref.screen");
    if (!screen) return std::move(screen).error();
    auto conic = buffer(*device_, total, 16, "ref.conic");
    if (!conic) return std::move(conic).error();
    auto colour = buffer(*device_, total, 16, "ref.colour");
    if (!colour) return std::move(colour).error();
    auto overflow = buffer(*device_, pixels, 4, "ref.overflow");
    if (!overflow) return std::move(overflow).error();
    auto overflowCount = buffer(*device_, 1, 4, "ref.overflowCount");
    if (!overflowCount) return std::move(overflowCount).error();
    if (targets.width != settings.width || targets.height != settings.height || !targets.colour.valid()) {
        auto c = buffer(*device_, pixels, 16, "ref.target.colour");
        if (!c) return std::move(c).error();
        auto d = buffer(*device_, pixels, 4, "ref.target.depth");
        if (!d) return std::move(d).error();
        targets.colour = *c;
        targets.depth = *d;
        targets.width = settings.width;
        targets.height = settings.height;
    }

    gpu::CommandBatch batch(*device_);
    uint32_t base = 0;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const Mat4 objectToView = projection.worldToView * instance.objectToWorld;
        const Vec3 eyeObject = aofx::xform::inverseAffine(instance.objectToWorld).point(projection.eyeWorld);
        project_.dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["shape"].setBinding(cloud->shape.rhi());
            cursor["sh"].setBinding(cloud->sh.rhi());
            cursor["refScreen"].setBinding(screen->rhi());
            cursor["refConic"].setBinding(conic->rhi());
            cursor["refColour"].setBinding(colour->rhi());
            setFrame(cursor, common);
            setObject(cursor, objectToView, eyeObject);
            cursor["params"]["count"].setData(cloud->count);
            cursor["params"]["base"].setData(base);
            cursor["params"]["restPerColour"].setData(cloud->restPerColour);
            cursor["params"]["shWords"].setData(cloud->shWords);
        });
        base += cloud->count;
    }
    for (const PointInstance& instance : points) {
        const scene::GpuPoints* cloud = instance.points;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const Mat4 objectToView = projection.worldToView * instance.objectToWorld;
        pointsProject_.dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["colours"].setBinding(cloud->colours.rhi());
            cursor["refScreen"].setBinding(screen->rhi());
            cursor["refConic"].setBinding(conic->rhi());
            cursor["refColour"].setBinding(colour->rhi());
            setFrame(cursor, common);
            setObject(cursor, objectToView, projection.eyeWorld);
            cursor["params"]["count"].setData(cloud->count);
            cursor["params"]["base"].setData(base);
            setPointParams(cursor["points"], projection, settings.width, settings.height,
                           objectToView, instance.style, cloud->count);
        });
        base += cloud->count;
    }
    blend_.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["refScreen"].setBinding(screen->rhi());
        cursor["refConic"].setBinding(conic->rhi());
        cursor["refColour"].setBinding(colour->rhi());
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["depth"].setBinding(targets.depth.rhi());
        cursor["overflow"].setBinding(overflow->rhi());
        setFrame(cursor, common);
        cursor["params"]["count"].setData(total);
    });
    count_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["values"].setBinding(overflow->rhi());
        cursor["total"].setBinding(overflowCount->rhi());
        cursor["params"]["count"].setData(static_cast<uint32_t>(pixels));
    });
    LRT_TRY(batch.submit(true));
    uint32_t over = 0;
    LRT_TRY(overflowCount->read(*device_, 0, sizeof(over), &over));
    return over;
}

Result<uint32_t> ReferenceRenderer::renderPeaks(const Camera& camera,
                                                std::span<const SplatInstance> instances,
                                                const RenderSettings& settings,
                                                RenderTargets& targets) {
    const Projection projection = projectionFor(camera, settings.width, settings.height);
    FrameCommon common{settings.width, settings.height, (settings.width + 15) / 16,
                       (settings.height + 15) / 16, &projection, &settings};
    uint32_t total = 0;
    for (const SplatInstance& instance : instances) {
        if (instance.splats != nullptr) {
            total += instance.splats->count;
        }
    }
    const uint64_t pixels = uint64_t{settings.width} * settings.height;
    auto row0 = buffer(*device_, total, 16, "peak.row0");
    if (!row0) return std::move(row0).error();
    auto row1 = buffer(*device_, total, 16, "peak.row1");
    if (!row1) return std::move(row1).error();
    auto row2 = buffer(*device_, total, 16, "peak.row2");
    if (!row2) return std::move(row2).error();
    auto colour = buffer(*device_, total, 16, "peak.colour");
    if (!colour) return std::move(colour).error();
    auto overflow = buffer(*device_, pixels, 4, "peak.overflow");
    if (!overflow) return std::move(overflow).error();
    auto overflowCount = buffer(*device_, 1, 4, "peak.overflowCount");
    if (!overflowCount) return std::move(overflowCount).error();
    if (targets.width != settings.width || targets.height != settings.height || !targets.colour.valid()) {
        auto c = buffer(*device_, pixels, 16, "peak.target.colour");
        if (!c) return std::move(c).error();
        auto d = buffer(*device_, pixels, 4, "peak.target.depth");
        if (!d) return std::move(d).error();
        targets.colour = *c;
        targets.depth = *d;
        targets.width = settings.width;
        targets.height = settings.height;
    }

    gpu::CommandBatch batch(*device_);
    uint32_t base = 0;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const Mat4 objectToView = projection.worldToView * instance.objectToWorld;
        const Vec3 eyeObject = aofx::xform::inverseAffine(instance.objectToWorld).point(projection.eyeWorld);
        peakProject_.dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["shape"].setBinding(cloud->shape.rhi());
            cursor["sh"].setBinding(cloud->sh.rhi());
            cursor["peakRow0"].setBinding(row0->rhi());
            cursor["peakRow1"].setBinding(row1->rhi());
            cursor["peakRow2"].setBinding(row2->rhi());
            cursor["peakColour"].setBinding(colour->rhi());
            setFrame(cursor, common);
            setObject(cursor, objectToView, eyeObject);
            cursor["params"]["count"].setData(cloud->count);
            cursor["params"]["base"].setData(base);
            cursor["params"]["restPerColour"].setData(cloud->restPerColour);
            cursor["params"]["shWords"].setData(cloud->shWords);
        });
        base += cloud->count;
    }
    peakBlend_.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["peakRow0"].setBinding(row0->rhi());
        cursor["peakRow1"].setBinding(row1->rhi());
        cursor["peakRow2"].setBinding(row2->rhi());
        cursor["peakColour"].setBinding(colour->rhi());
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["depth"].setBinding(targets.depth.rhi());
        cursor["overflow"].setBinding(overflow->rhi());
        setFrame(cursor, common);
        cursor["params"]["count"].setData(total);
    });
    count_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["values"].setBinding(overflow->rhi());
        cursor["total"].setBinding(overflowCount->rhi());
        cursor["params"]["count"].setData(static_cast<uint32_t>(pixels));
    });
    LRT_TRY(batch.submit(true));
    uint32_t over = 0;
    LRT_TRY(overflowCount->read(*device_, 0, sizeof(over), &over));
    return over;
}

Result<ImageDifference> compareImages(gpu::ShaderLibrary& library, const gpu::Buffer& a,
                                      const gpu::Buffer& b, uint32_t width, uint32_t height) {
    gpu::Device& device = library.device();
    auto compare = gpu::ComputeKernel::create(library, "lrt/reference/image_compare", "imageCompare");
    if (!compare) return std::move(compare).error();
    auto reduce = gpu::ComputeKernel::create(library, "lrt/reference/histogram_reduce", "histogramReduce");
    if (!reduce) return std::move(reduce).error();
    auto rows = buffer(device, uint64_t{height} * 256, 4, "compare.rows");
    if (!rows) return std::move(rows).error();
    auto bins = buffer(device, 256, 4, "compare.bins");
    if (!bins) return std::move(bins).error();
    gpu::CommandBatch batch(device);
    compare->dispatch(batch, {height, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["a"].setBinding(a.rhi());
        cursor["b"].setBinding(b.rhi());
        cursor["histogram"].setBinding(rows->rhi());
        cursor["params"]["width"].setData(width);
        cursor["params"]["height"].setData(height);
    });
    reduce->dispatch(batch, {256, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["histogram"].setBinding(rows->rhi());
        cursor["total"].setBinding(bins->rhi());
        cursor["params"]["rows"].setData(height);
        cursor["params"]["bins"].setData(uint32_t{256});
    });
    LRT_TRY(batch.submit(true));
    std::array<uint32_t, 256> histogram{};
    LRT_TRY(bins->read(device, 0, sizeof(histogram), histogram.data()));

    ImageDifference diff;
    for (uint32_t k = 0; k < 256; ++k) {
        diff.pixels += histogram[k];
        if (histogram[k] > 0) {
            diff.max = k;
        }
        if (k > 2) {
            diff.over2 += histogram[k];
        }
    }
    const uint64_t p99At = diff.pixels - diff.pixels / 100;
    uint64_t running = 0;
    for (uint32_t k = 0; k < 256; ++k) {
        running += histogram[k];
        if (running >= p99At) {
            diff.p99 = k;
            break;
        }
    }
    return diff;
}

}   // namespace lrt::render
