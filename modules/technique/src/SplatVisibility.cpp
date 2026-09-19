#include "lrt/technique/SplatVisibility.h"

#include <algorithm>
#include <cmath>

#include "lrt/core/Log.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {
namespace {

/// splat_visibility.slang's VisibilityPart, laid out as the shader reads it.
struct PartRecord {
    float    boxMin[4];
    float    boxMax[4];
    uint32_t grid;
    uint32_t octave;
    uint32_t offset;
    uint32_t pad;
};
static_assert(sizeof(PartRecord) == 48);

[[nodiscard]] Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t elementBytes,
                                               const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * elementBytes;
    desc.elementBytes = elementBytes;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

}   // namespace

VisibilityParts partitionJoints(const std::vector<std::string>& paths, uint32_t targetParts, uint32_t minJoints) {
    const size_t n = paths.size();
    std::vector<int32_t> parent(n, -1);
    std::vector<std::vector<uint32_t>> children(n);
    {
        // A joint's parent is the joint whose path is its own without the
        // last name; a lookup by string, since that is what the file holds.
        std::vector<std::pair<std::string, uint32_t>> byPath;
        byPath.reserve(n);
        for (uint32_t j = 0; j < n; ++j) byPath.emplace_back(paths[j], j);
        std::sort(byPath.begin(), byPath.end());
        for (uint32_t j = 0; j < n; ++j) {
            const size_t slash = paths[j].rfind('/');
            if (slash == std::string::npos) continue;
            const std::string above = paths[j].substr(0, slash);
            auto it = std::lower_bound(byPath.begin(), byPath.end(), std::make_pair(above, uint32_t{0}));
            if (it != byPath.end() && it->first == above) {
                parent[j] = static_cast<int32_t>(it->second);
                children[it->second].push_back(j);
            }
        }
    }
    std::vector<uint32_t> size(n, 1);
    for (size_t j = n; j-- > 0;) {
        if (parent[j] >= 0) size[static_cast<size_t>(parent[j])] += size[j];
    }
    // A part is a head joint and everything under it not claimed by a head
    // below. Start from the biggest root; the other roots are parts only if
    // they are big enough, otherwise they join it.
    std::vector<uint32_t> heads;
    uint32_t biggestRoot = 0;
    for (uint32_t j = 0; j < n; ++j) {
        if (parent[j] < 0 && (heads.empty() || size[j] > size[biggestRoot])) biggestRoot = j;
        if (parent[j] < 0) heads.push_back(j);
    }
    heads.erase(std::remove_if(heads.begin(), heads.end(),
                               [&](uint32_t h) { return h != biggestRoot && size[h] < minJoints; }),
                heads.end());
    // Then split: the largest subtree anywhere that is not yet a part and is
    // big enough becomes one, until there are enough. Biggest first is what
    // separates a wing from a chest before a toe from a foot.
    while (heads.size() < targetParts) {
        uint32_t best = static_cast<uint32_t>(n);
        for (uint32_t j = 0; j < n; ++j) {
            if (size[j] < minJoints) continue;
            if (std::find(heads.begin(), heads.end(), j) != heads.end()) continue;
            if (best == n || size[j] > size[best]) best = j;
        }
        if (best == n) break;
        heads.push_back(best);
    }
    VisibilityParts out;
    out.jointToPart.assign(n, 0);
    std::vector<int32_t> headOf(n, -1);
    for (uint32_t p = 0; p < heads.size(); ++p) headOf[heads[p]] = static_cast<int32_t>(p);
    for (uint32_t j = 0; j < n; ++j) {
        int32_t at = static_cast<int32_t>(j);
        while (at >= 0 && headOf[static_cast<size_t>(at)] < 0) at = parent[static_cast<size_t>(at)];
        out.jointToPart[j] = at >= 0 ? static_cast<uint32_t>(headOf[static_cast<size_t>(at)])
                                     : static_cast<uint32_t>(headOf[biggestRoot]);
    }
    out.partJoint = heads;
    return out;
}

Result<SplatVisibility> SplatVisibility::create(gpu::ShaderLibrary& library) {
    SplatVisibility v;
    v.device_ = &library.device();
    v.library_ = &library;
    const auto make = [&](std::optional<gpu::ComputeKernel>& into, const char* entry,
                          const char* module = "lrt/splat/splat_visibility") -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into.emplace(std::move(*kernel));
        return ok();
    };
    LRT_TRY(make(v.partOf_, "visPartOf"));
    LRT_TRY(make(v.mark_, "visMark"));
    LRT_TRY(make(v.gather_, "visGather"));
    // The bake traces an inline ray, which a device may not have (CUDA has
    // not): then there is no baking here, and a cloud baked elsewhere is
    // still read.
    if (auto bake = gpu::ComputeKernel::create(library, "lrt/splat/splat_visibility_bake", "visBake")) {
        v.bake_.emplace(std::move(*bake));
    } else {
        log::info("visibility: no bake on this device ({}); baked clouds are read", bake.error().toString());
    }
    LRT_TRY(make(v.factors_, "visFactors", "lrt/splat/splat_visibility_read"));
    LRT_TRY(make(v.histogram_, "visHistogram"));
    LRT_TRY(make(v.ambient_, "visAmbient"));
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    v.prefix_.emplace(std::move(*prefix));
    return v;
}

Result<void> SplatVisibility::bake(scene::GpuSplats& cloud, const gpu::Buffer& influences, uint32_t perSplat,
                                   const VisibilityParts& parts, const VisibilityBakeOptions& options) {
    const uint32_t count = cloud.count;
    const auto partCount = static_cast<uint32_t>(parts.partJoint.size());
    if (count == 0 || partCount == 0 || parts.jointToPart.empty()) {
        return Error(ErrorCode::InvalidArgument, "a visibility bake needs a cloud and at least one part");
    }
    if (!bake_.has_value()) {
        return Error(ErrorCode::DeviceFailure,
                     "a visibility bake traces inline rays, which this device has not: bake it where it can be");
    }
    if (!tracer_.has_value()) {
        render::RayTracerSettings settings;
        settings.route = render::RayTracingRoute::Hardware;
        auto made = render::GaussianRayTracer::create(*library_, settings);
        if (!made) return std::move(made).error();
        tracer_.emplace(std::move(*made));
    }

    // 1. Which part each gaussian is.
    auto table = gpu::Buffer::fromSpan<uint32_t>(*device_, parts.jointToPart, "visibility.jointToPart");
    if (!table) return std::move(table).error();
    auto partOf = deviceBuffer(*device_, count, 4, "visibility.partOf");
    if (!partOf) return std::move(partOf).error();
    {
        gpu::CommandBatch batch(*device_);
        partOf_->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["influences"].setBinding(influences.rhi());
            cursor["jointToPart"].setBinding(table->rhi());
            cursor["partOf"].setBinding(partOf->rhi());
            cursor["partParams"]["count"].setData(count);
            cursor["partParams"]["perSplat"].setData(perSplat);
        });
        LRT_TRY(batch.submit(true));
    }

    // 2. The box every field covers: the cloud's, with a margin.
    const float extent[3] = {cloud.bounds.max[0] - cloud.bounds.min[0], cloud.bounds.max[1] - cloud.bounds.min[1],
                             cloud.bounds.max[2] - cloud.bounds.min[2]};
    const float longest = std::max({extent[0], extent[1], extent[2], 1.0e-6F});
    const float pad = longest * options.margin;
    const uint32_t grid = std::max(options.grid, 2u);
    const uint32_t octave = std::max(options.octave, 2u);
    const uint64_t texelsPerPart = uint64_t{grid} * grid * grid * octave * octave;
    std::vector<PartRecord> records(partCount);
    for (uint32_t p = 0; p < partCount; ++p) {
        PartRecord& r = records[p];
        for (int a = 0; a < 3; ++a) {
            r.boxMin[a] = cloud.bounds.min[static_cast<size_t>(a)] - pad;
            r.boxMax[a] = cloud.bounds.max[static_cast<size_t>(a)] + pad;
        }
        r.boxMin[3] = static_cast<float>(parts.partJoint[p]);
        r.boxMax[3] = 0.0F;
        r.grid = grid;
        r.octave = octave;
        r.offset = static_cast<uint32_t>(texelsPerPart * p);
        r.pad = 0;
    }
    const uint64_t words = (texelsPerPart * partCount + 1) / 2;
    auto texels = deviceBuffer(*device_, words, 4, "visibility.texels");
    if (!texels) return std::move(texels).error();
    const uint64_t probesPerPart = uint64_t{grid} * grid * grid;
    const uint64_t ambientWords = (probesPerPart * partCount + 1) / 2;
    auto ambient = deviceBuffer(*device_, ambientWords, 4, "visibility.ambient");
    if (!ambient) return std::move(ambient).error();
    {
        const std::vector<uint32_t> zeros(static_cast<size_t>(ambientWords), 0u);
        LRT_TRY(ambient->write(*device_, 0, ambientWords * 4, zeros.data()));
    }
    {
        // Every half starts at zero and the bake ORs its own half in, so the
        // buffer is cleared once here: a device buffer holds what it held.
        const std::vector<uint32_t> zeros(static_cast<size_t>(words), 0u);
        LRT_TRY(texels->write(*device_, 0, words * 4, zeros.data()));
    }

    // 3. Each part: its gaussians gathered into a cloud of their own, proxies
    //    built over that, and every (probe, direction) traced through them.
    auto partBuffer = gpu::Buffer::fromSpan<PartRecord>(*device_, records, "visibility.parts");
    if (!partBuffer) return std::move(partBuffer).error();
    auto keep = deviceBuffer(*device_, count, 4, "visibility.keep");
    if (!keep) return std::move(keep).error();
    auto dest = deviceBuffer(*device_, count, 4, "visibility.dest");
    if (!dest) return std::move(dest).error();
    auto total = deviceBuffer(*device_, 1, 4, "visibility.total");
    if (!total) return std::move(total).error();
    const float cell = (longest + 2.0F * pad) / static_cast<float>(grid);
    for (uint32_t p = 0; p < partCount; ++p) {
        uint32_t kept = 0;
        {
            gpu::CommandBatch batch(*device_);
            mark_->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["gatherPartOf"].setBinding(partOf->rhi());
                cursor["keep"].setBinding(keep->rhi());
                cursor["gatherParams"]["count"].setData(count);
                cursor["gatherParams"]["part"].setData(p);
            });
            LRT_TRY(prefix_->apply(batch, *keep, *dest, *total, count));
            LRT_TRY(batch.submit(true));
            LRT_TRY(total->read(*device_, 0, sizeof(kept), &kept));
        }
        if (kept == 0) {
            // A PART NOTHING STANDS ON CASTS NOTHING. Its texels were never
            // written and hold zero, which read as a transmittance would be a
            // total shadow from every direction: the first bake of the
            // sparrow, whose Root and Center joints carry no gaussian, came
            // out black for exactly that. A grid of 0 is what the read skips.
            records[p].grid = 0;
            continue;
        }
        scene::GpuSplats part;
        part.source = cloud.source + " (part)";
        part.count = kept;
        part.declared = kept;
        part.restPerColour = 0;
        part.shWords = cloud.shWords;
        auto positions = deviceBuffer(*device_, kept, 16, "visibility.part.positions");
        if (!positions) return std::move(positions).error();
        auto shape = deviceBuffer(*device_, uint64_t{kept} * 4, 4, "visibility.part.shape");
        if (!shape) return std::move(shape).error();
        part.positions = std::move(*positions);
        part.shape = std::move(*shape);
        part.sh = cloud.sh;   // never read for a shadow, bound because it is declared
        {
            gpu::CommandBatch batch(*device_);
            gather_->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["gatherPartOf"].setBinding(partOf->rhi());
                cursor["dest"].setBinding(dest->rhi());
                cursor["keep"].setBinding(keep->rhi());
                cursor["restPositions"].setBinding(cloud.positions.rhi());
                cursor["restShape"].setBinding(cloud.shape.rhi());
                cursor["partPositions"].setBinding(part.positions.rhi());
                cursor["partShape"].setBinding(part.shape.rhi());
                cursor["gatherParams"]["count"].setData(count);
                cursor["gatherParams"]["part"].setData(p);
            });
            LRT_TRY(batch.submit(true));
        }
        // The whole cloud's box stands in for the part's: a superset, which is
        // all the proxies' quantisation needs, and no pass over the part.
        part.bounds = cloud.bounds;

        render::SplatInstance instance;
        instance.splats = &part;
        render::Projection projection;
        projection.eyeWorld = render::Vec3{(cloud.bounds.min[0] + cloud.bounds.max[0]) * 0.5,
                                           (cloud.bounds.min[1] + cloud.bounds.max[1]) * 0.5,
                                           (cloud.bounds.min[2] + cloud.bounds.max[2]) * 0.5};
        auto prepared = tracer_->prepare(projection, std::span<const render::SplatInstance>(&instance, 1), 0);
        if (!prepared) return std::move(prepared).error();
        const render::ShadowScene scene = tracer_->shadowScene();
        if (scene.tlas == nullptr) {
            return Error(ErrorCode::InternalError, "a visibility bake needs the Hardware route");
        }
        const auto threads = static_cast<uint32_t>(texelsPerPart);
        gpu::CommandBatch batch(*device_);
        bake_->dispatch(batch, {threads, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["parts"].setBinding(partBuffer->rhi());
            cursor["shadowFrames"].setBinding(scene.frames->rhi());
            cursor["shadowColours"].setBinding(scene.colours->rhi());
            cursor["shadowInstanceData"].setBinding(scene.instanceData->rhi());
            cursor["shadowInstanceIndices"].setBinding(scene.instanceIndices->rhi());
            cursor["texels"].setBinding(texels->rhi());
            cursor["scene"].setBinding(scene.tlas);
            rhi::ShaderCursor b = cursor["bakeParams"];
            b["part"].setData(p);
            b["threads"].setData(threads);
            b["cut"].setData(options.cut);
            b["start"].setData(cell);
        });
        LRT_TRY(batch.submit(true));
        // The ambient term: each probe's mean over its directions.
        {
            gpu::CommandBatch again(*device_);
            ambient_->dispatch(again, {static_cast<uint32_t>(probesPerPart), 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["parts"].setBinding(partBuffer->rhi());
                cursor["texels"].setBinding(texels->rhi());
                cursor["ambient"].setBinding(ambient->rhi());
                cursor["ambientParams"]["part"].setData(p);
                cursor["ambientParams"]["probes"].setData(static_cast<uint32_t>(probesPerPart));
            });
            LRT_TRY(again.submit(true));
        }
        // What the field holds, as a number: the share of its texels under a
        // half. A part whose field stops nothing shows up here, not in a frame.
        auto counts = deviceBuffer(*device_, 2, 4, "visibility.histogram");
        if (!counts) return std::move(counts).error();
        {
            const uint32_t zero[2] = {0, 0};
            LRT_TRY(counts->write(*device_, 0, sizeof(zero), zero));
            gpu::CommandBatch again(*device_);
            histogram_->dispatch(again, {threads, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["fieldWords"].setBinding(texels->rhi());
                cursor["fieldFloats"].setBinding(texels->rhi());
                cursor["histogram"].setBinding(counts->rhi());
                cursor["histogramParams"]["offset"].setData(records[p].offset);
                cursor["histogramParams"]["texels"].setData(threads);
                cursor["histogramParams"]["cut"].setData(0.5F);
                cursor["histogramParams"]["floats"].setData(0u);
            });
            LRT_TRY(again.submit(true));
        }
        uint32_t under[2] = {0, 0};
        LRT_TRY(counts->read(*device_, 0, sizeof(under), under));
        log::info("visibility: part {} of {}: {} gaussians, {} texels, {:.1f}% under a half", p + 1, partCount,
                  kept, threads, under[1] > 0 ? 100.0 * under[0] / under[1] : 0.0);
    }
    // The records again, with the empty parts marked for the read to skip.
    auto marked = gpu::Buffer::fromSpan<PartRecord>(*device_, records, "visibility.parts");
    if (!marked) return std::move(marked).error();
    cloud.visibilityParts = std::move(*marked);
    cloud.visibilityTexels = std::move(*texels);
    cloud.visibilityAmbient = std::move(*ambient);
    cloud.visibilityPartOf = std::move(*partOf);
    cloud.visibilityPartCount = partCount;
    return ok();
}

Result<double> SplatVisibility::shadowedShare(const gpu::Buffer& factors, uint32_t count, float cut) {
    auto counts = deviceBuffer(*device_, 2, 4, "visibility.histogram");
    if (!counts) return std::move(counts).error();
    const uint32_t zero[2] = {0, 0};
    LRT_TRY(counts->write(*device_, 0, sizeof(zero), zero));
    gpu::CommandBatch batch(*device_);
    histogram_->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["fieldWords"].setBinding(factors.rhi());
        cursor["fieldFloats"].setBinding(factors.rhi());
        cursor["histogram"].setBinding(counts->rhi());
        cursor["histogramParams"]["offset"].setData(0u);
        cursor["histogramParams"]["texels"].setData(count);
        cursor["histogramParams"]["cut"].setData(cut);
        cursor["histogramParams"]["floats"].setData(1u);
    });
    LRT_TRY(batch.submit(true));
    uint32_t under[2] = {0, 0};
    LRT_TRY(counts->read(*device_, 0, sizeof(under), under));
    return under[1] > 0 ? static_cast<double>(under[0]) / under[1] : 0.0;
}

Result<void> SplatVisibility::factors(gpu::CommandBatch& batch, const VisibilityFactorsJob& job,
                                      const gpu::Buffer& factors) {
    const scene::GpuSplats* cloud = job.cloud;
    if (cloud == nullptr || !cloud->hasVisibility() || job.positions == nullptr || job.lights == nullptr ||
        !cloud->visibilityPartOf.valid()) {
        return Error(ErrorCode::InvalidArgument,
                     "visibility factors need a baked cloud with its parts, positions and lights");
    }
    factors_->dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(job.positions->rhi());
        cursor["ownPart"].setBinding(cloud->visibilityPartOf.rhi());
        cursor["skinningXforms"].setBinding(job.skinningXforms != nullptr ? job.skinningXforms->rhi()
                                                                          : job.positions->rhi());
        cursor["lights"].setBinding(job.lights->rhi());
        cursor["parts"].setBinding(cloud->visibilityParts.rhi());
        cursor["texels"].setBinding(cloud->visibilityTexels.rhi());
        cursor["ambient"].setBinding(cloud->visibilityAmbient.valid() ? cloud->visibilityAmbient.rhi()
                                                                       : cloud->visibilityTexels.rhi());
        cursor["factors"].setBinding(factors.rhi());
        rhi::ShaderCursor f = cursor["factorParams"];
        f["count"].setData(cloud->count);
        f["base"].setData(job.base);
        f["lights"].setData(job.lightCount);
        f["partCount"].setData(cloud->visibilityPartCount);
        f["categoriesLo"].setData(static_cast<uint32_t>(job.categories & 0xFFFFFFFFu));
        f["categoriesHi"].setData(static_cast<uint32_t>(job.categories >> 32));
        f["skinned"].setData(uint32_t{job.skinningXforms != nullptr ? 1u : 0u});
        static const char* kRow[12] = {"w00", "w01", "w02", "w03", "w10", "w11",
                                       "w12", "w13", "w20", "w21", "w22", "w23"};
        for (int k = 0; k < 12; ++k) {
            f[kRow[k]].setData(job.objectToWorld[static_cast<size_t>(k)]);
        }
        static const char* kBindRow[12] = {"g00", "g01", "g02", "g03", "g10", "g11",
                                           "g12", "g13", "g20", "g21", "g22", "g23"};
        for (int k = 0; k < 12; ++k) {
            f[kBindRow[k]].setData(job.geomBind[static_cast<size_t>(k)]);
        }
    });
    return ok();
}

}   // namespace lrt::technique
