// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/Visibility.h"

#include <algorithm>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

void setCamera(rhi::ShaderCursor p, const render::Projection& projection, uint32_t width, uint32_t height) {
    p["width"].setData(width);
    p["height"].setData(height);
    p["focalX"].setData(static_cast<float>(projection.focalX));
    p["focalY"].setData(static_cast<float>(projection.focalY));
    p["centreX"].setData(static_cast<float>(projection.centreX));
    p["centreY"].setData(static_cast<float>(projection.centreY));
    p["nearZ"].setData(static_cast<float>(projection.nearZ));
    p["farZ"].setData(static_cast<float>(projection.farZ));
    p["orthographic"].setData(uint32_t{projection.orthographic ? 1u : 0u});
}

Result<VisibilityRaster> VisibilityRaster::create(gpu::ShaderLibrary& library) {
    gpu::RasterDesc desc;
    desc.module = "lrt/technique/visibility_raster";
    desc.vertexEntry = "visibilityVertex";
    desc.fragmentEntry = "visibilityFragment";
    rhi::ColorTargetDesc ids;
    ids.format = rhi::Format::RGBA32Uint;
    desc.targets = {ids};
    desc.depthFormat = rhi::Format::D32Float;
    desc.depthFunc = rhi::ComparisonFunc::Greater;   // reversed: nearer is larger
    auto pass = gpu::RasterKernel::create(library, desc);
    if (!pass) return std::move(pass).error();
    VisibilityRaster raster;
    raster.device_ = &library.device();
    raster.pass_ = std::move(*pass);
    return raster;
}

Result<void> VisibilityRaster::render(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                      const render::Projection& projection, uint32_t width, uint32_t height,
                                      VisibilityTargets& targets) {
    // Rays leave no depth texture: the rasteriser remakes both.
    if (targets.width != width || targets.height != height || !targets.ids.valid() || !targets.depth.valid()) {
        gpu::TextureDesc ids;
        ids.width = width;
        ids.height = height;
        ids.format = rhi::Format::RGBA32Uint;
        ids.usage = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::ShaderResource;
        ids.label = "visibility.ids";
        auto madeIds = gpu::Texture::create(*device_, ids);
        if (!madeIds) return std::move(madeIds).error();
        gpu::TextureDesc depth = ids;
        depth.format = rhi::Format::D32Float;
        depth.usage = rhi::TextureUsage::DepthStencil;
        depth.label = "visibility.depth";
        auto madeDepth = gpu::Texture::create(*device_, depth);
        if (!madeDepth) return std::move(madeDepth).error();
        targets.ids = std::move(*madeIds);
        targets.depth = std::move(*madeDepth);
        targets.width = width;
        targets.height = height;
    }
    auto idsView = targets.ids.view(0);
    if (!idsView) return std::move(idsView).error();
    auto depthView = targets.depth.view(0);
    if (!depthView) return std::move(depthView).error();
    gpu::RasterPass pass;
    pass.width = width;
    pass.height = height;
    pass.colours = {(*idsView).get()};
    pass.clearColours = {{0.0F, 0.0F, 0.0F, 0.0F}};
    pass.depth = (*depthView).get();
    pass.depthClear = 0.0F;
    std::vector<gpu::RasterDraw> draws;
    for (const world::DrawRange& range : scene.draws()) {
        const geom::GpuMesh& mesh = scene.mesh(range.mesh);
        gpu::RasterDraw draw;
        draw.vertexCount = mesh.triangles * 3;
        draw.instanceCount = range.instances;
        const uint32_t firstInstance = range.firstInstance;
        const uint32_t firstPoint = scene.firstPoint(range.mesh);
        const uint32_t firstTriangle = scene.firstTriangle(range.mesh);
        draw.bind = [&, firstInstance, firstPoint, firstTriangle](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(scene.positions().rhi());
            cursor["indices"].setBinding(scene.indices().rhi());
            cursor["instances"].setBinding(scene.instanceRecords().rhi());
            setCamera(cursor["camera"], projection, width, height);
            cursor["draw"]["firstInstance"].setData(firstInstance);
            cursor["draw"]["firstPoint"].setData(firstPoint);
            cursor["draw"]["firstTriangle"].setData(firstTriangle);
        };
        draws.push_back(std::move(draw));
    }
    pass_.run(batch, pass, draws);
    return ok();
}

Result<VisibilityTrace> VisibilityTrace::create(gpu::ShaderLibrary& library) {
    if (!library.device().caps().rayQuery) {
        return Error(ErrorCode::Unsupported, "no ray queries on this device");
    }
    auto kernel = gpu::ComputeKernel::create(library, "lrt/technique/visibility_trace", "visibilityTrace");
    if (!kernel) return std::move(kernel).error();
    VisibilityTrace trace;
    trace.device_ = &library.device();
    trace.trace_ = std::move(*kernel);
    return trace;
}

Result<void> VisibilityTrace::render(gpu::CommandBatch& batch, const world::RayTracingScene& scene,
                                     const render::Projection& projection, uint32_t width, uint32_t height,
                                     VisibilityTargets& targets) {
    if (targets.width != width || targets.height != height || !targets.ids.valid() ||
        (targets.ids.desc().usage & rhi::TextureUsage::UnorderedAccess) == rhi::TextureUsage::None) {
        gpu::TextureDesc ids;
        ids.width = width;
        ids.height = height;
        ids.format = rhi::Format::RGBA32Uint;
        ids.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource |
                    rhi::TextureUsage::RenderTarget;
        ids.label = "visibility.ids";
        auto made = gpu::Texture::create(*device_, ids);
        if (!made) return std::move(made).error();
        targets.ids = std::move(*made);
        targets.depth = {};
        targets.width = width;
        targets.height = height;
    }
    auto view = targets.ids.view(0);
    if (!view) return std::move(view).error();
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    trace_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["scene"].setBinding(scene.topLevel());
        cursor["ids"].setBinding((*view).get());
        setCamera(cursor["camera"], projection, width, height);
        static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                  "v12", "v13", "v20", "v21", "v22", "v23"};
        for (size_t k = 0; k < 12; ++k) {
            cursor["trace"][kNames[k]].setData(toWorld[k]);
        }
    });
    return ok();
}

namespace {

/// A storage-writable id texture of the frame's size.
Result<void> writableIds(gpu::Device& device, uint32_t width, uint32_t height, VisibilityTargets& targets) {
    if (targets.width == width && targets.height == height && targets.ids.valid() &&
        (targets.ids.desc().usage & rhi::TextureUsage::UnorderedAccess) != rhi::TextureUsage::None) {
        return ok();
    }
    gpu::TextureDesc ids;
    ids.width = width;
    ids.height = height;
    ids.format = rhi::Format::RGBA32Uint;
    ids.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource |
                rhi::TextureUsage::RenderTarget;
    ids.label = "visibility.ids";
    auto made = gpu::Texture::create(device, ids);
    if (!made) return std::move(made).error();
    targets.ids = std::move(*made);
    targets.depth = {};
    targets.width = width;
    targets.height = height;
    return ok();
}

}   // namespace

Result<VisibilityBvh> VisibilityBvh::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "lrt/technique/visibility_bvh", "visibilityBvh");
    if (!kernel) return std::move(kernel).error();
    VisibilityBvh v;
    v.device_ = &library.device();
    v.traverse_ = std::move(*kernel);
    return v;
}

Result<void> VisibilityBvh::render(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                   const world::BvhScene& bvh, const render::Projection& projection, uint32_t width,
                                   uint32_t height, VisibilityTargets& targets) {
    LRT_TRY(writableIds(*device_, width, height, targets));
    auto view = targets.ids.view(0);
    if (!view) return std::move(view).error();
    traverse_.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(scene.positions().rhi());
        cursor["indices"].setBinding(scene.indices().rhi());
        cursor["meshes"].setBinding(scene.meshRecords().rhi());
        cursor["instanceRecords"].setBinding(scene.instanceRecords().rhi());
        cursor["topBoxes"].setBinding(bvh.topBoxes().rhi());
        cursor["topChildren"].setBinding(bvh.topChildren().rhi());
        cursor["topLeaves"].setBinding(bvh.topLeaves().rhi());
        cursor["meshBoxes"].setBinding(bvh.meshBoxes().rhi());
        cursor["meshChildren"].setBinding(bvh.meshChildren().rhi());
        cursor["meshLeaves"].setBinding(bvh.meshLeaves().rhi());
        cursor["ids"].setBinding((*view).get());
        setCamera(cursor["camera"], projection, width, height);
        cursor["scene"]["instances"].setData(scene.instanceCount());
    });
    return ok();
}

Result<HeadlightShading> HeadlightShading::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "lrt/technique/shade_headlight", "shadeHeadlight");
    if (!kernel) return std::move(kernel).error();
    HeadlightShading shading;
    shading.device_ = &library.device();
    shading.shade_ = std::move(*kernel);
    return shading;
}

Result<void> HeadlightShading::shade(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                     const VisibilityTargets& targets, const render::Projection& projection,
                                     render::RenderTargets& out) {
    const uint64_t pixels = uint64_t{targets.width} * targets.height;
    if (out.width != targets.width || out.height != targets.height || !out.colour.valid()) {
        gpu::BufferDesc colour;
        colour.bytes = pixels * 16;
        colour.elementBytes = 16;
        colour.label = "shading.colour";
        auto madeColour = gpu::Buffer::create(*device_, colour);
        if (!madeColour) return std::move(madeColour).error();
        gpu::BufferDesc depth;
        depth.bytes = pixels * 4;
        depth.elementBytes = 4;
        depth.label = "shading.depth";
        auto madeDepth = gpu::Buffer::create(*device_, depth);
        if (!madeDepth) return std::move(madeDepth).error();
        out.colour = std::move(*madeColour);
        out.depth = std::move(*madeDepth);
        out.width = targets.width;
        out.height = targets.height;
    }
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    shade_.dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(scene.positions().rhi());
        cursor["primvarRecords"].setBinding(scene.primvarRecords().rhi());
        cursor["primvarValues"].setBinding(scene.primvarValues().rhi());
        cursor["primvarSlots"].setBinding(scene.primvarSlots().rhi());
        cursor["triangleCorners"].setBinding(scene.triangleCorners().rhi());
        cursor["triangleFaces"].setBinding(scene.triangleFaces().rhi());
        cursor["indices"].setBinding(scene.indices().rhi());
        cursor["meshes"].setBinding(scene.meshRecords().rhi());
        cursor["instances"].setBinding(scene.instanceRecords().rhi());
        cursor["visibility"].setBinding((*ids).get());
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        setCamera(cursor["camera"], projection, targets.width, targets.height);
    });
    return ok();
}

Result<AovShading> AovShading::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "lrt/technique/aovs", "aovs");
    if (!kernel) return std::move(kernel).error();
    AovShading shading;
    shading.device_ = &library.device();
    shading.aovs_ = std::move(*kernel);
    return shading;
}

Result<void> AovShading::shade(gpu::CommandBatch& batch, const world::GpuScene& scene,
                               const VisibilityTargets& targets, const render::Projection& projection,
                               std::span<const uint32_t> slots, AovBuffers& out) {
    const uint64_t pixels = uint64_t{targets.width} * targets.height;
    const uint32_t slotCount = static_cast<uint32_t>(slots.size());
    const auto make = [&](gpu::Buffer& into, uint64_t count, uint32_t element, const char* label) -> Result<void> {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(count, 1) * element;
        desc.elementBytes = element;
        desc.label = label;
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    if (out.width != targets.width || out.height != targets.height || out.primvarSlots != slotCount ||
        !out.ids.valid()) {
        LRT_TRY(make(out.ids, pixels * 3, 4, "aov.ids"));
        LRT_TRY(make(out.eyeNormals, pixels, 16, "aov.Neye"));
        LRT_TRY(make(out.worldNormals, pixels, 16, "aov.normal"));
        LRT_TRY(make(out.primvars, pixels * std::max<uint32_t>(slotCount, 1), 16, "aov.primvars"));
        out.width = targets.width;
        out.height = targets.height;
        out.primvarSlots = slotCount;
    }
    std::vector<uint32_t> slotList(slots.begin(), slots.end());
    if (slotList.empty()) {
        slotList.push_back(0);
    }
    auto slotBuffer = gpu::Buffer::fromSpan<uint32_t>(*device_, slotList, "aov.slots");
    if (!slotBuffer) return std::move(slotBuffer).error();
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    aovs_.dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(scene.positions().rhi());
        cursor["indices"].setBinding(scene.indices().rhi());
        cursor["meshes"].setBinding(scene.meshRecords().rhi());
        cursor["instances"].setBinding(scene.instanceRecords().rhi());
        cursor["primvarRecords"].setBinding(scene.primvarRecords().rhi());
        cursor["primvarValues"].setBinding(scene.primvarValues().rhi());
        cursor["primvarSlots"].setBinding(scene.primvarSlots().rhi());
        cursor["triangleCorners"].setBinding(scene.triangleCorners().rhi());
        cursor["triangleFaces"].setBinding(scene.triangleFaces().rhi());
        cursor["visibility"].setBinding((*ids).get());
        cursor["ids"].setBinding(out.ids.rhi());
        cursor["eyeNormals"].setBinding(out.eyeNormals.rhi());
        cursor["worldNormals"].setBinding(out.worldNormals.rhi());
        cursor["primvarsOut"].setBinding(out.primvars.rhi());
        cursor["primvarSlotOf"].setBinding(slotBuffer->rhi());
        setCamera(cursor["camera"], projection, targets.width, targets.height);
        cursor["params"]["primvarSlots"].setData(slotCount);
    });
    return ok();
}

}   // namespace lrt::technique
