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
    if (targets.width != width || targets.height != height || !targets.ids.valid()) {
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
        cursor["normals"].setBinding(scene.normals().rhi());
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

}   // namespace lrt::technique
