// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/Visibility.h"

#include <algorithm>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

namespace {

/// The pipeline the visibility raster draws with, whichever fragment shader
/// it ends in: one RGBA32Uint target of (instance + 1, triangle), reversed
/// depth, and no culling of its own (the fragment applies Hydra's rule).
gpu::RasterDesc visibilityDesc(const std::string& module, const char* vertexEntry, const char* fragmentEntry) {
    gpu::RasterDesc desc;
    desc.module = module;
    desc.vertexEntry = vertexEntry;
    desc.fragmentEntry = fragmentEntry;
    rhi::ColorTargetDesc ids;
    ids.format = rhi::Format::RGBA32Uint;
    desc.targets = {ids};
    desc.depthFormat = rhi::Format::D32Float;
    desc.depthFunc = rhi::ComparisonFunc::Greater;   // reversed: nearer is larger
    return desc;
}

/// How many samples one pixel's ray may have removed from it before it gives
/// up and shows what it last found. Sixteen was enough for the layers of
/// leaves a cutout texture is usually drawn for, and not for a sparrow's
/// belly: dozens of feather cards, clear where the ray crosses them, and the
/// body behind them was never reached -- a hole in the bird. Only a pixel
/// with that many layers pays for the rest.
const char* kCutoutSteps = "static const uint kCutoutSteps = 64;\n";

/// Past the sample a cutout removed, in the ray's own units: relative to how
/// far the ray has come (a float's spacing there is 1e-7 of it), with a floor
/// well under a feather card's distance from the body it lies on -- at a
/// floor of 1e-4 a card a tenth of a millimetre off the body hid it.
const char* kCutoutAdvance = "max(1.0e-6, t * 1.0e-5)";

std::string cutoutRasterSource(const std::string& materials) {
    return "import lrt.technique.visibility_geometry;\n"
           "import " + materials + ";\n"
           "\nConstantBuffer<CameraParams> camera;\n"
           "\n[shader(\"vertex\")]\n"
           "VertexOut cutoutVertex(uint vertex : SV_VertexID, uint instanceIndex : SV_InstanceID,\n"
           "                       uint startVertex : SV_StartVertexLocation,\n"
           "                       uint startInstance : SV_StartInstanceLocation) {\n"
           "    return visibilityVertexOf(camera, vertex, instanceIndex, startVertex, startInstance);\n"
           "}\n"
           "\n[shader(\"fragment\")]\n"
           "uint4 cutoutFragment(VertexOut input, bool front : SV_IsFrontFace) : SV_Target0 {\n"
           "    if (visibilityCulled(input, front)) {\n        discard;\n    }\n"
           "    const uint4 seen = uint4(input.instance + 1, input.triangle, 0, 0);\n"
           "    if (materialCuts(camera, visibilityPixel(camera, input.position), seen)) {\n        discard;\n    }\n"
           "    return seen;\n"
           "}\n";
}

std::string cutoutTraceSource(const std::string& materials) {
    return "import lrt.technique.visibility_trace;\n"
           "import " + materials + ";\n\n" + kCutoutSteps +
           "\n[shader(\"compute\")]\n[numthreads(16, 16, 1)]\n"
           "void traceCutout(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {\n"
           "    const uint2 tid = lrtQuadPixel(group.xy, index);\n"
           "    if (tid.x >= camera.width || tid.y >= camera.height) {\n        return;\n    }\n"
           "    float3 origin;\n    float3 direction;\n"
           "    traceRay(tid, origin, direction);\n"
           "    float tMin = traceTMin();\n"
           "    uint4 seen = uint4(0);\n"
           "    for (uint step = 0; step < kCutoutSteps; ++step) {\n"
           "        const TraceHit hit = traceNearest(origin, direction, tMin);\n"
           "        seen = hit.seen;\n"
           "        if (seen.x == 0 || !materialCuts(camera, tid, seen)) {\n            break;\n        }\n"
           "        seen = uint4(0);\n"
           "        const float t = hit.t;\n"
           "        tMin = t + " + kCutoutAdvance + ";\n"
           "    }\n"
           "    ids[int2(int(tid.x), int(camera.height - 1 - tid.y))] = seen;\n"
           "}\n";
}

std::string cutoutBvhSource(const std::string& materials) {
    return "import lrt.technique.visibility_bvh;\n"
           "import " + materials + ";\n\n" + kCutoutSteps +
           "\n[shader(\"compute\")]\n[numthreads(16, 16, 1)]\n"
           "void bvhCutout(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {\n"
           "    const uint2 tid = lrtQuadPixel(group.xy, index);\n"
           "    if (tid.x >= camera.width || tid.y >= camera.height) {\n        return;\n    }\n"
           "    float3 origin;\n    float3 direction;\n"
           "    viewRay(camera, float2(tid) + 0.5, origin, direction);\n"
           "    float tMin = bvhTMin();\n"
           "    uint4 seen = uint4(0);\n"
           "    for (uint step = 0; step < kCutoutSteps; ++step) {\n"
           "        const Best best = traverseNearest(origin, direction, tMin);\n"
           "        seen = best.seen();\n"
           "        if (seen.x == 0 || !materialCuts(camera, tid, seen)) {\n            break;\n        }\n"
           "        seen = uint4(0);\n"
           "        const float t = best.t;\n"
           "        tMin = t + " + kCutoutAdvance + ";\n"
           "    }\n"
           "    ids[int2(int(tid.x), int(camera.height - 1 - tid.y))] = seen;\n"
           "}\n";
}

}   // namespace

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
    auto pass = gpu::RasterKernel::create(
        library, visibilityDesc("lrt/technique/visibility_raster", "visibilityVertex", "visibilityFragment"));
    if (!pass) return std::move(pass).error();
    VisibilityRaster raster;
    raster.library_ = &library;
    raster.device_ = &library.device();
    raster.pass_ = std::move(*pass);
    return raster;
}

Result<void> VisibilityRaster::ensureCutout(const MaterialPrograms& programs) {
    if (cutoutModule_ == programs.module()) {
        return ok();
    }
    const std::string name = programs.module() + "_raster";
    auto loaded = library_->loadSource(name, cutoutRasterSource(programs.module()),
                                       {"cutoutVertex", "cutoutFragment"});
    if (!loaded) return std::move(loaded).error();
    auto pass = gpu::RasterKernel::create(*library_, visibilityDesc(name, "cutoutVertex", "cutoutFragment"));
    if (!pass) return std::move(pass).error();
    cutout_ = std::move(*pass);
    cutoutModule_ = programs.module();
    return ok();
}

Result<void> VisibilityRaster::render(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                      const render::Projection& projection, uint32_t width, uint32_t height,
                                      VisibilityTargets& targets, const MaterialFrame* cutouts) {
    if (cutouts != nullptr && !cutouts->valid()) {
        return Error(ErrorCode::InvalidArgument, "visibility raster: incomplete cutout materials");
    }
    if (cutouts != nullptr) {
        LRT_TRY(ensureCutout(*cutouts->programs));
    }
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
    pass.bind = [&](rhi::ShaderCursor cursor) {
        bindScene(cursor, scene);
        if (cutouts != nullptr) {
            bindMaterialFrame(cursor, *cutouts, projection);
        }
        setCamera(cursor["camera"], projection, width, height);
        cursor["pass"]["idsIncludeStart"].setData(uint32_t{device_->caps().drawIdsIncludeStart ? 1u : 0u});
    };
    std::vector<gpu::RasterDraw> draws;
    draws.reserve(scene.draws().size());
    for (const world::DrawRange& range : scene.draws()) {
        gpu::RasterDraw draw;
        draw.vertexCount = scene.mesh(range.mesh).triangles * 3;
        draw.instanceCount = range.instances;
        draw.firstVertex = scene.firstTriangle(range.mesh) * 3;
        draw.firstInstance = range.firstInstance;
        draws.push_back(draw);
    }
    (cutouts != nullptr ? cutout_ : pass_).run(batch, pass, draws);
    return ok();
}

Result<VisibilityTrace> VisibilityTrace::create(gpu::ShaderLibrary& library) {
    const gpu::Caps& caps = library.device().caps();
    if (!caps.rayQuery && !(caps.rayTracing && caps.accelerationStructure)) {
        return Error(ErrorCode::Unsupported, "no ray queries and no ray tracing pipeline on this device");
    }
    VisibilityTrace trace;
    trace.library_ = &library;
    trace.device_ = &library.device();
    if (caps.rayQuery) {
        auto kernel = gpu::ComputeKernel::create(library, "lrt/technique/visibility_trace", "visibilityTrace");
        if (!kernel) return std::move(kernel).error();
        trace.trace_ = std::move(*kernel);
    } else {
        // The pipeline route: one ray generation program, one miss, one
        // closest hit, no cutouts yet (those are generated kernels and their
        // ray is inline too).
        gpu::RayTracingDesc desc;
        desc.module = "lrt/technique/visibility_trace_rays";
        desc.rayGen = "visibilityTraceGen";
        desc.misses = {"visibilityMiss"};
        desc.hitGroups = {{"visibility", "visibilityHit", "", ""}};
        desc.maxRecursion = 1;
        desc.payloadBytes = 32;
        auto kernel = gpu::RayTracingKernel::create(library, desc);
        if (!kernel) return std::move(kernel).error();
        trace.traceRays_.emplace(std::move(*kernel));
    }
    return trace;
}

Result<void> VisibilityTrace::ensureCutout(const MaterialPrograms& programs) {
    if (cutoutModule_ == programs.module() && cutout_.has_value()) {
        return ok();
    }
    const std::string name = programs.module() + "_trace";
    auto loaded = library_->loadSource(name, cutoutTraceSource(programs.module()), {"traceCutout"});
    if (!loaded) return std::move(loaded).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "traceCutout");
    if (!kernel) return std::move(kernel).error();
    cutout_.emplace(std::move(*kernel));
    cutoutModule_ = programs.module();
    return ok();
}

Result<void> VisibilityTrace::render(gpu::CommandBatch& batch, const world::RayTracingScene& scene,
                                     const render::Projection& projection, uint32_t width, uint32_t height,
                                     VisibilityTargets& targets, const MaterialFrame* cutouts) {
    if (cutouts != nullptr && !cutouts->valid()) {
        return Error(ErrorCode::InvalidArgument, "visibility rays: incomplete cutout materials");
    }
    if (cutouts != nullptr) {
        LRT_TRY(ensureCutout(*cutouts->programs));
    }
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
    if (traceRays_.has_value() && cutouts != nullptr) {
        return Error(ErrorCode::Unsupported,
                     "visibility rays: a cutout pass needs an inline ray, which this device has not");
    }
    const auto bind = [&](rhi::ShaderCursor cursor) {
        if (cutouts != nullptr) {
            bindMaterialFrame(cursor, *cutouts, projection);
        }
        cursor["scene"].setBinding(scene.topLevel());
        cursor["ids"].setBinding((*view).get());
        setCamera(cursor["camera"], projection, width, height);
        static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                  "v12", "v13", "v20", "v21", "v22", "v23"};
        for (size_t k = 0; k < 12; ++k) {
            cursor["trace"][kNames[k]].setData(toWorld[k]);
        }
    };
    if (traceRays_.has_value()) {
        traceRays_->dispatch(batch, width, height, 1, bind);
    } else {
        (cutouts != nullptr ? *cutout_ : trace_).dispatch(batch, {width, height, 1}, bind);
    }
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
    v.library_ = &library;
    v.device_ = &library.device();
    v.traverse_ = std::move(*kernel);
    return v;
}

Result<void> VisibilityBvh::ensureCutout(const MaterialPrograms& programs) {
    if (cutoutModule_ == programs.module() && cutout_.has_value()) {
        return ok();
    }
    const std::string name = programs.module() + "_bvh";
    auto loaded = library_->loadSource(name, cutoutBvhSource(programs.module()), {"bvhCutout"});
    if (!loaded) return std::move(loaded).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "bvhCutout");
    if (!kernel) return std::move(kernel).error();
    cutout_.emplace(std::move(*kernel));
    cutoutModule_ = programs.module();
    return ok();
}

Result<void> VisibilityBvh::render(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                   const world::BvhScene& bvh, const render::Projection& projection, uint32_t width,
                                   uint32_t height, VisibilityTargets& targets, const MaterialFrame* cutouts) {
    if (cutouts != nullptr && !cutouts->valid()) {
        return Error(ErrorCode::InvalidArgument, "visibility bvh: incomplete cutout materials");
    }
    if (cutouts != nullptr) {
        LRT_TRY(ensureCutout(*cutouts->programs));
    }
    LRT_TRY(writableIds(*device_, width, height, targets));
    auto view = targets.ids.view(0);
    if (!view) return std::move(view).error();
    const gpu::ComputeKernel& kernel = cutouts != nullptr ? *cutout_ : traverse_;
    kernel.dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        bindScene(cursor, scene);
        if (cutouts != nullptr) {
            bindMaterialFrame(cursor, *cutouts, projection);
        }
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
