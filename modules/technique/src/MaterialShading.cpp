// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/MaterialShading.h"

#include <cstdio>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

namespace {

const char* kKernelBody = R"(
struct MaterialRecord {
    uint function;
    uint blob;
    uint flags;
    uint pad0;
};

struct ShadeParams {
    float time;
    uint  pad0; uint pad1; uint pad2;
};

Texture2D<uint4>                visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>      colour;
RWStructuredBuffer<float>       depth;
StructuredBuffer<MaterialRecord> materials;
ConstantBuffer<CameraParams>    camera;
ConstantBuffer<ViewToWorld>     toWorld;
ConstantBuffer<ShadeParams>     params;

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeMaterials(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        colour[at] = float4(0.0);
        depth[at] = 0.0;
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, params.time);
    const MaterialRecord m = materials[s.instance.flags >> 8];
    evaluateMaterial(m.function, inputs, m.blob);
    const LobeStack stack = gLrtResult;
    // The headlight: unit radiance from the eye, so a white Lambert surface
    // facing it shows 1.
    const float3 toEye = normalize(inputs.viewPosition - inputs.positionWorld);
    const float3 radiance = stack.emission + kPi * stackEval(stack, toEye, toEye);
    colour[at] = float4(radiance * stack.opacity, stack.opacity);
    depth[at] = s.depth;
}
)";

}   // namespace

Result<MaterialShading> MaterialShading::create(gpu::ShaderLibrary& library) {
    MaterialShading shading;
    shading.library_ = &library;
    shading.device_ = &library.device();
    LRT_TRY(shading.setModules({}));
    return shading;
}

Result<void> MaterialShading::setModules(std::span<const material::CompiledMaterial> modules) {
    std::string signature;
    for (const material::CompiledMaterial& m : modules) {
        signature += m.module + ";";
    }
    if (signature == signature_ && kernel_.has_value()) {
        return ok();
    }
    for (const material::CompiledMaterial& m : modules) {
        auto loaded = library_->loadSource(m.module, m.source, {});
        if (!loaded) return std::move(loaded).error();
    }
    std::string source = "import lrt.material.material_runtime;\nimport lrt.technique.material_surface;\n";
    for (const material::CompiledMaterial& m : modules) {
        source += "import " + m.module + ";\n";
    }
    source += "\nvoid evaluateMaterial(uint function, MaterialInputs inputs, uint blob) {\n    switch (function) {\n";
    for (size_t k = 0; k < modules.size(); ++k) {
        source += "    case " + std::to_string(k + 1) + ": " + modules[k].function + "(inputs, blob); break;\n";
    }
    source += "    default: lrtFallbackMaterial(inputs); break;\n    }\n}\n";
    source += kKernelBody;
    // Named after what it dispatches to: one kernel per distinct set of modules.
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : signature) {
        hash = (hash ^ c) * 1099511628211ULL;
    }
    char name[40];
    std::snprintf(name, sizeof(name), "lrt_shade_%016llx", static_cast<unsigned long long>(hash));
    auto program = library_->loadSource(name, source, {"shadeMaterials"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "shadeMaterials");
    if (!kernel) return std::move(kernel).error();
    kernel_.emplace(std::move(*kernel));
    signature_ = signature;
    return ok();
}

Result<void> MaterialShading::shade(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                    const VisibilityTargets& targets, const render::Projection& projection,
                                    const gpu::Buffer& records, const gpu::Buffer& blob,
                                    const material::TextureStore& textures, float time, render::RenderTargets& out) {
    const uint64_t pixels = uint64_t{targets.width} * targets.height;
    if (out.width != targets.width || out.height != targets.height || !out.colour.valid()) {
        gpu::BufferDesc colour;
        colour.bytes = pixels * 16;
        colour.elementBytes = 16;
        colour.label = "materials.colour";
        auto madeColour = gpu::Buffer::create(*device_, colour);
        if (!madeColour) return std::move(madeColour).error();
        gpu::BufferDesc depth;
        depth.bytes = pixels * 4;
        depth.elementBytes = 4;
        depth.label = "materials.depth";
        auto madeDepth = gpu::Buffer::create(*device_, depth);
        if (!madeDepth) return std::move(madeDepth).error();
        out.colour = std::move(*madeColour);
        out.depth = std::move(*madeDepth);
        out.width = targets.width;
        out.height = targets.height;
    }
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    kernel_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
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
        cursor["materials"].setBinding(records.rhi());
        cursor["gMaterialBlob"].setBinding(blob.rhi());
        textures.bind(cursor["gTextures"]);
        setCamera(cursor["camera"], projection, targets.width, targets.height);
        cursor["toWorld"]["row0"].setData(toWorld.data(), sizeof(float) * 4);
        cursor["toWorld"]["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
        cursor["toWorld"]["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
        cursor["params"]["time"].setData(time);
    });
    return ok();
}

}   // namespace lrt::technique
