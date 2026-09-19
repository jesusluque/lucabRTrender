// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/MaterialPrograms.h"

#include <array>
#include <cstdio>

#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

namespace {

/// Asking a sample's material whether it is there at all. A UsdPreviewSurface
/// with an `opacityThreshold` has opacity 0 or 1; one without has what its
/// opacity says, which is coverage: the raster route cuts by a pixel's lot
/// (a hash, so a frame is one image), the path tracer only what is fully
/// gone, and draws its own lot a sample. Only rows flagged as cutouts pay.
const char* kCutout = R"(
float pixelLot(uint2 pixel) {
    uint state = (pixel.y * 65536u + pixel.x) * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    // Never exactly 0: an opacity of 0 always cuts, one of 1 never does.
    return (float(word >> 8) + 0.5) * (1.0 / 16777216.0);
}

public bool materialCuts(CameraParams camera, uint2 pixel, uint4 seen) {
    const Surface s = surfaceAt(camera, pixel.x, pixel.y, seen);
    // An invisible face cuts as a cutout does, whatever its material. Its
    // flag rides in the subset word's top bit: Metal allows a kernel 31
    // buffers, and the path tracer had no room for one more.
    if ((triangleSubsets[s.mesh.firstTriangle + s.triangle] & 0x80000000u) != 0) {
        return true;
    }
    const MaterialRecord m = materials[materialRowOf(s)];
    if ((m.flags & kMaterialCutout) == 0) {
        return false;
    }
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, pixel.x, pixel.y, s, lookup.time);
    evaluateMaterial(m.function, inputs, m.blob);
    return gLrtResult.opacity < (lookup.alphaDither != 0 ? pixelLot(pixel) : 1.0 / 512.0);
}
)";

}   // namespace

Result<MaterialPrograms> MaterialPrograms::create(gpu::ShaderLibrary& library) {
    MaterialPrograms programs;
    programs.library_ = &library;
    LRT_TRY(programs.setModules({}));
    return programs;
}

Result<void> MaterialPrograms::setModules(std::span<const material::CompiledMaterial> modules) {
    std::string signature;
    for (const material::CompiledMaterial& m : modules) {
        signature += m.module + ";";
    }
    if (signature == signature_) {
        return ok();
    }
    for (const material::CompiledMaterial& m : modules) {
        auto loaded = library_->loadSource(m.module, m.source, {});
        if (!loaded) return std::move(loaded).error();
    }
    std::string source = "__exported import lrt.material.material_runtime;\n"
                         "__exported import lrt.technique.material_lookup;\n";
    for (const material::CompiledMaterial& m : modules) {
        source += "import " + m.module + ";\n";
    }
    source += "\npublic void evaluateMaterial(uint function, MaterialInputs inputs, uint blob) {\n    switch (function) {\n";
    for (size_t k = 0; k < modules.size(); ++k) {
        source += "    case " + std::to_string(k + 1) + ": " + modules[k].function + "(inputs, blob); break;\n";
    }
    source += "    default: lrtFallbackMaterial(inputs); break;\n    }\n}\n";
    source += kCutout;
    // Named after what it dispatches to: one module per distinct set.
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : signature) {
        hash = (hash ^ c) * 1099511628211ULL;
    }
    std::array<char, 40> name{};
    std::snprintf(name.data(), name.size(), "lrt_materials_%016llx", static_cast<unsigned long long>(hash));
    auto loaded = library_->loadSource(name.data(), source, {});
    if (!loaded) return std::move(loaded).error();
    module_ = name.data();
    signature_ = signature;
    return ok();
}

void bindScene(rhi::ShaderCursor cursor, const world::GpuScene& scene) {
    cursor["positions"].setBinding(scene.positions().rhi());
    cursor["indices"].setBinding(scene.indices().rhi());
    cursor["meshes"].setBinding(scene.meshRecords().rhi());
    cursor["instances"].setBinding(scene.instanceRecords().rhi());
    cursor["primvarRecords"].setBinding(scene.primvarRecords().rhi());
    cursor["primvarValues"].setBinding(scene.primvarValues().rhi());
    cursor["primvarSlots"].setBinding(scene.primvarSlots().rhi());
    cursor["triangleCorners"].setBinding(scene.triangleCorners().rhi());
    cursor["triangleFaces"].setBinding(scene.triangleFaces().rhi());
}

void bindMaterialFrame(rhi::ShaderCursor cursor, const MaterialFrame& frame, const render::Projection& projection) {
    bindScene(cursor, *frame.scene);
    cursor["materials"].setBinding(frame.records->rhi());
    cursor["triangleSubsets"].setBinding(frame.scene->triangleSubsets().rhi());
    cursor["subsetRows"].setBinding(frame.scene->subsetRows().rhi());
    cursor["gMaterialBlob"].setBinding(frame.blob->rhi());
    frame.textures->bind(cursor["gTextures"]);
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    cursor["toWorld"]["row0"].setData(toWorld.data(), sizeof(float) * 4);
    cursor["toWorld"]["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
    cursor["toWorld"]["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
    cursor["lookup"]["time"].setData(frame.time);
    cursor["lookup"]["alphaDither"].setData(frame.alphaDither);
}

}   // namespace lrt::technique
