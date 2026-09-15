// Copyright (c) 2026 lucabRTrender contributors.
//
// The generated Slang every pass that touches a material shares: a module
// that imports each compiled material (MaterialCompiler) and dispatches on a
// material row's function. Shading calls it to build a lobe stack; the
// visibility passes call it to ask whether a sample's material cuts the
// sample away (`opacityThreshold`), which is why the module has to be one
// thing both can import rather than a kernel of its own.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/light/LightTable.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"
#include "lrt/render/Camera.h"
#include "lrt/world/GpuScene.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::technique {

/// A material row, as shaders/lrt/technique/material_lookup.slang reads it.
/// `function` 0 is the fallback (displayColor); k is the k-th module given to
/// MaterialPrograms::setModules, counted from 1.
struct MaterialRecord {
    uint32_t function = 0;
    uint32_t blob = 0;      ///< its first word in the blob
    uint32_t flags = 0;
    uint32_t pad = 0;
};

/// MaterialRecord::flags: the material's opacity cuts samples out instead of
/// blending them, so visibility itself has to evaluate it.
inline constexpr uint32_t kMaterialCutout = 1u;

class MaterialPrograms {
public:
    [[nodiscard]] static Result<MaterialPrograms> create(gpu::ShaderLibrary& library);

    /// The compiled materials the module dispatches to, loaded into the
    /// library here. Modules that repeat are not loaded twice; a set that
    /// repeats regenerates nothing.
    [[nodiscard]] Result<void> setModules(std::span<const material::CompiledMaterial> modules);

    /// What to import: named after the set it dispatches to, so one module
    /// serves every pass and every frame that shows the same materials.
    [[nodiscard]] const std::string& module() const noexcept { return module_; }

    [[nodiscard]] gpu::ShaderLibrary& library() const noexcept { return *library_; }

private:
    gpu::ShaderLibrary* library_ = nullptr;
    std::string         module_;
    std::string         signature_ = "unset";
};

/// Where a frame's light groups go: `count` planes of float4, a pixel each,
/// one after another in `colour` (the mean, as the frame's colour) for the
/// material shading; the path tracer keeps them in its own accumulation
/// (PathTracer::lightGroupMeanOffset). A frame with none compiles the
/// kernels without them. At most kMaxLightGroups.
struct LightGroupTargets {
    const gpu::Buffer* colour = nullptr;
    uint32_t           count = 0;
};
constexpr uint32_t kMaxLightGroups = 8;

/// What generated material code reads: the scene a visibility sample is
/// rebuilt from, the material rows and the blob their values live in, the
/// textures, and the frame's time.
struct MaterialFrame {
    const MaterialPrograms*       programs = nullptr;
    const world::GpuScene*        scene = nullptr;
    const gpu::Buffer*            records = nullptr;
    const gpu::Buffer*            blob = nullptr;
    const material::TextureStore* textures = nullptr;
    float                         time = 0.0F;
    /// The frame's lights. None: the headlight, as meshes were lit before
    /// there were any.
    const light::LightTable*      lights = nullptr;
    /// What a shadow ray traces against, when the device traces at all.
    rhi::IAccelerationStructure*  shadows = nullptr;
    /// Samples per light. One is the interactive choice; a test that wants
    /// an area light's irradiance without noise asks for more.
    uint32_t                      samples = 1;
    /// One light per sample, chosen by power, instead of every light at every
    /// pixel. Exact either way; what changes is where the cost goes -- with
    /// the loop it grows with the number of lights, with the choice it does
    /// not, at the price of noise a frame has to average away.
    bool                          chooseLights = false;
    /// With `chooseLights`: choose through the light BVH -- by power over
    /// distance within the lights' cones, at the shading point -- where the
    /// table has one (any bounded light); by power alone otherwise.
    bool                          lightBvh = true;
    /// The light groups the frame writes beside its colour: a light's direct
    /// contribution, at every bounce, under its group. Emission and the
    /// background are in no group.
    LightGroupTargets             groups;
    /// The frame's volumes (world::VolumeSet's words) for the path tracer,
    /// which traces free flights through them and transmittance to the
    /// lights; null or none: no medium. The raster technique draws none.
    const gpu::Buffer*            volumes = nullptr;
    uint32_t                      volumeCount = 0;

    [[nodiscard]] bool valid() const noexcept {
        return programs != nullptr && scene != nullptr && records != nullptr && records->valid() && blob != nullptr &&
               blob->valid() && textures != nullptr;
    }
};

/// The scene buffers shaders/lrt/technique/surface.slang reads, by name.
void bindScene(rhi::ShaderCursor cursor, const world::GpuScene& scene);

/// Those, and material_lookup.slang's tables, textures and view to world.
/// Not the lights: only the shading kernel declares them, since the cutout
/// visibility passes evaluate a material for its opacity alone.
void bindMaterialFrame(rhi::ShaderCursor cursor, const MaterialFrame& frame, const render::Projection& projection);

}   // namespace lrt::technique
