// Copyright (c) 2026 lucabRTrender contributors.
//
// A visibility buffer shaded by its instances' MaterialX materials: one
// kernel, generated to import every compiled material module and dispatch on
// a material row's function, that reconstructs each pixel's surface
// (material_surface.slang), evaluates its material into a lobe stack, and
// lights it. Until scene lights arrive (M5) the light is the headlight: a
// unit light from the eye, as HeadlightShading draws unshaded meshes.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"
#include "lrt/render/Camera.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/GpuScene.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::technique {

/// shaders: a material row. `function` 0 is the fallback (displayColor);
/// k is the k-th module given to setModules, counted from 1.
struct MaterialRecord {
    uint32_t function = 0;
    uint32_t blob = 0;      ///< its first word in the blob
    uint32_t flags = 0;
    uint32_t pad = 0;
};

class MaterialShading {
public:
    [[nodiscard]] static Result<MaterialShading> create(gpu::ShaderLibrary& library);

    /// The material modules the kernel dispatches to, loaded here; function k
    /// (from 1) is modules[k - 1]. The kernel is rebuilt when the list changes.
    [[nodiscard]] Result<void> setModules(std::span<const material::CompiledMaterial> modules);

    [[nodiscard]] Result<void> shade(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                     const VisibilityTargets& targets, const render::Projection& projection,
                                     const gpu::Buffer& records, const gpu::Buffer& blob,
                                     const material::TextureStore& textures, float time, render::RenderTargets& out);

private:
    gpu::ShaderLibrary*              library_ = nullptr;
    gpu::Device*                     device_ = nullptr;
    std::optional<gpu::ComputeKernel> kernel_;
    std::string                      signature_ = "unset";
};

}   // namespace lrt::technique
