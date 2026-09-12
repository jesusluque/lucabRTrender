// Copyright (c) 2026 lucabRTrender contributors.
//
// A visibility buffer shaded by its instances' MaterialX materials: one
// kernel over MaterialPrograms' generated dispatch that reconstructs each
// pixel's surface (material_surface.slang), evaluates its material into a
// lobe stack, and lights it -- by the frame's lights, with shadows where the
// device traces rays, linking, and either every light at every pixel or one
// chosen a sample by its power. A frame with no lights at all falls back to
// the headlight, a unit light from the eye, as HeadlightShading draws
// unshaded meshes.
#pragma once

#include <optional>
#include <string>

#include "lrt/core/Result.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/render/Camera.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/technique/MaterialPrograms.h"
#include "lrt/technique/Visibility.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::technique {

class MaterialShading {
public:
    [[nodiscard]] static Result<MaterialShading> create(gpu::ShaderLibrary& library);

    /// The kernel is rebuilt when `programs` dispatches to a different set.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs);

    [[nodiscard]] Result<void> shade(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                     const render::Projection& projection, const MaterialFrame& frame,
                                     render::RenderTargets& out);

private:
    gpu::ShaderLibrary*              library_ = nullptr;
    gpu::Device*                     device_ = nullptr;
    std::optional<gpu::ComputeKernel> kernel_;
    std::string                      module_;
};

}   // namespace lrt::technique
