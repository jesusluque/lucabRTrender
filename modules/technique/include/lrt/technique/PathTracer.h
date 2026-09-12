// Copyright (c) 2026 lucabRTrender contributors.
//
// One bounce of path tracing over the same visibility buffer the raster
// shading reads, so the two can be compared: the surface is rebuilt the same
// way (material_surface.slang), its material evaluated into the same lobe
// stack, and its direct light gathered the same way -- and then one indirect
// ray is traced from it, shaded where it lands, and weighed against the light
// it might have hit directly.
//
// What this shares with MaterialShading is deliberate: the difference between
// them is the bounce, so a converged frame of each must agree wherever the
// bounce contributes nothing. What it does not share is the accumulation: a
// path traced frame is a running mean over samples, and it says how many it
// has taken.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
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

/// How a path traced frame is gathered.
struct PathSettings {
    uint32_t samples = 1;     ///< paths per pixel this call adds to the mean
    uint32_t bounces = 1;     ///< indirect bounces after the first hit
    uint32_t seed = 0;        ///< which samples these are: the frame's index
    bool     accumulate = false;   ///< add to what is there rather than replace it
};

class PathTracer {
public:
    [[nodiscard]] static Result<PathTracer> create(gpu::ShaderLibrary& library);

    /// The kernel is rebuilt when `programs` dispatches to a different set.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs);

    /// Traces `settings.samples` paths a pixel into `out`, as a running mean
    /// over everything accumulated so far.
    [[nodiscard]] Result<void> trace(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                     const render::Projection& projection, const MaterialFrame& frame,
                                     const PathSettings& settings, render::RenderTargets& out);

    /// How many paths a pixel the accumulation holds.
    [[nodiscard]] uint32_t accumulated() const noexcept { return accumulated_; }

    /// Forgets them, so the next trace starts the mean again.
    void restart() noexcept { accumulated_ = 0; }

private:
    gpu::ShaderLibrary*               library_ = nullptr;
    gpu::Device*                      device_ = nullptr;
    std::optional<gpu::ComputeKernel> kernel_;
    std::string                       module_;
    gpu::Buffer                       sum_;           ///< float4 a pixel: the paths added so far
    uint32_t                          accumulated_ = 0;
    uint32_t                          width_ = 0;
    uint32_t                          height_ = 0;
};

}   // namespace lrt::technique
