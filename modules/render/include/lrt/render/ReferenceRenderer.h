// Copyright (c) 2026 lucabRTrender contributors.
//
// The ground truth, on the GPU: shaders/lrt/reference. Every pixel considers
// every splat and sorts what reaches it exactly. For tests and for checking a
// scene by eye (`lrt render --technique reference`); far too slow for anything
// else. There is no CPU reference in this engine.
#pragma once

#include <cstdint>
#include <span>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/render/TileRasterizer.h"

namespace lrt::render {

class ReferenceRenderer {
public:
    [[nodiscard]] static Result<ReferenceRenderer> create(gpu::ShaderLibrary& library);

    /// Renders into `targets`; returns how many pixels had more contributors
    /// than the reference can hold (a comparison is only meaningful at zero).
    [[nodiscard]] Result<uint32_t> render(const Camera& camera,
                                          std::span<const SplatInstance> instances,
                                          const RenderSettings& settings, RenderTargets& targets,
                                          std::span<const PointInstance> points = {});

    /// The ray tracer's ground truth (GaussianRayTracer): every particle at
    /// its peak along every pixel's ray, sorted exactly by that peak. Splats
    /// only. Returns the overflowing pixel count, as `render` does.
    [[nodiscard]] Result<uint32_t> renderPeaks(const Camera& camera,
                                               std::span<const SplatInstance> instances,
                                               const RenderSettings& settings,
                                               RenderTargets& targets);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel project_;
    gpu::ComputeKernel pointsProject_;
    gpu::ComputeKernel blend_;
    gpu::ComputeKernel count_;
    gpu::ComputeKernel peakProject_;
    gpu::ComputeKernel peakBlend_;
};

/// Two renders compared on the GPU: the distribution of per-pixel worst
/// channel differences in 8-bit sRGB code values (alpha included).
struct ImageDifference {
    uint64_t pixels = 0;
    uint32_t p99 = 0;
    uint32_t max = 0;
    uint64_t over2 = 0;   ///< pixels differing by more than 2 code values
};

[[nodiscard]] Result<ImageDifference> compareImages(gpu::ShaderLibrary& library,
                                                    const gpu::Buffer& a, const gpu::Buffer& b,
                                                    uint32_t width, uint32_t height);

}   // namespace lrt::render
