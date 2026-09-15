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
    /// Adaptive: a pixel whose mean's relative standard error has fallen
    /// below `errorTarget` after at least `minSamples` paths takes no more.
    /// The estimate is the pixel's own, from its luminance's second moment.
    bool     adaptive = false;
    float    errorTarget = 0.02F;
    uint32_t minSamples = 16;
};

/// How much of a frame has converged, for the adaptive gate: pixels the
/// visibility buffer covers, and those of them that have stopped.
struct PathProgress {
    uint32_t covered = 0;
    uint32_t converged = 0;
};

/// What the denoiser wants beside the colour: the first hit's albedo and its
/// shading normal, in world space, as the material saw them. Written once a
/// frame, since they do not depend on the sample.
/// One buffer of two planes -- a Metal kernel binds at most 31 buffers and
/// the traced kernel stands at the limit -- the albedo's first, the
/// normal's after it.
struct PathAux {
    gpu::Buffer planes;   ///< float4 a pixel, twice: the lobes' directional albedo at the view direction, then the shading normal, unit, facing the eye
    uint32_t    width = 0;
    uint32_t    height = 0;
    [[nodiscard]] bool valid() const noexcept { return planes.valid(); }
    /// Where the normal's plane starts, in float4 entries (and in bytes).
    [[nodiscard]] uint64_t normalOffset() const noexcept { return uint64_t{width} * height; }
    [[nodiscard]] uint64_t normalOffsetBytes() const noexcept { return normalOffset() * 16; }
};

class PathTracer {
public:
    [[nodiscard]] static Result<PathTracer> create(gpu::ShaderLibrary& library);

    /// The kernel is rebuilt when `programs` dispatches to a different set.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs);
    /// The same, for a frame with light groups (their buffers declared) or
    /// without; `trace` switches as the frame asks.
    [[nodiscard]] Result<void> setPrograms(const MaterialPrograms& programs, bool groups, bool volumes);

    /// Traces `settings.samples` paths a pixel into `out`, as a running mean
    /// over everything accumulated so far.
    /// `aux`, when given, receives the first hit's albedo and shading normal.
    [[nodiscard]] Result<void> trace(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                     const render::Projection& projection, const MaterialFrame& frame,
                                     const PathSettings& settings, render::RenderTargets& out,
                                     PathAux* aux = nullptr);

    /// How many paths a pixel the accumulation holds -- the frame's count;
    /// an adaptive pixel that stopped holds fewer.
    [[nodiscard]] uint32_t accumulated() const noexcept { return accumulated_; }

    /// Counts the frame's covered and converged pixels on the device and
    /// reads the two counters back. Meaningful after an adaptive trace.
    [[nodiscard]] Result<PathProgress> progress(const VisibilityTargets& targets);

    /// Forgets them, so the next trace starts the mean again.
    void restart() noexcept { accumulated_ = 0; }

    /// The accumulation as it stands, for a check that reads the moments: the
    /// sum of colour times opacity (float4 a pixel, its w the paths' opacity
    /// summed), and `moments`: the luminance's second moment as float bits, a
    /// pixel, then the adaptive stop flags, a pixel, in one buffer of words.
    [[nodiscard]] const gpu::Buffer& sum() const noexcept { return sum_; }
    [[nodiscard]] const gpu::Buffer& moments() const noexcept { return moments_; }
    [[nodiscard]] uint64_t doneOffset() const noexcept { return uint64_t{width_} * height_; }
    /// With light groups: where group `g`'s mean plane starts in `sum`, in
    /// float4 entries (its sum plane is at 1 + g planes; the means follow
    /// the sums).
    [[nodiscard]] uint64_t lightGroupMeanOffset(uint32_t g) const noexcept {
        const uint32_t groups = (sumPlanes_ - 1) / 2;
        return (uint64_t{1} + groups + g) * width_ * height_;
    }
    [[nodiscard]] uint32_t lightGroups() const noexcept { return (sumPlanes_ - 1) / 2; }

private:
    gpu::ShaderLibrary*               library_ = nullptr;
    gpu::Device*                      device_ = nullptr;
    std::optional<gpu::ComputeKernel> kernel_;
    std::string                       module_;
    bool                              groups_ = false;
    bool                              volumes_ = false;
    uint32_t                          sumPlanes_ = 1;   ///< 1 + 2 * light groups
    gpu::Buffer                       sum_;           ///< float4 a pixel: the paths added so far
    gpu::Buffer                       moments_;       ///< words: the second moment's float bits a pixel, then the done flags a pixel
    gpu::Buffer                       progress_;      ///< [covered, converged]
    std::optional<gpu::ComputeKernel> progressKernel_;   ///< pathDecide: the stop rule and the counters
    float                             lastErrorTarget_ = 0.02F;
    uint32_t                          lastMinSamples_ = 16;
    uint32_t                          accumulated_ = 0;
    uint32_t                          width_ = 0;
    uint32_t                          height_ = 0;
};

}   // namespace lrt::technique
