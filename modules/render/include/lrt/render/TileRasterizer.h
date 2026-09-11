// Copyright (c) 2026 lucabRTrender contributors.
//
// The splat rasteriser: global depth sort, stable tile sort, per-pixel blend,
// no per-tile capacity. The stages are described in shaders/lrt/splat/frame.slang.
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/PrefixSum.h"
#include "lrt/gpu/algo/RadixSort.h"
#include "lrt/render/Camera.h"
#include "lrt/render/Points.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::render {

/// A selection volume in a cloud's own space, what happens to the splats in
/// it, and a grade: openFXplayer's SplatEdit. The rule is written once, in
/// shaders/lrt/common/edit.slang; every splat renderer reads it.
struct SplatEdit {
    enum class Shape : uint32_t { Box = 0, Sphere = 1 };
    enum class Mode : uint32_t { Keep = 0, Remove = 1, Grade = 2 };
    bool                 active = false;
    Shape                shape = Shape::Box;
    Mode                 mode = Mode::Grade;
    std::array<float, 3> centre{0, 0, 0};
    /// Half extents of the box; radius of the sphere in x.
    std::array<float, 3> size{1, 1, 1};
    std::array<float, 3> tint{1, 1, 1};
    float                saturation = 1.0F;
    float                brightness = 1.0F;
    /// Multiplies the opacity of what the volume selects.
    float                opacity = 1.0F;
    /// Wherever the volume is: below this opacity a splat goes (0: off).
    float                minOpacity = 0.0F;
    /// Wherever the volume is: longer than this on its longest axis, in the
    /// cloud's units, a splat goes (0: off).
    float                maxScale = 0.0F;
    /// The volume test turned round.
    bool                 invert = false;
};

struct SplatInstance {
    const scene::GpuSplats* splats = nullptr;
    Mat4                    objectToWorld = Mat4::identity();
    SplatEdit               edit;
};

struct RenderSettings {
    uint32_t width = 1920;
    uint32_t height = 1080;
    bool     antialias = true;
    uint32_t maxShDegree = 3;
    /// Premultiplied, linear. Transparent by default.
    std::array<float, 4> background{0, 0, 0, 0};
    enum class Depth { Mean, Threshold };
    Depth    depth = Depth::Mean;
    float    depthThreshold = 0.5F;
    bool     linearise = true;
    /// Wait after every stage so `FrameStats` times each one. Slower; for bench.
    bool     timeStages = false;
};

struct FrameStats {
    uint32_t splats = 0;
    uint32_t visible = 0;
    uint32_t pairs = 0;
    double   projectMs = 0, depthSortMs = 0, countsMs = 0, emitMs = 0, tileSortMs = 0,
             blendMs = 0, totalMs = 0;
};

/// The images a frame writes: bottom row first, width * height.
struct RenderTargets {
    uint32_t    width = 0;
    uint32_t    height = 0;
    gpu::Buffer colour;   ///< float4, premultiplied, linear
    gpu::Buffer depth;    ///< float, view z
};

class TileRasterizer {
public:
    [[nodiscard]] static Result<TileRasterizer> create(gpu::ShaderLibrary& library);

    /// `points` are drawn as opaque discs in the same depth order as the
    /// splats. `under`, when given, is an opaque layer (a rasterised point
    /// layer of the same size) the splats are composited over and cut by.
    [[nodiscard]] Result<FrameStats> render(const Camera& camera,
                                            std::span<const SplatInstance> instances,
                                            const RenderSettings& settings, RenderTargets& targets,
                                            std::span<const PointInstance> points = {},
                                            const RenderTargets* under = nullptr);
    /// The same, from a projection someone else computed (a Hydra host).
    [[nodiscard]] Result<FrameStats> render(const Projection& projection,
                                            std::span<const SplatInstance> instances,
                                            const RenderSettings& settings, RenderTargets& targets,
                                            std::span<const PointInstance> points = {},
                                            const RenderTargets* under = nullptr);

private:
    [[nodiscard]] Result<void> reserveSplats(uint32_t count);
    [[nodiscard]] Result<void> reservePairs(uint32_t count);
    [[nodiscard]] Result<void> reserveTargets(RenderTargets& targets, uint32_t width,
                                              uint32_t height, uint32_t tiles);

    gpu::Device*       device_ = nullptr;
    gpu::PrefixSum     prefix_;
    gpu::RadixSort     sort_;
    gpu::ComputeKernel project_;
    gpu::ComputeKernel compact_;
    gpu::ComputeKernel pointsProject_;
    gpu::ComputeKernel gather_;
    gpu::ComputeKernel emit_;
    gpu::ComputeKernel clear_;
    gpu::ComputeKernel ranges_;
    gpu::ComputeKernel blend_;
    gpu::ComputeKernel blendComposite_;

    uint32_t splatCapacity_ = 0;
    uint32_t pairCapacity_ = 0;
    uint32_t tileCapacity_ = 0;
    gpu::Buffer proj_, tileRects_, tilesTouched_, visible_, depthKeys_;
    gpu::Buffer visibleOffsets_, visibleTotal_, touchedOffsets_, touchedTotal_;
    gpu::SortBuffers depthSort_;   // keysLo = visible depth keys, values = splat index
    gpu::Buffer sortedCounts_, offsets_, totalPairs_;
    gpu::SortBuffers tileSort_;    // keysLo = pair tiles, values = pair splats
    gpu::Buffer ranges_buffer_;
    gpu::Buffer placeholderColour_, placeholderDepth_;
};

}   // namespace lrt::render
