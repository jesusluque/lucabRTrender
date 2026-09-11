// Copyright (c) 2026 lucabRTrender contributors.
//
// Level of detail for splat clouds, built and cut on the device. The method --
// Morton-ordered cells, additive moments, a per-cell cut that covers every
// place exactly once -- is written out in shaders/lrt/lod/lod_common.slang.
//
//   LodBuilder     once per cloud: Morton sort, the levels' groups, moments
//                  from the finest merged level up, one Gaussian a group
//   CutSelector    every frame, per instance: which groups of which levels and
//                  which splats to draw, gathered into one cloud the tile
//                  rasteriser draws as it would any other
//
// Only counts cross back to the CPU: one per level while building, and one per
// level per instance per frame, read together.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/PrefixSum.h"
#include "lrt/gpu/algo/RadixSort.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::lod {

/// One stored level: a merged Gaussian per octree cell that holds splats.
struct LodLevel {
    uint32_t          level = 0;   ///< octree level: cells of extent / 2^level
    scene::GpuSplats  gaussians;   ///< one per group
    gpu::Buffer       cells;       ///< uint per group: the cell's code at `level`
};

struct LodCloud {
    scene::GpuSplats      splats;   ///< the cloud's own splats, in Morton order
    gpu::Buffer           keys;     ///< uint per splat: its 30-bit Morton code
    std::vector<LodLevel> levels;   ///< coarsest first, consecutive levels
    float                 boundsLo[3] = {0, 0, 0};
    float                 extent = 0.0F;   ///< the largest edge of the bounds

    [[nodiscard]] uint32_t mergedGaussians() const noexcept {
        uint32_t n = 0;
        for (const LodLevel& l : levels) {
            n += l.gaussians.count;
        }
        return n;
    }
};

struct LodBuildSettings {
    /// The finest stored level is the deepest whose cells hold on average at
    /// least 1 / this fraction of splats each: deeper, a merged level would
    /// be nearly the splats again.
    float maxGroupFraction = 0.5F;
    /// The coarsest stored level.
    uint32_t coarsestLevel = 1;
};

class LodBuilder {
public:
    [[nodiscard]] static Result<LodBuilder> create(gpu::ShaderLibrary& library);
    [[nodiscard]] Result<LodCloud> build(const scene::GpuSplats& cloud, const LodBuildSettings& settings = {});

private:
    gpu::Device*       device_ = nullptr;
    gpu::RadixSort     sort_;
    gpu::PrefixSum     prefix_;
    gpu::ComputeKernel morton_, reorder_, boundaries_, groups_, leafMoments_, mergeMoments_, finalize_;
};

struct LodInstance {
    const LodCloud*   cloud = nullptr;
    render::Mat4      objectToWorld = render::Mat4::identity();
    render::SplatEdit edit;
};

struct CutStats {
    uint32_t splats = 0;     ///< the cloud's own splats drawn
    uint32_t merged = 0;     ///< merged Gaussians drawn
    uint32_t available = 0;  ///< splats in the cloud
};

class CutSelector {
public:
    [[nodiscard]] static Result<CutSelector> create(gpu::ShaderLibrary& library);
    CutSelector(CutSelector&&) noexcept;
    CutSelector& operator=(CutSelector&&) noexcept;
    ~CutSelector();

    /// Each instance's cut for this view, as instances the rasteriser draws.
    /// A cell is drawn merged when it projects to at most `threshold` pixels;
    /// 0 draws every splat as it is. The returned clouds belong to this
    /// selector and are rewritten by the next call.
    [[nodiscard]] Result<std::vector<render::SplatInstance>> select(const render::Projection& projection,
                                                                   std::span<const LodInstance> instances,
                                                                   float threshold,
                                                                   std::vector<CutStats>* stats = nullptr);

private:
    CutSelector();
    struct Frame;
    gpu::Device*                        device_ = nullptr;
    gpu::PrefixSum                      prefix_;
    gpu::ComputeKernel                  cutGroups_, cutSplats_, gather_;
    std::vector<std::unique_ptr<Frame>> frames_;
};

}   // namespace lrt::lod
