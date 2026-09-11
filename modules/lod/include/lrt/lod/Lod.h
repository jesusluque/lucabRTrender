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
// Only counts cross back to the CPU: one per level while building, and per
// instance per frame one read of the drawn counts (and, streaming, of the
// chunks the view wants).
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
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

/// A cloud with its levels of detail. The merged levels are always on the
/// device; the cloud's own splats come in chunks -- runs of `chunkSplats` of
/// the Morton order, the last holding the rest -- which need not be. A chunk
/// on the device sits in a slot of the store (slot k: splats from
/// k * chunkSplats); built in memory, chunk c is slot c and all are there.
struct LodCloud {
    uint32_t              count = 0;         ///< splats in the cloud
    uint32_t              chunkSplats = 0;   ///< splats per chunk
    scene::GpuSplats      splats;            ///< the store: chunks' splats in their slots
    gpu::Buffer           groups;            ///< uint per store splat: its finest-level group
    std::vector<int32_t>  slots;             ///< per chunk: its slot, or -1 when not on the device
    gpu::Buffer           resident;          ///< uint per chunk: 1 when on the device
    std::vector<LodLevel> levels;            ///< coarsest first, consecutive levels, never empty
    gpu::Buffer           starts;            ///< uint per finest-level group: its first splat
    float                 boundsLo[3] = {0, 0, 0};
    float                 extent = 0.0F;     ///< the largest edge of the bounds
    /// Chunks come and go (a StreamingPool's): the cut says which it wants.
    bool                  streamed = false;

    [[nodiscard]] uint32_t chunks() const noexcept { return static_cast<uint32_t>(slots.size()); }
    [[nodiscard]] uint32_t chunkCount(uint32_t chunk) const noexcept {
        return std::min(chunkSplats, count - chunk * chunkSplats);
    }
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
    /// Splats per chunk: what a stream loads and drops at once.
    uint32_t chunkSplats = uint32_t{1} << 16;
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
    const LodCloud*      cloud = nullptr;
    render::Mat4         objectToWorld = render::Mat4::identity();
    render::SplatEdit    edit;
    std::optional<float> threshold;   ///< its own, over the call's
};

struct CutStats {
    uint32_t splats = 0;     ///< the cloud's own splats drawn
    uint32_t merged = 0;     ///< merged Gaussians drawn
    uint32_t available = 0;  ///< splats in the cloud
    /// Streamed clouds: per chunk, 0 when this view does not want its splats,
    /// else how much -- the largest projected edge, in 1/16 pixels, of the
    /// cells wanting them -- on the device or not. Empty for a cloud that is
    /// all in memory.
    std::vector<uint32_t> needs;
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
    /// selector and are rewritten by the next call. `stats`, when given, holds
    /// one entry per instance, in order.
    [[nodiscard]] Result<std::vector<render::SplatInstance>> select(const render::Projection& projection,
                                                                   std::span<const LodInstance> instances,
                                                                   float threshold,
                                                                   std::vector<CutStats>* stats = nullptr);

private:
    CutSelector();
    struct Frame;
    gpu::Device*                        device_ = nullptr;
    gpu::PrefixSum                      prefix_;
    gpu::ComputeKernel                  cutGroups_, cutFinest_, cutSplats_, chunkNeeds_, gather_;
    std::vector<std::unique_ptr<Frame>> frames_;
};

}   // namespace lrt::lod
