// Copyright (c) 2026 lucabRTrender contributors.
//
// The frame's volumes on the device: every NanoVDB grid in one buffer, and
// beside it what kernels computed from them -- each grid's index bounds,
// each leaf's largest value, each grid's majorant. Nothing here reads a
// value on the host: the host places the grids' words and counts leaves.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/io/Vdb.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::world {

/// Where one grid sits in the set, for a kernel that reads it.
struct VolumeGridPlacement {
    uint32_t gridWord = 0;    ///< the grid's word offset in `words()`
    uint32_t leafCount = 0;
    uint32_t leafBase = 0;    ///< its first entry in `leafMax()`
};

class VolumeSet {
public:
    [[nodiscard]] static Result<VolumeSet> create(gpu::ShaderLibrary& library);

    /// Uploads the grids (each 32-byte aligned) and runs the bounds and
    /// leaf maximum kernels over them.
    [[nodiscard]] Result<void> set(gpu::CommandBatch& batch, std::span<const io::NanoGrid> grids);

    [[nodiscard]] uint32_t count() const noexcept { return static_cast<uint32_t>(placements_.size()); }
    [[nodiscard]] const VolumeGridPlacement& placement(uint32_t i) const noexcept { return placements_[i]; }
    [[nodiscard]] const gpu::Buffer& words() const noexcept { return words_; }
    /// Six ints a grid: min xyz, max xyz (exclusive), in index space.
    [[nodiscard]] const gpu::Buffer& bounds() const noexcept { return bounds_; }
    /// A float a leaf, in placement order.
    [[nodiscard]] const gpu::Buffer& leafMax() const noexcept { return leafMax_; }
    /// A float's bits a grid: its largest leaf maximum.
    [[nodiscard]] const gpu::Buffer& majorant() const noexcept { return majorant_; }

private:
    gpu::Device*                     device_ = nullptr;
    gpu::ComputeKernel               boundsKernel_, leafMaxKernel_;
    gpu::Buffer                      words_, bounds_, leafMax_, majorant_;
    std::vector<VolumeGridPlacement> placements_;
};

}   // namespace lrt::world
