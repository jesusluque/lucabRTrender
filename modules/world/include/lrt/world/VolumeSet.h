// Copyright (c) 2026 lucabRTrender contributors.
//
// The frame's volumes on the device, in one buffer of words: a header, a
// record a volume (grid offsets, world-to-index, extinction scale, albedo,
// phase, and -- written by kernels -- its index bounds and majorant), the
// NanoVDB grids, and each leaf's largest value. The path tracer reads it
// all from that one buffer (shaders/lrt/volume/medium.slang), since a
// Metal kernel binds at most 31 and the traced one stands at the limit.
// Nothing here reads a value on the host: the host places words, counts
// leaves and composes transforms.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/io/Vdb.h"
#include "lrt/render/Camera.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::world {

/// One volume: a grid, where it stands, and how it scatters.
struct VolumeInput {
    const io::NanoGrid*  grid = nullptr;
    render::Mat4         objectToWorld = render::Mat4::identity();   ///< the prim's; the grid's index-to-world composes under it
    float                densityScale = 1.0F;   ///< density to extinction, per world unit
    std::array<float, 3> albedo{1.0F, 1.0F, 1.0F};
    float                g = 0.0F;
};

/// Where one grid sits in the words, for a kernel that reads it.
struct VolumeGridPlacement {
    uint32_t gridWord = 0;
    uint32_t leafCount = 0;
    uint32_t leafBase = 0;   ///< word offset of its leaf maxima
};

class VolumeSet {
public:
    [[nodiscard]] static Result<VolumeSet> create(gpu::ShaderLibrary& library);

    /// Lays the volumes out and runs the bounds, leaf maximum and finish
    /// kernels over them.
    [[nodiscard]] Result<void> set(gpu::CommandBatch& batch, std::span<const VolumeInput> volumes);

    [[nodiscard]] uint32_t count() const noexcept { return static_cast<uint32_t>(placements_.size()); }
    [[nodiscard]] const VolumeGridPlacement& placement(uint32_t i) const noexcept { return placements_[i]; }
    /// The one buffer: medium.slang's layout.
    [[nodiscard]] const gpu::Buffer& words() const noexcept { return words_; }
    /// Six ints a grid: min xyz, max xyz (exclusive), in index space.
    [[nodiscard]] const gpu::Buffer& bounds() const noexcept { return bounds_; }
    /// A float's bits a grid: its largest leaf maximum.
    [[nodiscard]] const gpu::Buffer& majorant() const noexcept { return majorant_; }

    static constexpr uint32_t kHeaderWords = 8;
    static constexpr uint32_t kRecordWords = 32;

private:
    gpu::Device*                     device_ = nullptr;
    gpu::ComputeKernel               boundsKernel_, leafMaxKernel_, finishKernel_;
    gpu::Buffer                      words_, bounds_, majorant_;
    std::vector<VolumeGridPlacement> placements_;
};

}   // namespace lrt::world
