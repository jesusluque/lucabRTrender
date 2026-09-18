// Copyright (c) 2026 lucabRTrender contributors.
//
// A cloud carried by a skeleton: the gaussians are built once in the bind
// pose, each keeps the joints that move it, and this puts them where the
// skeleton is at an instant. The counterpart of `geom::Skinner`, which does
// the same for a mesh's points -- and it takes the same inputs, in the same
// layout, because they come from the same place.
#pragma once

#include <array>
#include <cstdint>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::scene {

struct GpuSplats;

/// What moves a cloud: the joints each gaussian is held by, the skeleton's
/// transforms at this instant, and the three matrices that take a point from
/// the cloud's own space to the skeleton's and back.
struct SplatSkinInput {
    const GpuSplats*   rest = nullptr;
    /// `(joint, weight)` float2s, `perSplat` of them a gaussian.
    const gpu::Buffer* influences = nullptr;
    uint32_t           perSplat = 4;
    /// Sixteen floats a joint, row major as `GfMatrix4f` holds them; they go
    /// over transposed, exactly as `geom::Skinner` sends a mesh's.
    const gpu::Buffer* skinningXforms = nullptr;
    std::array<float, 16> geomBindTransform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 16> skelLocalToWorld{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                           0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 16> primWorldToLocal{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                           0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
};

class SplatSkinner {
public:
    [[nodiscard]] static Result<SplatSkinner> create(gpu::ShaderLibrary& library);

    /// Writes the posed positions and shapes into buffers of the rest cloud's
    /// own size. Everything else a gaussian carries -- its opacity, its
    /// colour, its harmonics, its PBR channels -- is the bind pose's and is
    /// not touched, which is what makes a frame cost one kernel.
    [[nodiscard]] Result<void> skin(gpu::CommandBatch& batch, const SplatSkinInput& input,
                                    gpu::Buffer& positions, gpu::Buffer& shape);

private:
    gpu::ComputeKernel kernel_;
};

}   // namespace lrt::scene
