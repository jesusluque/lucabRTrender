// Copyright (c) 2026 lucabRTrender contributors.
//
// Skinning on the device: the inputs usdSkelImaging's ext computation
// carries -- rest points, blend shape offsets and their sub-shape weights,
// joint influences, skinning transforms as matrices or dual quaternions,
// and the bind and common-space transforms -- into skinned positions, one
// float4 a point, that a MeshBuilder takes as a mesh's points. Two readers
// feed it (Hydra's ext computation prims, and the deferred-skinning
// primvars) and neither does arithmetic on the host.
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::geom {

enum class SkinningMethod : uint32_t { LinearBlend = 0, DualQuaternion = 1 };

/// What skins a mesh, as usdSkelImaging lays it out. Arrays are byte spans
/// of what USD holds, uploaded as they are: float3 rest points, float4
/// offsets (xyz, w = sub-shape), int2 ranges a point, float weights a
/// sub-shape, float2 influences (joint, weight), 4x4 float matrices a joint
/// (row-major, as GfMatrix4f), (real, dual) float4 pairs a joint, 3x3 float
/// matrices a joint.
struct SkinningInput {
    uint32_t                   points = 0;
    std::span<const std::byte> restPoints;             ///< float3 a point
    std::span<const std::byte> blendShapeOffsets;      ///< float4 an offset
    std::span<const std::byte> blendShapeOffsetRanges; ///< int2 a point that has any
    std::span<const std::byte> blendShapeWeights;      ///< float a sub-shape
    std::span<const std::byte> influences;             ///< float2 (joint, weight)
    uint32_t                   numInfluencesPerPoint = 0;
    bool                       constantInfluences = false;
    SkinningMethod             method = SkinningMethod::LinearBlend;
    std::span<const std::byte> skinningXforms;         ///< 16 floats a joint
    std::span<const std::byte> skinningDualQuats;      ///< 8 floats a joint: real xyzw, dual xyzw
    std::span<const std::byte> skinningScaleXforms;    ///< 9 floats a joint; empty for none
    std::array<float, 16>      geomBindXform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};   ///< row-major
    std::array<float, 16>      skelLocalToWorld{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::array<float, 16>      primWorldToLocal{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

class Skinner {
public:
    [[nodiscard]] static Result<Skinner> create(gpu::ShaderLibrary& library);

    /// The skinned positions, float4 a point, in the prim's own space.
    [[nodiscard]] Result<gpu::Buffer> skin(const SkinningInput& input);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel kernel_;
    gpu::ComputeKernel points_;   ///< the builder's decode, float3 words to float4
};

}   // namespace lrt::geom
