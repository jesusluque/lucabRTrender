// Copyright (c) 2026 lucabRTrender contributors.
//
// Hydra instancers on the device: each level's transforms from its primvars,
// levels nested, every instance's transform left on the GPU for GpuScene's
// records. See shaders/lrt/world/instancing.slang for the composition, which
// is Storm's.
#pragma once

#include <cstdint>
#include <span>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/render/Camera.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::world {

/// One instancer, as it applies to one of its prototypes.
struct InstancerLevel {
    std::span<const int32_t> indices;        ///< the instancer's elements this prototype takes, in order
    scene::FloatStream       translations;   ///< xyz per element (float, half or double)
    scene::FloatStream       rotations;      ///< (ix, iy, iz, real) per element
    scene::FloatStream       scales;         ///< xyz per element
    scene::FloatStream       transforms;     ///< 16 per element, row-major (GfMatrix4d or 4f)
    render::Mat4             instancerTransform = render::Mat4::identity();
};

/// Every instance's transform, before the prototype's own: 3 float4 rows each.
struct InstanceChain {
    gpu::Buffer rows;
    uint32_t    count = 0;
};

class Instancing {
public:
    [[nodiscard]] static Result<Instancing> create(gpu::ShaderLibrary& library);

    /// `levels` innermost first: the instancer holding the prototype, then its
    /// parent, and so on. Instance k = parent i * inner + level j.
    [[nodiscard]] Result<InstanceChain> compose(std::span<const InstancerLevel> levels);

private:
    [[nodiscard]] Result<InstanceChain> level(const InstancerLevel& level);

    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel level_, compose_;
};

}   // namespace lrt::world
