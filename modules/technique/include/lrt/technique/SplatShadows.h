// Copyright (c) 2026 lucabRTrender contributors.
//
// A splat cloud as something meshes are shadowed by.
//
// `GaussianRayTracer` already builds what the shadow query needs -- the
// particles' frames, their colours, each instance's world-to-cloud rows and
// its bases -- and `render::ShadowScene` hands them over. A kernel of its own
// can bind those four buffers (rt_shadow_kernel.slang does); the path tracer
// cannot, because it sits at Metal's limit of 31 buffers. So they are packed
// into one buffer on the device by `rtShadowPack`, and the query reads them
// back through the offsets (rt_shadow_packed.slang).
//
// Packed once a frame, before the pass that reads it: the tables change when
// the cloud, its edits or its instances do, which is what `prepare` is asked
// again for.
#pragma once

#include "lrt/gpu/ComputeKernel.h"
#include "lrt/render/GaussianRayTracer.h"

namespace lrt::technique {

/// Where each table starts in the packed buffer, in float4 entries: the
/// shader's `PackedShadow`, and what the kernel that reads it is told.
struct PackedShadowLayout {
    uint32_t frames = 0;
    uint32_t colours = 0;
    uint32_t instances = 0;
    uint32_t indices = 0;
    uint32_t instanceCount = 0;
};

class SplatShadows {
public:
    [[nodiscard]] static Result<SplatShadows> create(gpu::ShaderLibrary& library);

    /// Packs `scene`'s tables into one buffer, growing it where the cloud did.
    /// A scene with no instances, or one the route cannot shadow with, leaves
    /// this empty and `valid()` false: the frame then traces meshes alone.
    [[nodiscard]] Result<void> prepare(gpu::CommandBatch& batch, const render::ShadowScene& scene);

    [[nodiscard]] bool valid() const noexcept { return layout_.instanceCount > 0 && packed_.valid(); }
    [[nodiscard]] const gpu::Buffer& packed() const noexcept { return packed_; }
    [[nodiscard]] const PackedShadowLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] rhi::IAccelerationStructure* topLevel() const noexcept { return tlas_; }

private:
    gpu::Device*                  device_ = nullptr;
    std::optional<gpu::ComputeKernel> pack_;
    gpu::Buffer                   packed_;
    PackedShadowLayout            layout_;
    rhi::IAccelerationStructure*  tlas_ = nullptr;
};

}   // namespace lrt::technique
