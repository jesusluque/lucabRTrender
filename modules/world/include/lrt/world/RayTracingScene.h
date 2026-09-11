// Copyright (c) 2026 lucabRTrender contributors.
//
// A GpuScene as hardware acceleration structures: one bottom level per mesh,
// over the scene's own pools (no copies), rebuilt when the pools are; one top
// level per frame, whose instance descriptors a kernel writes from the
// instance records in the backend's layout -- instanced sets' transforms never
// come to the host.
#pragma once

#include <vector>

#include <slang-rhi.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/world/GpuScene.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::world {

class RayTracingScene {
public:
    /// Unsupported where the device has no acceleration structures.
    [[nodiscard]] static Result<RayTracingScene> create(gpu::ShaderLibrary& library);

    [[nodiscard]] Result<void> build(const GpuScene& scene);

    [[nodiscard]] rhi::IAccelerationStructure* topLevel() const noexcept { return topLevel_.get(); }

private:
    gpu::Device*                                          device_ = nullptr;
    gpu::ComputeKernel                                    descs_;
    uint64_t                                              generation_ = ~uint64_t{0};
    std::vector<rhi::ComPtr<rhi::IAccelerationStructure>> bottom_;
    rhi::ComPtr<rhi::IAccelerationStructure>              topLevel_;
    gpu::Buffer                                           handles_;
    gpu::Buffer                                           instanceDescs_;
};

}   // namespace lrt::world
