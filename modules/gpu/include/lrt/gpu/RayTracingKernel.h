// Copyright (c) 2026 lucabRTrender contributors.
//
// A ray tracing pipeline and its shader table (OptiX on CUDA, Vulkan RT). Metal
// has none -- slang-rhi gives it inline RayQuery in compute only -- so on Metal
// this reports Unsupported and the engine traces with ComputeKernels there.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "lrt/core/Result.h"

namespace lrt::gpu {

class CommandBatch;
class ShaderLibrary;
struct Program;

struct RayTracingDesc {
    struct HitGroup {
        std::string name;
        std::string closestHit;
        std::string anyHit;         ///< empty: none
        std::string intersection;   ///< empty: triangles
    };
    std::string              module;
    std::string              rayGen;
    std::vector<std::string> misses;
    std::vector<HitGroup>    hitGroups;
    uint32_t                 maxRecursion = 1;
    uint32_t                 payloadBytes = 64;
};

class RayTracingKernel {
public:
    RayTracingKernel() = default;

    [[nodiscard]] static Result<RayTracingKernel> create(ShaderLibrary& library, const RayTracingDesc& desc);

    using Bind = std::function<void(rhi::ShaderCursor)>;
    /// Queues the ray generation over width x height x depth into `batch`.
    void dispatch(CommandBatch& batch, uint32_t width, uint32_t height, uint32_t depth, const Bind& bind) const;

private:
    std::string                           name_;
    std::shared_ptr<const Program>        program_;
    rhi::ComPtr<rhi::IRayTracingPipeline> pipeline_;
    rhi::ComPtr<rhi::IShaderTable>        table_;
};

}   // namespace lrt::gpu
