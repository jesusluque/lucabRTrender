// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::gpu {

class CommandBatch;

/// One compute entry point, as a pipeline.
///
/// Bound by name through reflection -- `cursor["splats"].setBinding(buffer)` --
/// never by slot: counting slots is how openFXplayer's D30 bugs happened.
class ComputeKernel {
public:
    using Bind = std::function<void(rhi::ShaderCursor)>;

    [[nodiscard]] static Result<ComputeKernel> create(ShaderLibrary& library,
                                                      const std::string& module,
                                                      const std::string& entry);

    ComputeKernel() = default;

    /// Queues the kernel over `threads` (in threads, not groups) into `batch`.
    /// The group count is threads / [numthreads], rounded up, on every
    /// backend; the kernel bounds-checks.
    void dispatch(CommandBatch& batch, std::array<uint32_t, 3> threads,
                  const Bind& bind) const;

    [[nodiscard]] const std::array<uint32_t, 3>& threadGroup() const noexcept {
        return program_->threadGroup;
    }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    std::string                           name_;
    std::shared_ptr<const Program>        program_;
    rhi::ComPtr<rhi::IComputePipeline>    pipeline_;
};

[[nodiscard]] constexpr uint32_t groupsFor(uint32_t threads, uint32_t perGroup) noexcept {
    return perGroup == 0 ? 0 : (threads + perGroup - 1) / perGroup;
}

}   // namespace lrt::gpu
