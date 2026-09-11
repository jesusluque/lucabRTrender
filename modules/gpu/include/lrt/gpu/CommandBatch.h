// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <slang-rhi.h>

#include "lrt/core/Result.h"

namespace lrt::gpu {

class Device;

/// Commands recorded together and submitted once.
///
/// One per frame stage, not one per dispatch: a command buffer per dispatch is
/// the Metal cost gpe pays today and this engine does not.
class CommandBatch {
public:
    explicit CommandBatch(Device& device);
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    ~CommandBatch();

    [[nodiscard]] rhi::ICommandEncoder* encoder() noexcept { return encoder_.get(); }
    /// For callers that record straight into `encoder()` (copies, clears).
    void markDirty() noexcept { dirty_ = true; }

    /// Submits what was recorded and starts a fresh encoder. `wait` blocks
    /// until the device has run it.
    [[nodiscard]] Result<void> submit(bool wait = false);

private:
    Device&                           device_;
    rhi::ComPtr<rhi::ICommandEncoder> encoder_;
    bool                              dirty_ = false;
    friend class ComputeKernel;
};

}   // namespace lrt::gpu
