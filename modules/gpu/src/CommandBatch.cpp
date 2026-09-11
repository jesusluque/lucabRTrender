// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/CommandBatch.h"

#include "lrt/gpu/Device.h"

namespace lrt::gpu {

CommandBatch::CommandBatch(Device& device)
    : device_(device), encoder_(device.queue()->createCommandEncoder()) {}

CommandBatch::~CommandBatch() {
    if (dirty_) {
        (void)submit(false);
    }
}

Result<void> CommandBatch::submit(bool wait) {
    if (dirty_) {
        if (SLANG_FAILED(device_.queue()->submit(encoder_->finish()))) {
            return Error(ErrorCode::DeviceFailure, "command submit failed");
        }
        encoder_ = device_.queue()->createCommandEncoder();
        dirty_ = false;
    }
    if (wait) {
        device_.queue()->waitOnHost();
    }
    return ok();
}

}   // namespace lrt::gpu
