// Copyright (c) 2026 lucabRTrender contributors.
//
// Intel Open Image Denoise on the engine's own device: its Metal command queue
// (or CUDA stream), images that are the engine's buffers. Built from source
// with no CPU device (scripts/build-oidn.sh), so where OIDN cannot run on the
// GPU there is no denoiser at all -- never a CPU one.
#pragma once

#include <memory>
#include <string>

#include "lrt/core/Result.h"

namespace lrt::gpu {
class Device;
}

namespace lrt::technique {

class Denoiser {
public:
    [[nodiscard]] static Result<Denoiser> create(gpu::Device& device);

    /// "OIDN 2.5.1 on Metal", for `lrt info` and logs.
    [[nodiscard]] const std::string& description() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/// Whether this build carries OIDN at all.
[[nodiscard]] bool denoiserBuilt() noexcept;

}   // namespace lrt::technique
