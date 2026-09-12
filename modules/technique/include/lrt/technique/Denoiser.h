// Copyright (c) 2026 lucabRTrender contributors.
//
// Intel Open Image Denoise on the engine's own device: its Metal command queue
// (or CUDA stream), images that are the engine's buffers. Built from source
// with no CPU device (scripts/build-oidn.sh), so where OIDN cannot run on the
// GPU there is no denoiser at all -- never a CPU one.
#pragma once

#include <memory>
#include <string>

#include <cstdint>

#include "lrt/core/Result.h"

namespace lrt::gpu {
class Buffer;
class Device;
class ShaderLibrary;
}

namespace lrt::technique {

class Denoiser {
public:
    /// With a library, a denoiser that can run: on Metal its staging copies
    /// are a kernel. With a device alone, enough to say what it is (lrt info).
    [[nodiscard]] static Result<Denoiser> create(gpu::ShaderLibrary& library);
    [[nodiscard]] static Result<Denoiser> create(gpu::Device& device);

    /// "OIDN 2.5.1 on Metal", for `lrt info` and logs.
    [[nodiscard]] const std::string& description() const noexcept;

    /// Denoises `colour` (float4 a pixel, linear radiance) into `out` on the
    /// device, guided by `albedo` and `normal` where given (float4 a pixel,
    /// the path tracer's PathAux). The images are the engine's own buffers,
    /// shared with OIDN rather than copied; nothing touches the host. Waits
    /// for the filter, so the caller's earlier work must be submitted.
    /// `aux`, when given, holds the albedo's plane at `albedoOffsetBytes` and
    /// the normal's at `normalOffsetBytes` (PathAux's one buffer).
    [[nodiscard]] Result<void> denoise(const gpu::Buffer& colour, const gpu::Buffer* aux,
                                       uint64_t albedoOffsetBytes, uint64_t normalOffsetBytes, gpu::Buffer& out,
                                       uint32_t width, uint32_t height);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/// Whether this build carries OIDN at all.
[[nodiscard]] bool denoiserBuilt() noexcept;

}   // namespace lrt::technique
