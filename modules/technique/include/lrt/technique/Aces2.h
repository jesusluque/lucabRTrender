// Copyright (c) 2026 lucabRTrender contributors.
//
// The ACES 2.0 output transform's parameters and tables on the device, for
// one peak luminance and one set of limiting primaries: what the
// reference computes once at init, built by kernels (aces2_prepare.slang)
// into two buffers the display kernel and the checks read.
#pragma once

#include <cstdint>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::technique {

/// The display's primaries the transform limits to.
enum class Aces2Limiting : uint32_t { Rec709 = 0, P3D65 = 1 };

class Aces2Tables {
public:
    [[nodiscard]] static Result<Aces2Tables> create(gpu::ShaderLibrary& library);

    /// Builds the tables for `peakLuminance` nits (100 for SDR) and the
    /// limiting primaries, unless they are already built for those.
    [[nodiscard]] Result<void> prepare(gpu::CommandBatch& batch, float peakLuminance, Aces2Limiting limiting);
    /// Whether `prepare` has built anything at all.
    [[nodiscard]] bool ready() const noexcept { return ready_; }
    [[nodiscard]] float peakLuminance() const noexcept { return peakLuminance_; }
    [[nodiscard]] Aces2Limiting limiting() const noexcept { return limiting_; }

    /// One Aces2Params (aces2.slang), and kTableWords floats.
    [[nodiscard]] const gpu::Buffer& params() const noexcept { return params_; }
    [[nodiscard]] const gpu::Buffer& tables() const noexcept { return tables_; }

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel paramsKernel_, tablesKernel_, wrapKernel_;
    gpu::Buffer        params_, tables_;
    bool               ready_ = false;
    float              peakLuminance_ = 0.0F;
    Aces2Limiting      limiting_ = Aces2Limiting::Rec709;
};

}   // namespace lrt::technique
