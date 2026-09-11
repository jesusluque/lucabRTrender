// Copyright (c) 2026 lucabRTrender contributors.
//
// Exclusive prefix sum over uint32, on the GPU, in three dispatches.
// See shaders/lrt/algo/prefix_chunk_totals.slang for why chunks.
#pragma once

#include <cstdint>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::gpu {

class CommandBatch;
class ShaderLibrary;

class PrefixSum {
public:
    /// Elements per chunk for `count` elements.
    ///
    /// Each chunk is one GPU invocation walking its elements in order, so the
    /// chunk count is the parallelism. A fixed 4096 gave a 741k-splat depth
    /// sort 182 invocations -- 8 ms, over twice the per-element cost of ten
    /// million. Smaller chunks for smaller inputs, bounded below where a
    /// histogram row (256 digits) would cost more than the chunk it counts.
    [[nodiscard]] static uint32_t chunkFor(uint32_t count) noexcept {
        uint32_t chunk = kMinChunk;
        while (chunk < kMaxChunk && count / chunk > kTargetChunks) {
            chunk *= 2;
        }
        return chunk;
    }
    static constexpr uint32_t kMinChunk = 256;
    static constexpr uint32_t kMaxChunk = 4096;
    static constexpr uint32_t kTargetChunks = 2048;

    [[nodiscard]] static Result<PrefixSum> create(ShaderLibrary& library);

    /// output[i] = input[0] + ... + input[i-1]; total[0] = the sum of all.
    /// `input` and `output` must be different buffers; `total` holds at least
    /// one uint. Queued into `batch`, not waited for.
    [[nodiscard]] Result<void> apply(CommandBatch& batch, const Buffer& input,
                                     const Buffer& output, const Buffer& total, uint32_t count);

private:
    [[nodiscard]] Result<void> reserve(uint32_t chunks);

    Device*       device_ = nullptr;
    ComputeKernel totals_;
    ComputeKernel starts_;
    ComputeKernel local_;
    Buffer        chunkTotals_;
    Buffer        chunkStarts_;
    uint32_t      capacity_ = 0;
};

}   // namespace lrt::gpu
