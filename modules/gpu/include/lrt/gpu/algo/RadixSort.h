// Copyright (c) 2026 lucabRTrender contributors.
//
// Stable LSD radix sort of (key, value) pairs on the GPU: 32- or 64-bit keys,
// eight bits a pass, portable across every backend slang-rhi has.
// See shaders/lrt/algo/radix_common.slang for the algorithm and why it looks
// the way it does.
#pragma once

#include <cstdint>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::gpu {

class CommandBatch;
class ShaderLibrary;

/// The pairs being sorted, and the same-sized buffers the passes alternate
/// with. `keysHi`/`scratchKeysHi` only for 64-bit keys.
struct SortBuffers {
    Buffer keysLo;
    Buffer keysHi;
    Buffer values;
    Buffer scratchKeysLo;
    Buffer scratchKeysHi;
    Buffer scratchValues;
};

class RadixSort {
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

    [[nodiscard]] static Result<RadixSort> create(ShaderLibrary& library);

    /// Sorts the first `count` pairs by their low `keyBits` bits (1..64).
    /// The result is always back in `keysLo`/`keysHi`/`values`: an odd number
    /// of passes ends with a copy. Queued into `batch`, not waited for.
    [[nodiscard]] Result<void> sort(CommandBatch& batch, SortBuffers& buffers, uint32_t count,
                                    uint32_t keyBits);

    [[nodiscard]] static uint32_t passesFor(uint32_t keyBits) noexcept {
        return (keyBits + 7) / 8;
    }

private:
    [[nodiscard]] Result<void> reserve(uint32_t chunks);

    Device*       device_ = nullptr;
    ComputeKernel histogram_;
    ComputeKernel totals_;
    ComputeKernel starts_;
    ComputeKernel scatter_;
    Buffer        histogramBuffer_;
    Buffer        digitTotals_;
    Buffer        chunkStarts_;
    Buffer        dummy_;
    uint32_t      capacity_ = 0;
};

}   // namespace lrt::gpu
