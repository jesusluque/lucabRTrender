// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/algo/PrefixSum.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::gpu {
namespace {

void setParams(rhi::ShaderCursor cursor, uint32_t count, uint32_t chunks) {
    cursor["params"]["count"].setData(count);
    cursor["params"]["chunkSize"].setData(PrefixSum::chunkFor(count));
    cursor["params"]["chunkCount"].setData(chunks);
}

}   // namespace

Result<PrefixSum> PrefixSum::create(ShaderLibrary& library) {
    PrefixSum sum;
    sum.device_ = &library.device();
    auto totals = ComputeKernel::create(library, "lrt/algo/prefix_chunk_totals", "prefixChunkTotals");
    if (!totals) return std::move(totals).error();
    auto starts = ComputeKernel::create(library, "lrt/algo/prefix_chunk_starts", "prefixChunkStarts");
    if (!starts) return std::move(starts).error();
    auto local = ComputeKernel::create(library, "lrt/algo/prefix_local", "prefixLocal");
    if (!local) return std::move(local).error();
    sum.totals_ = std::move(*totals);
    sum.starts_ = std::move(*starts);
    sum.local_ = std::move(*local);
    return sum;
}

Result<void> PrefixSum::reserve(uint32_t chunks) {
    if (chunks <= capacity_) {
        return ok();
    }
    // Geometric, so a frame that grows by a splat does not reallocate.
    uint32_t grown = capacity_ == 0 ? 16 : capacity_;
    while (grown < chunks) {
        grown *= 2;
    }
    BufferDesc desc;
    desc.bytes = uint64_t{grown} * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "prefix.chunkTotals";
    auto totals = Buffer::create(*device_, desc);
    if (!totals) return std::move(totals).error();
    desc.label = "prefix.chunkStarts";
    auto starts = Buffer::create(*device_, desc);
    if (!starts) return std::move(starts).error();
    chunkTotals_ = *totals;
    chunkStarts_ = *starts;
    capacity_ = grown;
    return ok();
}

Result<void> PrefixSum::apply(CommandBatch& batch, const Buffer& input, const Buffer& output,
                              const Buffer& total, uint32_t count) {
    if (input.rhi() == output.rhi()) {
        return Error(ErrorCode::InvalidArgument, "prefix sum input and output must differ");
    }
    if (input.count() < count || output.count() < count || total.count() < 1) {
        return Error(ErrorCode::InvalidArgument, "prefix sum buffers are too small");
    }
    const uint32_t chunk = chunkFor(count);
    const uint32_t chunks = count == 0 ? 1 : (count + chunk - 1) / chunk;
    LRT_TRY(reserve(chunks));

    totals_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["input"].setBinding(input.rhi());
        cursor["chunkTotals"].setBinding(chunkTotals_.rhi());
        setParams(cursor, count, chunks);
    });
    starts_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["chunkTotals"].setBinding(chunkTotals_.rhi());
        cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
        cursor["total"].setBinding(total.rhi());
        setParams(cursor, count, chunks);
    });
    local_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["input"].setBinding(input.rhi());
        cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
        cursor["output"].setBinding(output.rhi());
        setParams(cursor, count, chunks);
    });
    return ok();
}

}   // namespace lrt::gpu
