// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/algo/RadixSort.h"

#include <cstdlib>
#include <utility>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::gpu {
namespace {

constexpr uint32_t kDigits = 256;

void setParams(rhi::ShaderCursor cursor, uint32_t count, uint32_t chunks, uint32_t shift,
               bool wide) {
    cursor["params"]["count"].setData(count);
    cursor["params"]["chunkSize"].setData(RadixSort::chunkFor(count));
    cursor["params"]["chunkCount"].setData(chunks);
    cursor["params"]["shift"].setData(shift);
    cursor["params"]["wide"].setData(uint32_t{wide ? 1u : 0u});
}

}   // namespace

Result<RadixSort> RadixSort::create(ShaderLibrary& library) {
    RadixSort sort;
    sort.device_ = &library.device();
    auto histogram = ComputeKernel::create(library, "lrt/algo/radix_histogram", "radixHistogram");
    if (!histogram) return std::move(histogram).error();
    auto totals = ComputeKernel::create(library, "lrt/algo/radix_totals", "radixTotals");
    if (!totals) return std::move(totals).error();
    auto starts = ComputeKernel::create(library, "lrt/algo/radix_starts", "radixStarts");
    if (!starts) return std::move(starts).error();
    auto scatter = ComputeKernel::create(library, "lrt/algo/radix_scatter", "radixScatter");
    if (!scatter) return std::move(scatter).error();
    sort.histogram_ = std::move(*histogram);
    sort.totals_ = std::move(*totals);
    sort.starts_ = std::move(*starts);
    sort.scatter_ = std::move(*scatter);

    BufferDesc desc;
    desc.bytes = kDigits * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "radix.digitTotals";
    auto totalsBuffer = Buffer::create(*sort.device_, desc);
    if (!totalsBuffer) return std::move(totalsBuffer).error();
    sort.digitTotals_ = *totalsBuffer;
    desc.bytes = sizeof(uint32_t);
    desc.label = "radix.unused";
    auto dummy = Buffer::create(*sort.device_, desc);
    if (!dummy) return std::move(dummy).error();
    sort.dummy_ = *dummy;
    return sort;
}

Result<void> RadixSort::reserve(uint32_t chunks) {
    if (chunks <= capacity_) {
        return ok();
    }
    uint32_t grown = capacity_ == 0 ? 16 : capacity_;
    while (grown < chunks) {
        grown *= 2;
    }
    BufferDesc desc;
    desc.bytes = uint64_t{grown} * kDigits * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "radix.histogram";
    auto histogram = Buffer::create(*device_, desc);
    if (!histogram) return std::move(histogram).error();
    desc.label = "radix.chunkStarts";
    auto starts = Buffer::create(*device_, desc);
    if (!starts) return std::move(starts).error();
    histogramBuffer_ = *histogram;
    chunkStarts_ = *starts;
    capacity_ = grown;
    return ok();
}

Result<void> RadixSort::sort(CommandBatch& batch, SortBuffers& buffers, uint32_t count,
                             uint32_t keyBits) {
    if (keyBits == 0 || keyBits > 64) {
        return Error::make(ErrorCode::InvalidArgument, "cannot sort on {} key bits", keyBits);
    }
    const bool wide = keyBits > 32;
    if (buffers.keysLo.count() < count || buffers.values.count() < count ||
        buffers.scratchKeysLo.count() < count || buffers.scratchValues.count() < count ||
        (wide && (buffers.keysHi.count() < count || buffers.scratchKeysHi.count() < count))) {
        return Error(ErrorCode::InvalidArgument, "radix sort buffers are too small");
    }
    if (count <= 1) {
        return ok();
    }
    const uint32_t chunk = chunkFor(count);
    const uint32_t chunks = (count + chunk - 1) / chunk;
    LRT_TRY(reserve(chunks));
    chunks_ = chunks;

    Buffer* srcLo = &buffers.keysLo;
    Buffer* srcHi = wide ? &buffers.keysHi : &dummy_;
    Buffer* srcVal = &buffers.values;
    Buffer* dstLo = &buffers.scratchKeysLo;
    Buffer* dstHi = wide ? &buffers.scratchKeysHi : &dummy_;
    Buffer* dstVal = &buffers.scratchValues;

    const uint32_t passes = passesFor(keyBits);
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t shift = pass * 8;
        histogram_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["keysLo"].setBinding(srcLo->rhi());
            cursor["keysHi"].setBinding(srcHi->rhi());
            cursor["histogram"].setBinding(histogramBuffer_.rhi());
            setParams(cursor, count, chunks, shift, wide);
        });
        totals_.dispatch(batch, {kDigits, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["histogram"].setBinding(histogramBuffer_.rhi());
            cursor["digitTotals"].setBinding(digitTotals_.rhi());
            setParams(cursor, count, chunks, shift, wide);
        });
        starts_.dispatch(batch, {kDigits, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["histogram"].setBinding(histogramBuffer_.rhi());
            cursor["digitTotals"].setBinding(digitTotals_.rhi());
            cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
            setParams(cursor, count, chunks, shift, wide);
        });
        // Distinct placeholders for the unused hi bindings: a buffer bound as
        // both read-only and read-write in one dispatch is refused by D3D and
        // Vulkan validation even when the kernel never touches it.
        Buffer* scatterSrcHi = wide ? srcHi : &dummy_;
        Buffer* scatterDstHi = wide ? dstHi : &histogramBuffer_;
        scatter_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["srcKeysLo"].setBinding(srcLo->rhi());
            cursor["srcKeysHi"].setBinding(scatterSrcHi->rhi());
            cursor["srcValues"].setBinding(srcVal->rhi());
            cursor["dstKeysLo"].setBinding(dstLo->rhi());
            cursor["dstKeysHi"].setBinding(scatterDstHi->rhi());
            cursor["dstValues"].setBinding(dstVal->rhi());
            cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
            setParams(cursor, count, chunks, shift, wide);
        });
        if (std::getenv("LRT_RADIX_SUBMIT_EACH_PASS") != nullptr) {
            LRT_TRY(batch.submit(true));
        }
        std::swap(srcLo, dstLo);
        std::swap(srcHi, dstHi);
        std::swap(srcVal, dstVal);
    }

    if (passes % 2 == 1) {
        // The last pass wrote the scratch buffers; bring the result home.
        rhi::ICommandEncoder* encoder = batch.encoder();
        encoder->copyBuffer(buffers.keysLo.rhi(), 0, srcLo->rhi(), 0, uint64_t{count} * 4);
        encoder->copyBuffer(buffers.values.rhi(), 0, srcVal->rhi(), 0, uint64_t{count} * 4);
        if (wide) {
            encoder->copyBuffer(buffers.keysHi.rhi(), 0, srcHi->rhi(), 0, uint64_t{count} * 4);
        }
        batch.markDirty();
    }
    return ok();
}

}   // namespace lrt::gpu
