// Copyright (c) 2026 lucabRTrender contributors.
//
// The GPU prefix sum and radix sort, checked on the GPU.
#include "GpuTest.h"

#include <chrono>
#include <cstdio>

#include "lrt/gpu/algo/PrefixSum.h"
#include "lrt/gpu/algo/RadixSort.h"

using namespace lrt;

namespace {

constexpr uint32_t kChunk = 4096;

uint32_t chunksFor(uint32_t count) { return count == 0 ? 1 : (count + kChunk - 1) / kChunk; }

std::array<uint32_t, 8> statsOf(test::Gpu& gpu, const gpu::SortBuffers& buffers, uint32_t count,
                                bool wide) {
    static gpu::ComputeKernel kStats = test::kernel(gpu, "lrt/test/sort_stats");
    const uint32_t chunks = chunksFor(count);
    gpu::Buffer stats = test::uintBuffer(*gpu.device, uint64_t{chunks} * 8, "sort.stats");
    gpu::CommandBatch batch(*gpu.device);
    kStats.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["keysLo"].setBinding(buffers.keysLo.rhi());
        cursor["keysHi"].setBinding(buffers.keysHi.rhi());
        cursor["values"].setBinding(buffers.values.rhi());
        cursor["stats"].setBinding(stats.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["chunkSize"].setData(kChunk);
        cursor["params"]["chunkCount"].setData(chunks);
        cursor["params"]["wide"].setData(uint32_t{wide ? 1u : 0u});
    });
    REQUIRE(batch.submit(true));
    return test::reduceStats(gpu, stats, chunks);
}

gpu::SortBuffers sortBuffers(gpu::Device& device, uint32_t count) {
    gpu::SortBuffers buffers;
    buffers.keysLo = test::uintBuffer(device, count, "keysLo");
    buffers.keysHi = test::uintBuffer(device, count, "keysHi");
    buffers.values = test::uintBuffer(device, count, "values");
    buffers.scratchKeysLo = test::uintBuffer(device, count, "scratchKeysLo");
    buffers.scratchKeysHi = test::uintBuffer(device, count, "scratchKeysHi");
    buffers.scratchValues = test::uintBuffer(device, count, "scratchValues");
    return buffers;
}

void generate(test::Gpu& gpu, gpu::SortBuffers& buffers, uint32_t count, uint32_t mode,
              uint32_t keyBits) {
    static gpu::ComputeKernel kGenerate = test::kernel(gpu, "lrt/test/sort_generate");
    const uint32_t maskLo = keyBits >= 32 ? 0xFFFFFFFFu : ((1u << keyBits) - 1u);
    const uint32_t maskHi = keyBits <= 32 ? 0u : keyBits >= 64 ? 0xFFFFFFFFu : ((1u << (keyBits - 32)) - 1u);
    gpu::CommandBatch batch(*gpu.device);
    kGenerate.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["keysLo"].setBinding(buffers.keysLo.rhi());
        cursor["keysHi"].setBinding(buffers.keysHi.rhi());
        cursor["values"].setBinding(buffers.values.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["mode"].setData(mode);
        cursor["params"]["seed"].setData(uint32_t{0x9E3779B9u ^ count ^ (mode << 8)});
        cursor["params"]["keyMaskLo"].setData(maskLo);
        cursor["params"]["keyMaskHi"].setData(maskHi);
    });
    REQUIRE(batch.submit(true));
}

}   // namespace

TEST_CASE("the prefix sum of every size is right, checked element by element on the GPU",
          "[gpu][algo]") {
    LRT_REQUIRE_GPU(gpu);
    auto prefix = gpu::PrefixSum::create(*gpu->library);
    if (!prefix) {
        FAIL(prefix.error().toString());
    }
    gpu::ComputeKernel generate = test::kernel(*gpu, "lrt/test/counts_generate");
    gpu::ComputeKernel check = test::kernel(*gpu, "lrt/test/prefix_check");

    for (const uint32_t count : {1u, 7u, 4095u, 4096u, 4097u, 100003u, 1u << 22}) {
        CAPTURE(count);
        gpu::Buffer input = test::uintBuffer(*gpu->device, count, "counts");
        gpu::Buffer output = test::uintBuffer(*gpu->device, count, "offsets");
        gpu::Buffer total = test::uintBuffer(*gpu->device, 1, "total");
        const uint32_t chunks = chunksFor(count);
        gpu::Buffer errors = test::uintBuffer(*gpu->device, uint64_t{chunks} * 8, "errors");

        gpu::CommandBatch batch(*gpu->device);
        generate.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["counts"].setBinding(input.rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["seed"].setData(uint32_t{1234});
        });
        REQUIRE(prefix->apply(batch, input, output, total, count));
        check.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["input"].setBinding(input.rhi());
            cursor["output"].setBinding(output.rhi());
            cursor["total"].setBinding(total.rhi());
            cursor["errors"].setBinding(errors.rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["chunkSize"].setData(kChunk);
            cursor["params"]["chunkCount"].setData(chunks);
        });
        REQUIRE(batch.submit(true));
        CHECK(test::reduceStats(*gpu, errors, chunks)[0] == 0);
    }
}

/// The pairs themselves, before and after, for the smallest sort there is.
/// When a backend gets the sort wrong, this says which stage wrote what: the
/// generator's two pairs, then what the sort left in their place.
TEST_CASE("two pairs sort into one order, and are the pairs that went in", "[gpu][algo][probe]") {
    LRT_REQUIRE_GPU(gpu);
    auto sort = gpu::RadixSort::create(*gpu->library);
    if (!sort) FAIL(sort.error().toString());
    static gpu::ComputeKernel kDump = test::kernel(*gpu, "lrt/test/sort_dump");
    constexpr uint32_t kPairs = 2;
    gpu::SortBuffers buffers = sortBuffers(*gpu->device, kPairs);
    generate(*gpu, buffers, kPairs, 0, 32);
    const auto dump = [&](const char* when) {
        gpu::Buffer out = test::uintBuffer(*gpu->device, kPairs * 3, "sort.dump");
        gpu::CommandBatch batch(*gpu->device);
        kDump.dispatch(batch, {kPairs, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["keysLo"].setBinding(buffers.keysLo.rhi());
            cursor["keysHi"].setBinding(buffers.keysHi.rhi());
            cursor["values"].setBinding(buffers.values.rhi());
            cursor["dump"].setBinding(out.rhi());
            cursor["params"]["count"].setData(kPairs);
        });
        REQUIRE(batch.submit(true));
        std::array<uint32_t, kPairs * 3> words{};
        REQUIRE(out.read(*gpu->device, 0, sizeof(words), words.data()));
        std::printf("  %s: (key %u, value %u) (key %u, value %u)\n", when, words[0], words[2], words[3], words[5]);
        return words;
    };
    const auto before = dump("generated");
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(sort->sort(batch, buffers, kPairs, 32));
        REQUIRE(batch.submit(true));
    }
    const auto after = dump("sorted   ");
    CHECK(after[0] <= after[3]);   // ordered
    const uint64_t inKeys = uint64_t{before[0]} + before[3];
    const uint64_t outKeys = uint64_t{after[0]} + after[3];
    const uint64_t inValues = uint64_t{before[2]} + before[5];
    const uint64_t outValues = uint64_t{after[2]} + after[5];
    CHECK(outKeys == inKeys);       // the same two keys
    CHECK(outValues == inValues);   // carrying the same two values
}

TEST_CASE("the radix sort orders, keeps ties stable and loses nothing, for every pattern",
          "[gpu][algo]") {
    LRT_REQUIRE_GPU(gpu);
    auto sort = gpu::RadixSort::create(*gpu->library);
    if (!sort) {
        FAIL(sort.error().toString());
    }
    struct Case {
        uint32_t count;
        uint32_t keyBits;
    };
    for (const Case c : {Case{1, 32}, Case{2, 32}, Case{4097, 13}, Case{65537, 32},
                         Case{300007, 24}, Case{250001, 64}, Case{100000, 40}}) {
        for (uint32_t mode = 0; mode < 5; ++mode) {
            CAPTURE(c.count, c.keyBits, mode);
            const bool wide = c.keyBits > 32;
            gpu::SortBuffers buffers = sortBuffers(*gpu->device, c.count);
            generate(*gpu, buffers, c.count, mode, c.keyBits);
            const auto before = statsOf(*gpu, buffers, c.count, wide);

            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(sort->sort(batch, buffers, c.count, c.keyBits));
            REQUIRE(batch.submit(true));
            const auto after = statsOf(*gpu, buffers, c.count, wide);

            CHECK(after[0] == 0);   // ordered, and stable on ties
            for (int k = 1; k < 8; ++k) {
                CAPTURE(k);
                CHECK(after[k] == before[k]);   // the same pairs, permuted
            }
        }
    }
}

TEST_CASE("sorting ten million pairs, timed", "[gpu][algo][bench]") {
    LRT_REQUIRE_GPU(gpu);
    auto sort = gpu::RadixSort::create(*gpu->library);
    if (!sort) FAIL(sort.error().toString());
    constexpr uint32_t kCount = 10'000'000;
    gpu::SortBuffers buffers = sortBuffers(*gpu->device, kCount);
    generate(*gpu, buffers, kCount, 0, 32);
    const auto before = statsOf(*gpu, buffers, kCount, false);

    double best = 1e9;
    for (int round = 0; round < 3; ++round) {
        generate(*gpu, buffers, kCount, 0, 32);
        const auto start = std::chrono::steady_clock::now();
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(sort->sort(batch, buffers, kCount, 32));
        REQUIRE(batch.submit(true));
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        best = std::min(best, ms);
    }
    const auto after = statsOf(*gpu, buffers, kCount, false);
    CHECK(after[0] == 0);
    CHECK(after[1] == before[1]);
    CHECK(after[5] == before[5]);
    std::printf("radix sort, 10M pairs, 32-bit keys: %.1f ms (best of 3)\n", best);
}
