// Copyright (c) 2026 lucabRTrender contributors.
//
// What a launch reads when the uniform changed since the last one.
//
// Every frame this renderer draws is a launch whose parameter block differs
// from the launch before it -- the sample count, the bounce limit, the camera.
// A launch that reads the block of the launch before it draws the frame
// before it, and nothing in an image says so. This is the smallest question
// that can be asked about that, so that when it fails the answer is not
// "the path tracer is wrong" but "this device's launches read a stale
// parameter block", which is a different repair.
//
// The kernel is one line: write the uniform into a buffer. The compute entry
// and the ray generation entry are the same line, so a failure of one and not
// the other says where the fault lies. The ray generation entry traces
// nothing: a pipeline is launchable with a ray generation program alone, and
// what is under test is the parameter block, not the traversal.
#include "GpuTest.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "lrt/gpu/RayTracingKernel.h"

using namespace lrt;

namespace {

constexpr uint32_t kThreads = 4096;   // enough that a launch is not instant
constexpr uint32_t kLaunches = 64;

/// The values a run walks through. They alternate rather than climb, so that
/// a launch that answers with the one before it is off by a known amount and
/// not by one.
uint32_t valueFor(uint32_t launch) { return (launch % 2 == 0) ? 0x1111u : 0x2222u; }

/// Counts the launches whose answer was not the value they were given, and
/// how many of those answered with the value of the launch before. Only
/// counters cross back to the host.
struct Echo {
    uint32_t wrong = 0;
    uint32_t stale = 0;
    uint32_t wrongWords = 0;
    uint32_t wrongRw = 0;
    std::string pattern;
};

}   // namespace

TEST_CASE("a launch reads the uniform it was given, not the one before it", "[gpu][uniforms]") {
    LRT_REQUIRE_GPU(gpu);
    gpu::Buffer answers = test::uintBuffer(*gpu->device, 2 * kThreads + 8, "echo.answers");
    gpu::Buffer words = test::uintBuffer(*gpu->device, 4, "echo.words");

    SECTION("in a compute kernel") {
        auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/uniform_echo", "echoCompute");
        if (!made) FAIL(made.error().toString());
        gpu::ComputeKernel echo = std::move(*made);
        Echo counts;
        for (uint32_t launch = 0; launch < kLaunches; ++launch) {
            const uint32_t value = valueFor(launch);
            REQUIRE(words.write(*gpu->device, 0, sizeof(value), &value));   // the host's copy
            {
                gpu::CommandBatch batch(*gpu->device);
                echo.dispatch(batch, {(kThreads + 255) / 256, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["answers"].setBinding(answers.rhi());
                    cursor["echoWords"].setBinding(words.rhi());
                    cursor["echoRw"].setBinding(words.rhi());
                    cursor["echo"]["value"].setData(value);
                    cursor["echo"]["count"].setData(kThreads);
                });
                REQUIRE(batch.submit(true));
            }
            uint32_t got[2] = {0, 0};
            REQUIRE(answers.read(*gpu->device, 0, sizeof(uint32_t), &got[0]));
            REQUIRE(answers.read(*gpu->device, (kThreads + 4) * sizeof(uint32_t), sizeof(uint32_t), &got[1]));
            if (got[0] != value) {
                ++counts.wrong;
                if (launch > 0 && got[0] == valueFor(launch - 1)) {
                    ++counts.stale;
                }
            }
            if (got[1] != value) {
                ++counts.wrongWords;
            }
            counts.pattern += (got[0] == value) ? '.' : 'X';
            uint32_t rw = 0;
            REQUIRE(answers.read(*gpu->device, (kThreads + 1) * sizeof(uint32_t), sizeof(rw), &rw));
            if (rw != value) {
                ++counts.wrongRw;
            }
        }
        uint32_t ran = 0;
        REQUIRE(answers.read(*gpu->device, kThreads * sizeof(uint32_t), sizeof(ran), &ran));
        std::printf("  compute: %u of %u launches wrong, %u of them the launch before, %u ran, %u wrong through a buffer, %u through a writable one\n", counts.wrong,
                    kLaunches, counts.stale, ran, counts.wrongWords, counts.wrongRw);
        std::printf("    %s\n", counts.pattern.c_str());
        CHECK(counts.wrong == 0);
        CHECK(counts.wrongWords == 0);
        CHECK(counts.wrongRw == 0);
        CHECK(ran == kLaunches);
    }

    SECTION("in a ray generation entry") {
        const gpu::Caps& caps = gpu->device->caps();
        if (!caps.rayTracing) {
            SKIP("no ray tracing pipelines: there is no ray generation entry to launch");
        }
        gpu::RayTracingDesc desc;
        desc.module = "lrt/test/uniform_echo_rays";
        desc.rayGen = "echoRaysGen";
        desc.maxRecursion = 1;
        desc.payloadBytes = 16;
        auto made = gpu::RayTracingKernel::create(*gpu->library, desc);
        if (!made) FAIL(made.error().toString());
        gpu::RayTracingKernel echo = std::move(*made);
        Echo counts;
        for (uint32_t launch = 0; launch < kLaunches; ++launch) {
            const uint32_t value = valueFor(launch);
            REQUIRE(words.write(*gpu->device, 0, sizeof(value), &value));   // the host's copy
            {
                gpu::CommandBatch batch(*gpu->device);
                echo.dispatch(batch, kThreads, 1, 1, [&](rhi::ShaderCursor cursor) {
                    cursor["answers"].setBinding(answers.rhi());
                    cursor["echoWords"].setBinding(words.rhi());
                    cursor["echoRw"].setBinding(words.rhi());
                    cursor["echo"]["value"].setData(value);
                    cursor["echo"]["count"].setData(kThreads);
                });
                REQUIRE(batch.submit(true));
            }
            uint32_t got[2] = {0, 0};
            REQUIRE(answers.read(*gpu->device, 0, sizeof(uint32_t), &got[0]));
            REQUIRE(answers.read(*gpu->device, (kThreads + 4) * sizeof(uint32_t), sizeof(uint32_t), &got[1]));
            if (got[0] != value) {
                ++counts.wrong;
                if (launch > 0 && got[0] == valueFor(launch - 1)) {
                    ++counts.stale;
                }
            }
            if (got[1] != value) {
                ++counts.wrongWords;
            }
            counts.pattern += (got[0] == value) ? '.' : 'X';
            uint32_t rw = 0;
            REQUIRE(answers.read(*gpu->device, (kThreads + 1) * sizeof(uint32_t), sizeof(rw), &rw));
            if (rw != value) {
                ++counts.wrongRw;
            }
        }
        uint32_t ran = 0;
        REQUIRE(answers.read(*gpu->device, kThreads * sizeof(uint32_t), sizeof(ran), &ran));
        std::printf("  rays: %u of %u launches wrong, %u of them the launch before, %u ran, %u wrong through a buffer, %u through a writable one\n", counts.wrong,
                    kLaunches, counts.stale, ran, counts.wrongWords, counts.wrongRw);
        std::printf("    %s\n", counts.pattern.c_str());
        CHECK(counts.wrong == 0);
        CHECK(counts.wrongWords == 0);
        CHECK(counts.wrongRw == 0);
        CHECK(ran == kLaunches);
    }
}
