// Copyright (c) 2026 lucabRTrender contributors.
//
// The aofx host: what it loads, what it refuses and why, what it renders, and
// that openFXplayer's own bundles are welcome here.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "aofx/Effect.h"
#include "lrt/aofx/EffectRegistry.h"
#include "lrt/aofx/EffectRender.h"
#include "lrt/aofx/EffectRunner.h"
#include "lrt/core/Platform.h"
#include "lrt/gpu_host/Context.h"

namespace fs = std::filesystem;
using namespace lrt;

namespace {

const aofx_host::BundleReport* reportFor(const aofx_host::EffectRegistry& registry,
                                         const std::string& name) {
    for (const auto& report : registry.reports()) {
        if (report.path.filename().string() == name + ".aofx.bundle") {
            return &report;
        }
    }
    return nullptr;
}

gpu_host::Context* context() {
    return gpu_host::installProcessContext();
}

}   // namespace

TEST_CASE("the host loads a correct bundle and refuses the others with a reason", "[aofx]") {
    gpu_host::Context* gpu = context();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(LRT_AOFX_TEST_BUNDLES);
    registry.scan(gpu);

    const auto* good = reportFor(registry, "good");
    REQUIRE(good != nullptr);
    CHECK(good->loaded);
    CHECK(registry.find("tv.mediapro.aofx.test.nothing") != nullptr);

    const auto* abi = reportFor(registry, "wrongabi");
    REQUIRE(abi != nullptr);
    CHECK_FALSE(abi->loaded);
    CHECK(abi->reason.find("ABI") != std::string::npos);

    const auto* toolchain = reportFor(registry, "wrongtoolchain");
    REQUIRE(toolchain != nullptr);
    CHECK_FALSE(toolchain->loaded);
    CHECK(toolchain->reason.find("built with") != std::string::npos);

    const auto* hidden = reportFor(registry, "hidden");
    REQUIRE(hidden != nullptr);
    CHECK_FALSE(hidden->loaded);
    CHECK(hidden->reason.find("hidden visibility") != std::string::npos);
}

TEST_CASE("Invert renders one minus the picture, and the channel switches put alpha back",
          "[aofx][gpu]") {
    gpu_host::Context* gpu = context();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(LRT_AOFX_BUNDLES);
    registry.scan(gpu);
    aofx::Effect* invert = registry.find("tv.mediapro.aofx.invert");
    REQUIRE(invert != nullptr);

    // 37 x 23: neither a multiple of the kernel's 16x16 group, so whole groups
    // run past the edge and the kernel's bounds check is exercised.
    auto input = image::Image::create({0, 0, 37, 23});
    REQUIRE(input);
    {
        auto floats = (*input)->floats();
        const int stride = (*input)->stride();
        for (int y = 0; y < 23; ++y) {
            for (int x = 0; x < 37; ++x) {
                const size_t at = static_cast<size_t>((y * stride + x) * 4);
                floats[at] = static_cast<float>(x) / 37.0f;
                floats[at + 1] = static_cast<float>(y) / 23.0f;
                floats[at + 2] = 0.25f;
                floats[at + 3] = 0.5f;
            }
        }
    }

    aofx_host::EffectJob job;
    job.inputs.push_back({"Source", *input});
    auto rendered = aofx_host::renderEffect(*gpu, *invert, job);
    if (!rendered) {
        FAIL(rendered.error().toString());
    }
    const auto out = (*rendered)->floats();
    const int stride = (*rendered)->stride();
    CHECK(out[static_cast<size_t>((5 * stride + 7) * 4)] == Catch::Approx(1.0 - 7.0 / 37.0));
    CHECK(out[static_cast<size_t>((22 * stride + 36) * 4 + 1)] == Catch::Approx(1.0 - 22.0 / 23.0));
    CHECK(out[static_cast<size_t>((22 * stride + 36) * 4 + 3)] == Catch::Approx(0.5));

    job.params.push_back(aofx::ParamValue{aofx_host::kChannelRedParam, {1.0}, {}});
    job.params.push_back(aofx::ParamValue{aofx_host::kChannelGreenParam, {0.0}, {}});
    job.params.push_back(aofx::ParamValue{aofx_host::kChannelBlueParam, {0.0}, {}});
    job.params.push_back(aofx::ParamValue{aofx_host::kChannelAlphaParam, {0.0}, {}});
    auto redOnly = aofx_host::renderEffect(*gpu, *invert, job);
    REQUIRE(redOnly);
    const auto red = (*redOnly)->floats();
    CHECK(red[static_cast<size_t>((5 * stride + 7) * 4)] == Catch::Approx(1.0 - 7.0 / 37.0));
    CHECK(red[static_cast<size_t>((5 * stride + 7) * 4 + 1)] == Catch::Approx(5.0 / 23.0));
    CHECK(red[static_cast<size_t>((5 * stride + 7) * 4 + 2)] == Catch::Approx(0.25));
}

TEST_CASE("a run with the wrong number of buffers is refused with the kernel's name",
          "[aofx][gpu]") {
    gpu_host::Context* gpu = context();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(LRT_AOFX_TEST_RUN_BUNDLES);
    registry.scan(gpu);
    aofx::Effect* miscount = registry.find("tv.mediapro.aofx.test.miscount");
    REQUIRE(miscount != nullptr);

    auto input = image::Image::create({0, 0, 8, 8});
    REQUIRE(input);
    aofx_host::EffectJob job;
    job.inputs.push_back({"Source", *input});
    auto rendered = aofx_host::renderEffect(*gpu, *miscount, job);
    REQUIRE_FALSE(rendered);
    CHECK(rendered.error().message().find("declares 2 buffers") != std::string::npos);
    CHECK(rendered.error().message().find("lrt.test.miscount") != std::string::npos);
}

TEST_CASE("bundles openFXplayer built load in this host", "[aofx][compat]") {
    const fs::path theirs = fs::path(std::getenv("HOME") != nullptr ? std::getenv("HOME") : "") /
                            "openFXplayer/build/macos-arm64-debug/aofx";
    std::error_code ignored;
    if (!fs::is_directory(theirs, ignored)) {
        SKIP("no openFXplayer build at " + theirs.string());
    }
    gpu_host::Context* gpu = context();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(theirs);
    registry.scan(gpu);

    size_t loaded = 0;
    std::string refusals;
    for (const auto& report : registry.reports()) {
        if (report.loaded) {
            ++loaded;
        } else {
            refusals += report.path.filename().string() + ": " + report.reason + "\n";
        }
    }
    INFO(refusals);
    if (loaded == 0 && refusals.find("built with") != std::string::npos) {
        SKIP("openFXplayer's bundles were built with another toolchain; rebuild it:\n" + refusals);
    }
    CHECK(loaded > 0);
    CHECK(registry.find("tv.mediapro.aofx.invert") != nullptr);
}

// ABI 24: an effect's pages bound where they are. On a device that reads host
// memory in place the borrowed buffer *is* the pages -- a change to them after
// the borrow is what the device reads -- while a kept buffer is a copy taken
// at the call. Numbers the test wrote, read back: bookkeeping, not a picture.
TEST_CASE("a borrowed buffer is the effect's pages in place, and a kept one is a copy", "[aofx][abi24]") {
    gpu_host::Context* gpu = context();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    const uint64_t bytes = platform::pageSize();
    REQUIRE(bytes % 16 == 0);
    void* pages = platform::mapPages(bytes);
    REQUIRE(pages != nullptr);
    auto* words = static_cast<uint32_t*>(pages);
    for (uint64_t i = 0; i < bytes / 4; ++i) {
        words[i] = static_cast<uint32_t>(i * 7 + 3);
    }
    bool borrowed = false;
    uint32_t borrowedBefore = 0, borrowedAfter = 0, keptAfter = 0;
    bool sameBuffer = false, rebound = false, otherPagesRefused = false;
    REQUIRE(gpu->run([&] {
        aofx_host::EffectRunner runner(*gpu);
        const aofx::Buffer b = runner.borrow("test.borrow", pages, bytes);
        borrowed = b.isValid();
        if (!borrowed) {
            return;
        }
        const aofx::Buffer k = runner.keep("test.keep", pages, bytes);
        (void)runner.read(b, &borrowedBefore, sizeof(borrowedBefore));
        words[0] = 0xB0770u;
        (void)runner.read(b, &borrowedAfter, sizeof(borrowedAfter));
        (void)runner.read(k, &keptAfter, sizeof(keptAfter));
        sameBuffer = runner.borrow("test.borrow", pages, bytes).device == b.device;
        // Pages that are not page-aligned cannot be bound in place.
        otherPagesRefused = !runner.borrow("test.borrow.odd", static_cast<const char*>(pages) + 16, bytes - 16).isValid();
        // The same key over other pages is a new binding, not the old one.
        void* more = platform::mapPages(bytes);
        rebound = more != nullptr && runner.borrow("test.borrow", more, bytes).isValid();
        runner.drop("test.borrow");
        runner.drop("test.keep");
        platform::unmapPages(more, bytes);
    }));
    platform::unmapPages(pages, bytes);
    if (!borrowed) {
        SKIP("this device does not read host memory in place: borrow answers invalid, and keep is the path");
    }
    std::printf("  borrowed read %u, then %#x after the pages changed; kept read %u\n", borrowedBefore,
                borrowedAfter, keptAfter);
    CHECK(borrowedBefore == 3u);
    CHECK(borrowedAfter == 0xB0770u);
    CHECK(keptAfter == 3u);
    CHECK(sameBuffer);
    CHECK(otherPagesRefused);
    CHECK(rebound);
}

// ABI 24: what the host fills for an effect -- the project's format, the
// output's bounds when the job names none -- and the effect's own reason for
// failing, carried into the error instead of "did not render".
TEST_CASE("an effect is told the project's format and its complaint reaches the caller", "[aofx][abi24]") {
    gpu_host::Context* gpu = context();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(LRT_AOFX_TEST_RUN_BUNDLES);
    registry.scan(gpu);
    aofx::Effect* reporter = registry.find("tv.mediapro.aofx.test.reporter");
    REQUIRE(reporter != nullptr);

    aofx_host::EffectJob job;
    job.bounds = {0, 0, 64, 36};
    job.instance = "reporter.format";
    job.projectWidth = 1920;
    job.projectHeight = 1080;
    REQUIRE(aofx_host::renderEffect(*gpu, *reporter, job));
    auto state = aofx_host::EffectRunner::publishedState();
    CHECK(state[job.instance]["projectWidth"] == 1920.0);
    CHECK(state[job.instance]["projectHeight"] == 1080.0);

    job.instance = "reporter.bounds";
    job.projectWidth = 0;
    job.projectHeight = 0;
    REQUIRE(aofx_host::renderEffect(*gpu, *reporter, job));
    state = aofx_host::EffectRunner::publishedState();
    CHECK(state[job.instance]["projectWidth"] == 64.0);
    CHECK(state[job.instance]["projectHeight"] == 36.0);

    job.params.push_back(aofx::ParamValue{"fail", {1.0}, {}});
    auto failed = aofx_host::renderEffect(*gpu, *reporter, job);
    REQUIRE_FALSE(failed);
    std::printf("  %s\n", failed.error().toString().c_str());
    CHECK(failed.error().toString().find("the reporter was told to fail") != std::string::npos);
}
