// Copyright (c) 2026 lucabRTrender contributors.
//
// Levels of detail, built and cut on the device, checked by renders compared on
// the device.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/io/Exr.h"
#include "lrt/lod/Lod.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

using namespace lrt;
using test::CloudBuilder;
using test::randomCloud;

namespace {

struct Harness {
    test::Gpu*             gpu;
    scene::CloudLoader     loader;
    render::TileRasterizer raster;
    lod::LodBuilder        builder;
    lod::CutSelector       cut;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto builder = lod::LodBuilder::create(*gpu->library);
    auto cut = lod::CutSelector::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!cut) FAIL(cut.error().toString());
    return std::unique_ptr<Harness>(
        new Harness{gpu, std::move(*loader), std::move(*raster), std::move(*builder), std::move(*cut)});
}

render::ImageDifference renderBoth(Harness& h, const render::Camera& camera, const scene::GpuSplats& full,
                                   const lod::LodCloud& lod, float threshold, const render::RenderSettings& settings,
                                   lod::CutStats* stats = nullptr) {
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);
    render::RenderTargets a, b;
    REQUIRE(h.raster.render(projection, std::vector<render::SplatInstance>{{&full, render::Mat4::identity()}},
                            settings, a));
    std::vector<lod::CutStats> cutStats;
    const std::vector<lod::LodInstance> instances{{&lod, render::Mat4::identity()}};
    auto selected = h.cut.select(projection, instances, threshold, &cutStats);
    if (!selected) FAIL(selected.error().toString());
    REQUIRE(h.raster.render(projection, *selected, settings, b));
    if (stats != nullptr && !cutStats.empty()) {
        *stats = cutStats.front();
    }
    if (const char* dump = std::getenv("LRT_TEST_DUMP"); dump != nullptr) {
        static int serial = 0;
        ++serial;
        for (const auto& [t, name] : {std::pair{&a, "full"}, std::pair{&b, "lod"}}) {
            auto c = t->colour.readAll<float>(*h.gpu->device);
            REQUIRE(c);
            REQUIRE(io::writeExr(std::string(dump) + "/lod" + std::to_string(serial) + "_" + name + ".exr",
                                 settings.width, settings.height, *c));
        }
    }
    auto diff = render::compareImages(*h.gpu->library, a.colour, b.colour, settings.width, settings.height);
    REQUIRE(diff);
    return *diff;
}

}   // namespace

TEST_CASE("a cut with threshold zero draws the cloud as it is", "[lod][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(5000, 41);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    auto lod = h->builder.build(*cloud);
    if (!lod) FAIL(lod.error().toString());
    CHECK(!lod->levels.empty());
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
    lod::CutStats stats;
    const auto diff = renderBoth(*h, camera, *cloud, *lod, 0.0F, settings, &stats);
    std::printf("  threshold 0: %u splats, %u merged; p99 %u, max %u\n", stats.splats, stats.merged, diff.p99, diff.max);
    CHECK(stats.splats == cloud->count);
    CHECK(stats.merged == 0);
    // Not bit-identical: the cut draws the splats in Morton order, and splats
    // whose 24-bit depth keys tie (nearer than 3e-5 relative -- common among
    // 5000 large splats in a few units of depth) keep index order, which is
    // now a different order. Measured: p99 1, 0.6% of pixels over 2.
    CHECK(diff.p99 <= 1);
    CHECK(diff.over2 * 100 <= diff.pixels);
}

TEST_CASE("a cell holding one splat merges into that splat", "[lod][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // Splats far enough apart that every level-1 cell holds at most one: each
    // merged Gaussian is its single splat's moments, turned back into a splat.
    CloudBuilder b;
    const std::array<std::array<float, 3>, 6> at{{{-2, -2, -2}, {2, -2, -2}, {-2, 2, -2}, {2, 2, 2}, {-2, 2, 2}, {2, -2, 2}}};
    test::Lcg rng;
    for (const auto& p : at) {
        b.add(p[0], p[1], p[2], rng.range(0.3F, 0.95F), rng.range(0.1F, 0.5F), rng.range(0.1F, 0.5F),
              rng.range(0.1F, 0.5F), {rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)},
              {rng.range(0.1F, 1.0F), rng.range(0.1F, 1.0F), rng.range(0.1F, 1.0F)},
              {0.2F, -0.1F, 0.3F, 0.0F, 0.1F, -0.2F, 0.1F, 0.0F, -0.3F});
    }
    auto cloud = h->loader.upload(b.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings one;
    one.coarsestLevel = 1;
    one.maxGroupFraction = 1.0F;   // keep level 1 though it merges nothing
    auto lod = h->builder.build(*cloud, one);
    if (!lod) FAIL(lod.error().toString());
    REQUIRE(!lod->levels.empty());
    CHECK(lod->levels.front().gaussians.count == 6);
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    render::Camera camera = render::Camera::lookingAt({3.0, 2.0, 12.0}, {0.0, 0.0, 0.0});
    lod::CutStats stats;
    // A threshold every cell meets: everything is drawn merged, at level 1.
    const auto diff = renderBoth(*h, camera, *cloud, *lod, 1e6F, settings, &stats);
    std::printf("  single-splat cells: %u splats, %u merged; p99 %u, max %u\n", stats.splats, stats.merged,
                diff.p99, diff.max);
    CHECK(stats.splats == 0);
    CHECK(stats.merged == 6);
    // The same covariance to 1e-4 (the 10-bit quaternion); what differs is
    // the outermost ring of pixels, where a coverage of 1/255 more or less is
    // a dozen code values once encoded.
    CHECK(diff.p99 <= 1);
    CHECK(diff.over2 * 100 <= diff.pixels);
}

TEST_CASE("far away, a dense cloud draws a fraction of its splats and looks the same", "[lod][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(60000, 43, 0.003F, 0.05F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    auto lod = h->builder.build(*cloud);
    if (!lod) FAIL(lod.error().toString());
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::Camera camera = render::Camera::lookingAt({6.0, 4.0, 30.0}, {0.0, 0.0, 0.0});
    for (const float threshold : {1.0F, 2.0F, 4.0F}) {
        lod::CutStats stats;
        const auto diff = renderBoth(*h, camera, *cloud, *lod, threshold, settings, &stats);
        std::printf("  threshold %.0f px: %u splats + %u merged of %u (%.1f%%); p99 %u, max %u\n", threshold,
                    stats.splats, stats.merged, stats.available,
                    100.0 * (stats.splats + stats.merged) / stats.available, diff.p99, diff.max);
        // Random colours splat to splat are the worst case merging has: every
        // merged Gaussian is an average of colours no pixel showed. Bounds
        // measured on this cloud (p99 0 / 9 / 29 at 1 / 2 / 4 px).
        if (threshold == 1.0F) {
            CHECK(diff.p99 <= 1);
        } else if (threshold == 2.0F) {
            CHECK(stats.splats + stats.merged < stats.available * 3 / 4);
            CHECK(diff.p99 <= 12);
        } else {
            CHECK(stats.splats + stats.merged < stats.available / 4);
            CHECK(diff.p99 <= 36);
        }
    }
}


namespace {

/// The cloud as a stream would hold it: chunk c in slot chunks-1-c (copied on
/// the device), and only the chunks `keep` says on it.
lod::LodCloud streamedCopy(Harness& h, const lod::LodCloud& lod, const std::vector<bool>& keep) {
    gpu::Device& device = *h.gpu->device;
    lod::LodCloud out = lod;
    out.streamed = true;
    const auto like = [&](const gpu::Buffer& b) {
        gpu::BufferDesc desc;
        desc.bytes = b.bytes();
        desc.elementBytes = b.elementBytes();
        desc.label = "test.store";
        auto made = gpu::Buffer::create(device, desc);
        REQUIRE(made);
        return *made;
    };
    out.splats.positions = like(lod.splats.positions);
    out.splats.shape = like(lod.splats.shape);
    out.splats.sh = like(lod.splats.sh);
    out.groups = like(lod.groups);
    std::vector<uint32_t> resident(lod.chunks(), 0);
    gpu::CommandBatch batch(device);
    for (uint32_t c = 0; c < lod.chunks(); ++c) {
        if (!keep[c]) {
            out.slots[c] = -1;
            continue;
        }
        const uint32_t slot = lod.chunks() - 1 - c;
        out.slots[c] = static_cast<int32_t>(slot);
        resident[c] = 1;
        const uint64_t from = uint64_t{c} * lod.chunkSplats;
        const uint64_t to = uint64_t{slot} * lod.chunkSplats;
        const uint64_t n = lod.chunkCount(c);
        const auto copy = [&](const gpu::Buffer& dst, const gpu::Buffer& src, uint64_t perSplat) {
            batch.encoder()->copyBuffer(dst.rhi(), to * perSplat, src.rhi(), from * perSplat, n * perSplat);
        };
        copy(out.splats.positions, lod.splats.positions, 16);
        copy(out.splats.shape, lod.splats.shape, 16);
        copy(out.splats.sh, lod.splats.sh, 4 * uint64_t{lod.splats.shWords});
        copy(out.groups, lod.groups, 4);
    }
    batch.markDirty();
    REQUIRE(batch.submit(true));
    auto flags = gpu::Buffer::fromSpan(device, std::span<const uint32_t>(resident), "test.resident");
    REQUIRE(flags);
    out.resident = *flags;
    return out;
}

struct Cut {
    lod::CutStats           stats;
    render::RenderTargets   targets;
};

Cut cutAndRender(Harness& h, const render::Projection& projection, const lod::LodCloud& lod, float threshold,
                 const render::RenderSettings& settings) {
    Cut cut;
    std::vector<lod::CutStats> stats;
    const std::vector<lod::LodInstance> instances{{&lod, render::Mat4::identity()}};
    auto selected = h.cut.select(projection, instances, threshold, &stats);
    if (!selected) FAIL(selected.error().toString());
    REQUIRE(stats.size() == 1);
    cut.stats = stats.front();
    REQUIRE(h.raster.render(projection, *selected, settings, cut.targets));
    return cut;
}

}   // namespace

TEST_CASE("chunks not on the device are drawn merged, and chunks not wanted change nothing", "[lod][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(20000, 47, 0.003F, 0.05F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 1000;
    auto lod = h->builder.build(*cloud, chunked);
    if (!lod) FAIL(lod.error().toString());
    REQUIRE(lod->chunks() == 20);
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::Camera camera = render::Camera::lookingAt({4.0, 3.0, 18.0}, {0.0, 0.0, 0.0});
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);
    const auto compare = [&](const Cut& a, const Cut& b) {
        auto diff = render::compareImages(*gpu->library, a.targets.colour, b.targets.colour, settings.width,
                                          settings.height);
        REQUIRE(diff);
        return *diff;
    };

    SECTION("every chunk there, in other slots: the same cut, the same image") {
        const lod::LodCloud all = streamedCopy(*h, *lod, std::vector<bool>(20, true));
        for (const float threshold : {0.0F, 2.0F, 1e6F}) {
            const Cut memory = cutAndRender(*h, projection, *lod, threshold, settings);
            const Cut stream = cutAndRender(*h, projection, all, threshold, settings);
            const auto diff = compare(memory, stream);
            CHECK(stream.stats.splats == memory.stats.splats);
            CHECK(stream.stats.merged == memory.stats.merged);
            CHECK(diff.max == 0);
            REQUIRE(stream.stats.needs.size() == 20);
            const auto wanted = std::count(stream.stats.needs.begin(), stream.stats.needs.end(), uint8_t{1});
            if (threshold == 0.0F) {
                CHECK(wanted == 20);
            } else if (threshold == 1e6F) {
                CHECK(wanted == 0);
                CHECK(stream.stats.splats == 0);
            }
        }
    }

    SECTION("chunks missing: their places drawn merged until they come") {
        std::vector<bool> keep(20, true);
        uint32_t missing = 0;
        for (uint32_t c = 0; c < 20; c += 3) {
            keep[c] = false;
            missing += lod->chunkCount(c);
        }
        const lod::LodCloud partial = streamedCopy(*h, *lod, keep);
        const Cut memory = cutAndRender(*h, projection, *lod, 0.0F, settings);
        const Cut stream = cutAndRender(*h, projection, partial, 0.0F, settings);
        const auto diff = compare(memory, stream);
        std::printf("  %u of %u splats missing: %u splats + %u merged drawn; p99 %u, max %u\n", missing, lod->count,
                    stream.stats.splats, stream.stats.merged, diff.p99, diff.max);
        CHECK(stream.stats.splats <= lod->count - missing);
        CHECK(stream.stats.merged > 0);
        CHECK(std::count(stream.stats.needs.begin(), stream.stats.needs.end(), uint8_t{1}) == 20);
        CHECK(diff.max > 0);
    }

    SECTION("dropping the chunks a view does not want leaves it as it was") {
        // Close to one side of the cloud: the near chunks are wanted, the far
        // ones merged away.
        const render::Projection near = render::projectionFor(
            render::Camera::lookingAt({0.0, 0.5, 4.5}, {0.0, 0.0, 0.0}), settings.width, settings.height);
        const lod::LodCloud all = streamedCopy(*h, *lod, std::vector<bool>(20, true));
        for (const float threshold : {36.0F, 48.0F}) {
            const Cut full = cutAndRender(*h, near, all, threshold, settings);
            std::vector<bool> keep(20);
            for (uint32_t c = 0; c < 20; ++c) {
                keep[c] = full.stats.needs[c] != 0;
            }
            const auto wanted = std::count(keep.begin(), keep.end(), true);
            const lod::LodCloud needed = streamedCopy(*h, *lod, keep);
            const Cut trimmed = cutAndRender(*h, near, needed, threshold, settings);
            const auto diff = compare(full, trimmed);
            std::printf("  threshold %.0f px: %ld of 20 chunks wanted; %u splats + %u merged; max %u\n", threshold,
                        static_cast<long>(wanted), trimmed.stats.splats, trimmed.stats.merged, diff.max);
            CHECK(wanted < 20);
            CHECK(trimmed.stats.splats == full.stats.splats);
            CHECK(trimmed.stats.merged == full.stats.merged);
            CHECK(diff.max == 0);
        }
    }
}
