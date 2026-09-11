// Copyright (c) 2026 lucabRTrender contributors.
//
// Levels of detail, built and cut on the device, checked by renders compared on
// the device.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

