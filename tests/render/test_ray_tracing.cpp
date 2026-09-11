// Copyright (c) 2026 lucabRTrender contributors.
//
// The Gaussian ray tracer against its GPU reference, and against the
// rasteriser where the two should agree.
//
// The ray tracer and the rasteriser are different renderers of the same
// particles (rt_render.slang lists how): the rasteriser projects a particle
// with EWA's local affine map and orders by centre depth, the ray tracer
// evaluates the particle exactly along the ray and orders by peak. So the
// ray tracer's ground truth is its own -- ReferenceRenderer::renderPeaks, the
// same definition computed without a BVH, segments or carry -- and it is held
// to that closely. Against the rasteriser it is held to agree where EWA is
// exact (orthographic) or nearly so (small particles), and only where
// particles do not overlap enough for the two orders to differ.
#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "lrt/io/Exr.h"
#include "lrt/render/GaussianRayTracer.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

using namespace lrt;
using test::CloudBuilder;
using test::randomCloud;

namespace {

// The tolerances, in 8-bit sRGB code values (docs/decisions.md). Against its
// reference the ray tracer differs only where the carry overflows or a ray
// grazes a proxy's edge: a handful of pixels, so the bound is on how many.
constexpr uint32_t kToPeaksP99 = 1;
constexpr uint64_t kToPeaksOver2PerMillion = 500;
// Against the rasteriser: orthographic, where EWA is exact, and perspective
// with particles of a few pixels, where EWA's affine map is off by this much.
constexpr uint32_t kToRasterOrthographicP99 = 2;
constexpr uint32_t kToRasterPerspectiveP99 = 4;

bool fewOver2(const render::ImageDifference& d) {
    return d.over2 * 1000000 <= d.pixels * kToPeaksOver2PerMillion;
}

struct Harness {
    test::Gpu*                 gpu;
    scene::CloudLoader         loader;
    render::TileRasterizer     raster;
    render::ReferenceRenderer  reference;
    render::GaussianRayTracer  rt;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu, render::RayTracerSettings rtSettings = {}) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto reference = render::ReferenceRenderer::create(*gpu->library);
    auto rt = render::GaussianRayTracer::create(*gpu->library, rtSettings);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!reference) FAIL(reference.error().toString());
    if (!rt) FAIL(rt.error().toString());
    return std::unique_ptr<Harness>(new Harness{gpu, std::move(*loader), std::move(*raster),
                                                std::move(*reference), std::move(*rt)});
}

/// Both routes, each test: the hardware one where the device has it.
#define LRT_ROUTE(gpu, route)                                                                  \
    const render::RayTracingRoute route =                                                      \
        GENERATE(render::RayTracingRoute::Hardware, render::RayTracingRoute::ComputeBvh);      \
    if (route == render::RayTracingRoute::Hardware &&                                          \
        !render::GaussianRayTracer::hardwareSupported(*(gpu)->device)) {                       \
        SKIP("the device has no RayQuery or acceleration structures");                         \
    }                                                                                          \
    std::printf("  route: %s\n", route == render::RayTracingRoute::Hardware ? "hardware" : "compute BVH");

render::RayTracerSettings on(render::RayTracingRoute route, render::RayTracerSettings settings = {}) {
    settings.route = route;
    return settings;
}

struct Comparison {
    render::ImageDifference toPeaks;
    render::ImageDifference toRaster;
};

Comparison compare(Harness& h, const render::Camera& camera,
                   std::span<const render::SplatInstance> instances,
                   const render::RenderSettings& settings) {
    render::RenderTargets traced, rasterised, peaks;
    auto stats = h.rt.render(camera, instances, settings, traced);
    if (!stats) FAIL(stats.error().toString());
    auto rasterStats = h.raster.render(camera, instances, settings, rasterised);
    if (!rasterStats) FAIL(rasterStats.error().toString());
    auto overflow = h.reference.renderPeaks(camera, instances, settings, peaks);
    if (!overflow) FAIL(overflow.error().toString());
    CHECK(*overflow == 0);
    auto toPeaks = render::compareImages(*h.gpu->library, traced.colour, peaks.colour,
                                         settings.width, settings.height);
    auto toRaster = render::compareImages(*h.gpu->library, traced.colour, rasterised.colour,
                                          settings.width, settings.height);
    if (!toPeaks) FAIL(toPeaks.error().toString());
    if (!toRaster) FAIL(toRaster.error().toString());
    // LRT_TEST_DUMP=<dir> keeps the three images for looking at.
    if (const char* dump = std::getenv("LRT_TEST_DUMP"); dump != nullptr) {
        static int serial = 0;
        serial += 1;
        const auto write = [&](const render::RenderTargets& t, const char* name) {
            auto colour = t.colour.readAll<float>(*h.gpu->device);
            auto depth = t.depth.readAll<float>(*h.gpu->device);
            REQUIRE(colour);
            REQUIRE(depth);
            const std::string path = std::string(dump) + "/rt" + std::to_string(serial) + "_" + name + ".exr";
            REQUIRE(io::writeExr(path, settings.width, settings.height, *colour, *depth, false));
        };
        write(traced, "traced");
        write(rasterised, "raster");
        write(peaks, "peaks");
    }
    std::printf("  rt %u splats, %u chunks, %u instances, %.1f ms build + %.1f ms trace\n"
                "    vs its reference: p99 %u, max %u, %llu of %llu over 2\n"
                "    vs the rasteriser: p99 %u, max %u, %llu over 2\n",
                stats->splats, stats->chunks, stats->instances, stats->buildMs, stats->renderMs,
                toPeaks->p99, toPeaks->max, static_cast<unsigned long long>(toPeaks->over2),
                static_cast<unsigned long long>(toPeaks->pixels), toRaster->p99, toRaster->max,
                static_cast<unsigned long long>(toRaster->over2));
    return {*toPeaks, *toRaster};
}

std::vector<float> readFloats(test::Gpu& gpu, const gpu::Buffer& buffer) {
    auto values = buffer.readAll<float>(*gpu.device);
    REQUIRE(values);
    return std::move(*values);
}

}   // namespace

TEST_CASE("the ray tracer renders what its GPU reference renders", "[render][rt][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    LRT_ROUTE(gpu, route)
    auto h = harness(gpu, on(route));
    // Dense and large: most rays need several segments and many carry.
    CloudBuilder built = randomCloud(2000, 11, 0.04F, 0.5F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;

    SECTION("perspective") {
        render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 7.0}, {0.0, 0.0, 0.0});
        camera.lens.focal = 30.0;
        const Comparison c = compare(*h, camera, instances, settings);
        CHECK(c.toPeaks.p99 <= kToPeaksP99);
        CHECK(fewOver2(c.toPeaks));
    }
    SECTION("orthographic") {
        render::Camera camera = render::Camera::lookingAt({0.0, 3.0, 6.0}, {0.0, 0.0, 0.0});
        camera.lens.projection = render::Lens::Projection::Orthographic;
        camera.lens.focal = 5.0;
        const Comparison c = compare(*h, camera, instances, settings);
        CHECK(c.toPeaks.p99 <= kToPeaksP99);
        CHECK(fewOver2(c.toPeaks));
    }
    SECTION("a window offset, a rolled film back, a background") {
        render::Camera camera = render::Camera::lookingAt({-2.0, 0.5, 6.0}, {0.0, 0.0, 0.0});
        camera.lens.windowTranslate[0] = 0.2;
        camera.lens.windowScale[1] = 1.3;
        camera.lens.windowRoll = 25.0;
        settings.background = {0.1F, 0.2F, 0.3F, 1.0F};
        const Comparison c = compare(*h, camera, instances, settings);
        CHECK(c.toPeaks.p99 <= kToPeaksP99);
        CHECK(fewOver2(c.toPeaks));
    }
    SECTION("instances: one turned, scaled and moved, one mirrored") {
        std::vector<render::SplatInstance> three = instances;
        three.push_back({&*cloud, aofx::xform::translation({0.5, 0.0, -2.5}) *
                                      aofx::xform::rotationY(40.0) *
                                      aofx::xform::scaling({1.0, 0.7, 1.3})});
        three.push_back({&*cloud, aofx::xform::translation({-3.0, 0.5, -1.0}) *
                                      aofx::xform::scaling({-1.0, 1.0, 1.0})});
        render::Camera camera = render::Camera::lookingAt({1.0, 1.5, 9.0}, {-0.5, 0.0, -0.5});
        const Comparison c = compare(*h, camera, three, settings);
        CHECK(c.toPeaks.p99 <= kToPeaksP99);
        CHECK(fewOver2(c.toPeaks));
    }
}

TEST_CASE("rays deeper than one record still render what the reference renders",
          "[render][rt][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    LRT_ROUTE(gpu, route)
    auto h = harness(gpu, on(route));
    // Faint and dense: most rays enter more proxies than one traversal
    // records before they turn opaque, so segments, the overlap between them
    // and the carry all run. (With one traversal allowed this scene is off
    // by ~20 code values at p99; checked below so it stays that deep.)
    CloudBuilder built = randomCloud(12000, 23, 0.08F, 0.5F, 0.01F, 0.04F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    instances.push_back({&*cloud, aofx::xform::translation({0.3, 0.2, -0.4}) * aofx::xform::rotationY(70.0)});
    render::RenderSettings settings;
    settings.width = 120;
    settings.height = 90;
    render::Camera camera = render::Camera::lookingAt({0.5, 0.3, 6.0}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    const Comparison c = compare(*h, camera, instances, settings);
    CHECK(c.toPeaks.p99 <= kToPeaksP99);
    CHECK(fewOver2(c.toPeaks));

    render::RayTracerSettings one;
    one.maxSegments = 1;
    auto shallow = harness(gpu, on(route, one));
    const Comparison d = compare(*shallow, camera, instances, settings);
    CHECK(d.toPeaks.p99 > 8);
}

TEST_CASE("where EWA is exact or nearly, the ray tracer agrees with the rasteriser",
          "[render][rt][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    LRT_ROUTE(gpu, route)
    auto h = harness(gpu, on(route));
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;

    SECTION("orthographic, large particles, sparse") {
        CloudBuilder built = randomCloud(12, 3, 0.2F, 0.4F);
        auto cloud = h->loader.upload(built.raw, 3);
        REQUIRE(cloud);
        const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
        render::Camera camera = render::Camera::lookingAt({0.0, 3.0, 6.0}, {0.0, 0.0, 0.0});
        camera.lens.projection = render::Lens::Projection::Orthographic;
        camera.lens.focal = 5.0;
        const Comparison c = compare(*h, camera, instances, settings);
        CHECK(c.toRaster.p99 <= kToRasterOrthographicP99);
    }
    // Small enough that EWA's affine map is close, large enough (several
    // pixels) that the rasteriser's 0.3 px dilation is not what is measured.
    SECTION("perspective, small particles, sparse") {
        CloudBuilder built = randomCloud(12, 4, 0.08F, 0.12F);
        auto cloud = h->loader.upload(built.raw, 3);
        REQUIRE(cloud);
        const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
        render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 7.0}, {0.0, 0.0, 0.0});
        camera.lens.focal = 30.0;
        const Comparison c = compare(*h, camera, instances, settings);
        CHECK(c.toRaster.p99 <= kToRasterPerspectiveP99);
    }
}

TEST_CASE("a ray deeper than one traversal's record keeps going", "[render][rt][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    LRT_ROUTE(gpu, route)
    // Four hundred thin layers of 1% on the camera's axis: 1 - 0.99^400 when
    // every layer is taken, 1 - 0.99^256 when one traversal's record is.
    CloudBuilder b;
    for (int layer = 0; layer < 400; ++layer) {
        b.add(0.0F, 0.0F, -3.0F - 0.01F * static_cast<float>(layer), 0.01F, 0.5F, 0.5F, 0.0005F,
              {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
    }
    render::RenderSettings settings;
    // Odd, so the middle pixel's ray is the optical axis.
    settings.width = 33;
    settings.height = 33;
    render::Camera camera;
    const auto alphaAtCentre = [&](uint32_t segments) {
        render::RayTracerSettings rt;
        rt.maxSegments = segments;
        auto h = harness(gpu, on(route, rt));
        auto cloud = h->loader.upload(b.raw);
        REQUIRE(cloud);
        const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
        render::RenderTargets targets;
        auto stats = h->rt.render(camera, instances, settings, targets);
        if (!stats) FAIL(stats.error().toString());
        const auto p = readFloats(*gpu, targets.colour);
        return p[(16 * 33 + 16) * 4 + 3];
    };
    CHECK(std::abs(alphaAtCentre(16) - (1.0F - std::pow(0.99F, 400.0F))) < 1e-3F);
    CHECK(std::abs(alphaAtCentre(1) - (1.0F - std::pow(0.99F, 256.0F))) < 1e-3F);
}

TEST_CASE("ray traced depth is view depth, in both depth modes", "[render][rt][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    LRT_ROUTE(gpu, route)
    auto h = harness(gpu, on(route));
    CloudBuilder b;
    b.add(0.0F, 0.0F, -3.0F, 0.99F, 0.6F, 0.6F, 0.001F, {1, 0, 0, 0}, {1, 1, 1});
    b.add(0.0F, 0.0F, -5.0F, 0.99F, 0.6F, 0.6F, 0.001F, {1, 0, 0, 0}, {1, 1, 1});
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::RenderSettings settings;
    settings.width = 33;
    settings.height = 33;
    render::Camera camera;
    render::RenderTargets targets;
    // First crossing: the front layer alone is past half coverage.
    settings.depth = render::RenderSettings::Depth::Threshold;
    REQUIRE(h->rt.render(camera, instances, settings, targets));
    CHECK(std::abs(readFloats(*gpu, targets.depth)[16 * 33 + 16] - 3.0F) < 1e-3F);
    // Mean: 0.99 of the weight at 3, 0.0099 at 5.
    settings.depth = render::RenderSettings::Depth::Mean;
    REQUIRE(h->rt.render(camera, instances, settings, targets));
    const float mean = (0.99F * 3.0F + 0.0099F * 5.0F) / 0.9999F;
    CHECK(std::abs(readFloats(*gpu, targets.depth)[16 * 33 + 16] - mean) < 1e-3F);
}

TEST_CASE("splitting a cloud into chunks does not change the image", "[render][rt][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    const render::RayTracingRoute route = render::RayTracingRoute::Hardware;
    if (!render::GaussianRayTracer::hardwareSupported(*gpu->device)) {
        SKIP("chunks are the hardware route's");
    }
    render::RayTracerSettings small;
    small.chunkSplats = 97;
    auto whole = harness(gpu, on(route));
    auto chunked = harness(gpu, on(route, small));
    CloudBuilder built = randomCloud(1000, 5, 0.04F, 0.5F);
    auto cloud = whole->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::RenderSettings settings;
    settings.width = 160;
    settings.height = 120;
    render::Camera camera = render::Camera::lookingAt({1.0, 0.5, 6.0}, {0.0, 0.0, 0.0});
    render::RenderTargets a, b;
    auto sa = whole->rt.render(camera, instances, settings, a);
    auto sb = chunked->rt.render(camera, instances, settings, b);
    if (!sa) FAIL(sa.error().toString());
    if (!sb) FAIL(sb.error().toString());
    CHECK(sb->chunks == 11);
    auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
    REQUIRE(diff);
    if (const char* dump = std::getenv("LRT_TEST_DUMP"); dump != nullptr) {
        for (const auto& [targets, name] : {std::pair{&a, "whole"}, std::pair{&b, "chunked"}}) {
            auto colour = targets->colour.readAll<float>(*gpu->device);
            REQUIRE(colour);
            REQUIRE(io::writeExr(std::string(dump) + "/chunks_" + name + ".exr", settings.width,
                                 settings.height, *colour, {}, false));
        }
    }
    std::printf("  chunked: p99 %u, max %u, %llu pixels over 2\n", diff->p99, diff->max,
                static_cast<unsigned long long>(diff->over2));
    // Not bit-identical: a different BVH makes the intersector report entry
    // distances a few ulps apart, which moves segment boundaries and, where
    // the carry overflows, the order of near-equal peaks.
    CHECK(diff->max <= 1);
    // And a second frame reuses what the first built.
    auto again = chunked->rt.render(camera, instances, settings, b);
    REQUIRE(again);
    CHECK_FALSE(again->rebuilt);
}


