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



TEST_CASE("a SplatEdit traces as the ray tracer's reference renders it", "[render][rt][gpu][edit]") {
    LRT_REQUIRE_GPU(gpu);
    LRT_ROUTE(gpu, route)
    auto h = harness(gpu, on(route));
    CloudBuilder built = randomCloud(2000, 31, 0.04F, 0.5F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    render::SplatEdit removeSphere;
    removeSphere.active = true;
    removeSphere.mode = render::SplatEdit::Mode::Remove;
    removeSphere.shape = render::SplatEdit::Shape::Sphere;
    removeSphere.size = {1.0F, 0.0F, 0.0F};
    render::SplatEdit gradeBox;
    gradeBox.active = true;
    gradeBox.mode = render::SplatEdit::Mode::Grade;
    gradeBox.size = {1.5F, 1.0F, 1.5F};
    gradeBox.tint = {1.0F, 0.5F, 0.2F};
    gradeBox.saturation = 0.4F;
    gradeBox.opacity = 0.4F;
    gradeBox.maxScale = 0.35F;
    const std::vector<render::SplatInstance> instances{
        {&*cloud, render::Mat4::identity(), removeSphere},
        {&*cloud, aofx::xform::translation({0.5, 0.0, -2.5}) * aofx::xform::rotationY(40.0), gradeBox}};
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    render::Camera camera = render::Camera::lookingAt({1.0, 1.5, 9.0}, {-0.5, 0.0, -0.5});
    const Comparison c = compare(*h, camera, instances, settings);
    CHECK(c.toPeaks.p99 <= kToPeaksP99);
    CHECK(fewOver2(c.toPeaks));
}

// A shadow ray asks one thing of a cloud -- how much light gets through --
// and rt_shadow.slang answers it with a product of (1 - alpha), no k-buffer,
// no segments and no order. Three closed forms hold it to that, each one
// exact rather than a tolerance on an image:
//   - a ray through a particle's centre peaks at power 0, so its alpha is the
//     particle's opacity and the transmittance is exactly 1 - opacity;
//   - N such particles in a line give (1 - opacity)^N, whatever order the
//     traversal offers them in -- which is the point of not ordering;
//   - deep enough, the product falls under the cut and the ray stops, which
//     the kernel counts.
TEST_CASE("a shadow ray through splats is the product of what each lets through",
          "[render][rt][gpu][shadow]") {
    LRT_REQUIRE_GPU(gpu);
    if (!render::GaussianRayTracer::hardwareSupported(*gpu->device)) {
        SKIP("the device has no RayQuery or acceleration structures");
    }
    auto h = harness(gpu, on(render::RayTracingRoute::Hardware));
    auto kernel = gpu::ComputeKernel::create(*gpu->library, "lrt/rt/rt_shadow_kernel", "rtShadowRays");
    if (!kernel) FAIL(kernel.error().toString());

    // Particles on the z axis, every one isotropic, so a ray down -z through
    // the axis peaks at each centre with power 0.
    const float opacity = 0.5F;
    const uint32_t stacked = 4;
    CloudBuilder built;
    for (uint32_t k = 0; k < stacked; ++k) {
        built.add(0.0F, 0.0F, -2.0F * float(k), opacity, 0.2F, 0.2F, 0.2F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
    }
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};

    // One render builds the structures the shadow rays trace against.
    render::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    render::RenderTargets targets;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 6.0}, {0.0, 0.0, -3.0});
    auto stats = h->rt.render(camera, instances, settings, targets);
    if (!stats) FAIL(stats.error().toString());
    const render::ShadowScene scene = h->rt.shadowScene();
    REQUIRE(scene.tlas != nullptr);

    // Four rays: through every particle, through the nearest one alone,
    // past the cloud entirely, and away from it (its particles behind).
    struct Ray {
        std::array<float, 4> origin;      // xyz, tMin
        std::array<float, 4> direction;   // xyz, tMax
        float                want;
        const char*          name;
    };
    const float one = 1.0F - opacity;
    const std::vector<Ray> cases = {
        {{0.0F, 0.0F, 5.0F, 1e-3F}, {0.0F, 0.0F, -1.0F, 100.0F}, std::pow(one, float(stacked)),
         "through all four"},
        {{0.0F, 0.0F, 5.0F, 1e-3F}, {0.0F, 0.0F, -1.0F, 6.0F}, one, "through the nearest, tMax before the rest"},
        {{3.0F, 0.0F, 5.0F, 1e-3F}, {0.0F, 0.0F, -1.0F, 100.0F}, 1.0F, "past the cloud"},
        {{0.0F, 0.0F, 5.0F, 1e-3F}, {0.0F, 0.0F, 1.0F, 100.0F}, 1.0F, "away from it"},
    };
    std::vector<float> rayData;
    for (const Ray& r : cases) {
        rayData.insert(rayData.end(), r.origin.begin(), r.origin.end());
        rayData.insert(rayData.end(), r.direction.begin(), r.direction.end());
    }
    gpu::BufferDesc desc;
    desc.bytes = rayData.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "shadow.rays";
    auto rays = gpu::Buffer::create(*gpu->device, desc, rayData.data());
    REQUIRE(rays);
    gpu::BufferDesc out;
    out.bytes = cases.size() * sizeof(float);
    out.elementBytes = 4;
    out.label = "shadow.transmittance";
    auto shadow = gpu::Buffer::create(*gpu->device, out);
    REQUIRE(shadow);
    gpu::Buffer counters = test::uintBuffer(*gpu->device, 4, "shadow.counters");

    const auto trace = [&](float cut, uint32_t cull = 0) {
        {
            gpu::CommandBatch batch(*gpu->device);
            kernel->dispatch(batch, {static_cast<uint32_t>(cases.size()), 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["scene"].setBinding(scene.tlas);
                cursor["frames"].setBinding(scene.frames->rhi());
                cursor["colours"].setBinding(scene.colours->rhi());
                cursor["instanceData"].setBinding(scene.instanceData->rhi());
                cursor["instanceIndices"].setBinding(scene.instanceIndices->rhi());
                cursor["rays"].setBinding(rays->rhi());
                cursor["shadow"].setBinding(shadow->rhi());
                cursor["counters"].setBinding(counters.rhi());
                rhi::ShaderCursor p = cursor["shadowParams"];
                p["rays"].setData(static_cast<uint32_t>(cases.size()));
                p["cut"].setData(cut);
                p["nearZ"].setData(0.0F);
                p["farZ"].setData(1.0e9F);
                p["cull"].setData(cull);
            });
            REQUIRE(batch.submit(true));
        }
        std::vector<float> through(cases.size(), 0.0F);
        REQUIRE(shadow->read(*gpu->device, 0, through.size() * sizeof(float), through.data()));
        std::array<uint32_t, 4> n{};
        REQUIRE(counters.read(*gpu->device, 0, sizeof(n), n.data()));
        return std::make_pair(through, n);
    };

    const auto [through, n] = trace(1e-3F);
    for (size_t k = 0; k < cases.size(); ++k) {
        std::printf("  %-38s transmittance %.5f, closed form %.5f\n", cases[k].name, double(through[k]),
                    double(cases[k].want));
        CHECK(std::abs(through[k] - cases[k].want) < 2e-3F);
    }
    std::printf("  %u particles taken, %u duplicates caught, %u rays cut\n", n[0], n[1], n[2]);
    // Every particle counted once, though each proxy was offered twice --
    // entry face and exit face, since a shadow ray culls neither. The ring
    // is what turns those five second offers into five duplicates and one
    // factor each in the product.
    CHECK(n[0] == stacked + 1);
    CHECK(n[2] == 0);

    // The cut: with the four particles at 0.5 the product is 0.0625, so a cut
    // above it stops the ray -- and stops it having taken fewer particles
    // than the ray that ran to the end. That is what the counter says.
    const auto [cutThrough, cutN] = trace(0.3F, 0u);
    std::printf("  cut at 0.3: transmittance %.5f through all four, %u particles taken, %u rays cut\n",
                double(cutThrough[0]), cutN[0] - n[0], cutN[2] - n[2]);
    CHECK(cutThrough[0] <= 0.3F);
    CHECK(cutN[2] - n[2] == 1);          // only the ray through all four is cut
    CHECK(cutN[0] - n[0] < stacked + 1); // and it stopped early
}

// A shadow ray is born on a surface, and a surface inside a cloud is
// surrounded by proxies. The primary ray culls back faces because it starts
// at the camera, outside everything; a ray that starts inside a proxy sees
// only that proxy's exit face, so with culling on it misses the particle it
// is standing in -- exactly the particles whose shadow touches the geometry.
// With culling off the proxy offers both faces and the ring collapses them,
// so the particle is taken once, at its peak, if the peak is ahead.
TEST_CASE("a shadow ray born inside a proxy still takes that particle", "[render][rt][gpu][shadow]") {
    LRT_REQUIRE_GPU(gpu);
    if (!render::GaussianRayTracer::hardwareSupported(*gpu->device)) {
        SKIP("the device has no RayQuery or acceleration structures");
    }
    auto h = harness(gpu, on(render::RayTracingRoute::Hardware));
    auto kernel = gpu::ComputeKernel::create(*gpu->library, "lrt/rt/rt_shadow_kernel", "rtShadowRays");
    if (!kernel) FAIL(kernel.error().toString());

    // One large particle at the origin: its proxy reaches about 4.4 units, so
    // a ray starting at z = 1 starts inside it.
    const float opacity = 0.5F;
    CloudBuilder built;
    built.add(0.0F, 0.0F, 0.0F, opacity, 2.0F, 2.0F, 2.0F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    render::RenderTargets targets;
    auto stats = h->rt.render(render::Camera::lookingAt({0.0, 0.0, 12.0}, {0.0, 0.0, 0.0}), instances, settings,
                              targets);
    if (!stats) FAIL(stats.error().toString());
    const render::ShadowScene scene = h->rt.shadowScene();
    REQUIRE(scene.tlas != nullptr);

    // Both rays start inside the proxy. The first has the particle's centre
    // ahead of it (the peak at t = 1), the second behind.
    const std::vector<float> rayData = {
        0.0F, 0.0F, 1.0F, 1e-3F,   0.0F, 0.0F, -1.0F, 100.0F,
        0.0F, 0.0F, 1.0F, 1e-3F,   0.0F, 0.0F, 1.0F, 100.0F,
    };
    gpu::BufferDesc desc;
    desc.bytes = rayData.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "shadow.inside.rays";
    auto rays = gpu::Buffer::create(*gpu->device, desc, rayData.data());
    REQUIRE(rays);
    gpu::BufferDesc out;
    out.bytes = 2 * sizeof(float);
    out.elementBytes = 4;
    out.label = "shadow.inside.transmittance";
    auto shadow = gpu::Buffer::create(*gpu->device, out);
    REQUIRE(shadow);

    const auto trace = [&](uint32_t cull) {
        gpu::Buffer counters = test::uintBuffer(*gpu->device, 4, "shadow.inside.counters");
        {
            gpu::CommandBatch batch(*gpu->device);
            kernel->dispatch(batch, {2, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["scene"].setBinding(scene.tlas);
                cursor["frames"].setBinding(scene.frames->rhi());
                cursor["colours"].setBinding(scene.colours->rhi());
                cursor["instanceData"].setBinding(scene.instanceData->rhi());
                cursor["instanceIndices"].setBinding(scene.instanceIndices->rhi());
                cursor["rays"].setBinding(rays->rhi());
                cursor["shadow"].setBinding(shadow->rhi());
                cursor["counters"].setBinding(counters.rhi());
                rhi::ShaderCursor p = cursor["shadowParams"];
                p["rays"].setData(uint32_t{2});
                p["cut"].setData(1e-3F);
                p["nearZ"].setData(0.0F);
                p["farZ"].setData(1.0e9F);
                p["cull"].setData(cull);
            });
            REQUIRE(batch.submit(true));
        }
        std::array<float, 2> through{};
        std::array<uint32_t, 4> n{};
        REQUIRE(shadow->read(*gpu->device, 0, sizeof(through), through.data()));
        REQUIRE(counters.read(*gpu->device, 0, sizeof(n), n.data()));
        return std::make_pair(through, n);
    };

    const auto [inside, insideN] = trace(0u);
    const auto [culled, culledN] = trace(1u);
    std::printf("  born inside, peak ahead: %.5f (closed form %.5f); peak behind: %.5f\n"
                "  the control, back faces culled: %.5f and %.5f, %u particles taken against %u\n",
                double(inside[0]), double(1.0F - opacity), double(inside[1]), double(culled[0]),
                double(culled[1]), culledN[0], insideN[0]);
    CHECK(std::abs(inside[0] - (1.0F - opacity)) < 2e-3F);   // taken, once, at its peak
    CHECK(inside[1] == 1.0F);                                // the peak behind is not taken
    CHECK(insideN[0] == 1);
    // Born inside, only the exit face is ahead of the origin, so the ring has
    // nothing to collapse here -- unlike a ray that crosses a proxy whole,
    // which is offered both faces (the case above catches five of those).
    CHECK(insideN[1] == 0);
    CHECK(culled[0] == 1.0F);                                // the control: missed entirely
    CHECK(culledN[0] == 0);
}

// The segment query, held to the only thing that makes it a query: a ray cut
// in two and put back together is the ray. The same frame is drawn once as
// [nearZ, farZ] and once as [nearZ, s] composed over [s, farZ] -- the near
// query's transmittance in front of the far one's radiance -- with the cut
// falling through the middle of the cloud, so most rays are split where they
// have particles on both sides.
TEST_CASE("a query cut in two and composed again is the query", "[render][rt][gpu][segment]") {
    LRT_REQUIRE_GPU(gpu);
    if (!render::GaussianRayTracer::hardwareSupported(*gpu->device)) {
        SKIP("the device has no RayQuery or acceleration structures");
    }
    // Two clouds: one sparse enough that no ray carries more than the
    // integrator holds, one dense enough that many do.
    const bool dense = GENERATE(false, true);
    CloudBuilder built = dense ? randomCloud(2000, 11, 0.04F, 0.5F) : randomCloud(600, 11, 0.02F, 0.08F);
    std::printf("  %s cloud\n", dense ? "dense" : "sparse");
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    const render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 7.0}, {0.0, 0.0, 0.0});

    const auto draw = [&](float splitAt, render::RenderTargets& targets) {
        render::RayTracerSettings rtSettings;
        rtSettings.route = render::RayTracingRoute::Hardware;
        rtSettings.splitAt = splitAt;
        auto h = harness(gpu, rtSettings);
        auto cloud = h->loader.upload(built.raw, 3);
        REQUIRE(cloud);
        const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
        auto stats = h->rt.render(camera, instances, settings, targets);
        if (!stats) FAIL(stats.error().toString());
        return stats->renderMs;
    };
    render::RenderTargets whole, composed;
    const double wholeMs = draw(0.0F, whole);
    const double splitMs = draw(7.0F, composed);   // the camera is 7.3 from the centre

    auto same = render::compareImages(*gpu->library, whole.colour, composed.colour, settings.width, settings.height);
    REQUIRE(same);
    std::printf("  one query %.1f ms, two composed %.1f ms; p99 %u, max %u, %llu of %llu pixels over 2\n",
                wholeMs, splitMs, same->p99, same->max, static_cast<unsigned long long>(same->over2),
                static_cast<unsigned long long>(same->pixels));
    // The order is the same either way -- both queries blend by peak -- so a
    // sparse cloud comes out bit for bit. Dense, the cut moves where the
    // carry overflows (kCarry, "the only place order can be approximate"),
    // and a handful of pixels differ: the bound is on how many, as it is
    // against the reference renderer.
    if (dense) {
        CHECK(fewOver2(*same));
        CHECK(same->max <= 1);   // measured: p99 1, max 1, 0 pixels over 2
    } else {
        CHECK(same->max == 0);   // bit for bit
    }
}
