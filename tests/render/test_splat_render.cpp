// Copyright (c) 2026 lucabRTrender contributors.
//
// The tile rasteriser against the GPU reference, and against arithmetic.
#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "lrt/io/RawSplats.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

using namespace lrt;
using test::CloudBuilder;
using test::randomCloud;

namespace {

struct Harness {
    test::Gpu*                 gpu;
    scene::CloudLoader         loader;
    render::TileRasterizer     raster;
    render::ReferenceRenderer  reference;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto reference = render::ReferenceRenderer::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!reference) FAIL(reference.error().toString());
    return std::unique_ptr<Harness>(
        new Harness{gpu, std::move(*loader), std::move(*raster), std::move(*reference)});
}

render::ImageDifference compareToReference(Harness& h, const render::Camera& camera,
                                           std::span<const render::SplatInstance> instances,
                                           const render::RenderSettings& settings) {
    render::RenderTargets ours;
    render::RenderTargets truth;
    auto stats = h.raster.render(camera, instances, settings, ours);
    if (!stats) FAIL(stats.error().toString());
    auto overflow = h.reference.render(camera, instances, settings, truth);
    if (!overflow) FAIL(overflow.error().toString());
    CHECK(*overflow == 0);
    auto diff = render::compareImages(*h.gpu->library, ours.colour, truth.colour, settings.width,
                                      settings.height);
    if (!diff) FAIL(diff.error().toString());
    std::printf("  %u splats, %u pairs: p99 %u, max %u, %llu pixels over 2 of %llu\n",
                stats->splats, stats->pairs, diff->p99, diff->max,
                static_cast<unsigned long long>(diff->over2),
                static_cast<unsigned long long>(diff->pixels));
    return *diff;
}

std::vector<float> readColour(test::Gpu& gpu, const render::RenderTargets& targets) {
    auto values = targets.colour.readAll<float>(*gpu.device);
    REQUIRE(values);
    return std::move(*values);
}

}   // namespace

TEST_CASE("the tile rasteriser renders what the GPU reference renders", "[render][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(3000, 7);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};

    render::RenderSettings settings;
    // Not multiples of 16: the last tile row and column are partial.
    settings.width = 250;
    settings.height = 190;

    SECTION("perspective") {
        render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
        camera.lens.focal = 30.0;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
        CHECK(diff.max <= 16);
    }
    SECTION("close enough that splats cross the frustum edge") {
        render::Camera camera = render::Camera::lookingAt({0.3, -0.2, 2.2}, {0.0, 0.0, 0.0});
        camera.lens.focal = 18.0;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
    }
    SECTION("orthographic") {
        render::Camera camera = render::Camera::lookingAt({0.0, 3.0, 5.0}, {0.0, 0.0, 0.0});
        camera.lens.projection = render::Lens::Projection::Orthographic;
        camera.lens.focal = 5.0;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
    }
    SECTION("no antialiasing, a window offset and a rolled film back") {
        render::Camera camera = render::Camera::lookingAt({-2.0, 0.5, 5.0}, {0.0, 0.0, 0.0});
        camera.lens.windowTranslate[0] = 0.2;
        camera.lens.windowScale[1] = 1.3;
        camera.lens.windowRoll = 25.0;
        settings.antialias = false;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
    }
    SECTION("two instances of the cloud, one turned, scaled and moved") {
        std::vector<render::SplatInstance> two = instances;
        two.push_back({&*cloud, aofx::xform::translation({0.5, 0.0, -1.5}) *
                                    aofx::xform::rotationY(40.0) *
                                    aofx::xform::scaling({1.0, 0.7, 1.3})});
        render::Camera camera = render::Camera::lookingAt({2.0, 1.5, 7.0}, {0.0, 0.0, -0.5});
        const auto diff = compareToReference(*h, camera, two, settings);
        CHECK(diff.p99 <= 2);
    }
}

TEST_CASE("a single splat lands on the pixel the camera arithmetic says", "[render][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder b;
    b.add(1.0F, 0.5F, -10.0F, 0.9F, 0.02F, 0.02F, 0.02F, {1, 0, 0, 0}, {1, 1, 1});
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::Camera camera;   // at the origin, looking down -Z
    camera.lens.focal = camera.lens.haperture;   // fx = width
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    render::RenderTargets targets;
    REQUIRE(h->raster.render(camera, instances, settings, targets));
    const auto pixels = readColour(*gpu, targets);
    size_t best = 0;
    for (size_t i = 0; i < pixels.size() / 4; ++i) {
        if (pixels[i * 4 + 3] > pixels[best * 4 + 3]) {
            best = i;
        }
    }
    // x: 160 + 320 * 1 / 10 = 192; y: 120 + 320 * 0.5 / 10 = 136, bottom-up.
    const int x = static_cast<int>(best % 320);
    const int y = static_cast<int>(best / 320);
    CHECK(std::abs(x - 192) <= 1);
    CHECK(std::abs(y - 136) <= 1);
}

TEST_CASE("the nearer splat wins from either side, and left stays left", "[render][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder b;
    b.add(0.0F, 0.0F, -1.0F, 0.99F, 0.5F, 0.5F, 0.05F, {1, 0, 0, 0}, {1, 0, 0});   // red, z = -1
    b.add(0.0F, 0.0F, 1.0F, 0.99F, 0.5F, 0.5F, 0.05F, {1, 0, 0, 0}, {0, 0, 1});    // blue, z = +1
    b.add(-3.0F, 0.0F, 0.0F, 0.99F, 0.3F, 0.3F, 0.3F, {1, 0, 0, 0}, {0, 1, 0});    // green, -x
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 100;
    settings.linearise = false;

    const auto centre = [&](const std::vector<float>& p) {
        const size_t at = (50 * 200 + 100) * 4;
        return std::array<float, 3>{p[at], p[at + 1], p[at + 2]};
    };
    const auto brightestX = [&](const std::vector<float>& p, int channel) {
        int bestX = 0;
        float best = -1.0F;
        for (int x = 0; x < 200; ++x) {
            const float v = p[static_cast<size_t>((50 * 200 + x) * 4 + channel)];
            if (v > best) {
                best = v;
                bestX = x;
            }
        }
        return bestX;
    };

    render::RenderTargets targets;
    // From +Z, looking down -Z: blue (z = +1) is in front; -x is on the left.
    render::Camera front = render::Camera::lookingAt({0.0, 0.0, 10.0}, {0.0, 0.0, 0.0});
    REQUIRE(h->raster.render(front, instances, settings, targets));
    auto pixels = readColour(*gpu, targets);
    CHECK(centre(pixels)[2] > 0.9F);
    CHECK(centre(pixels)[0] < 0.1F);
    CHECK(brightestX(pixels, 1) < 100);

    // From -Z, looking down +Z: red is in front; -x is now on the right.
    render::Camera back = render::Camera::lookingAt({0.0, 0.0, -10.0}, {0.0, 0.0, 0.0});
    REQUIRE(h->raster.render(back, instances, settings, targets));
    pixels = readColour(*gpu, targets);
    CHECK(centre(pixels)[0] > 0.9F);
    CHECK(centre(pixels)[2] < 0.1F);
    CHECK(brightestX(pixels, 1) > 100);
}

TEST_CASE("antialiasing pays back the energy dilation adds to a sub-pixel splat", "[render][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder b;
    b.add(0.0F, 0.0F, -50.0F, 0.9F, 0.002F, 0.002F, 0.002F, {1, 0, 0, 0}, {1, 1, 1});
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::Camera camera;
    render::RenderSettings settings;
    settings.width = 64;
    settings.height = 64;
    const auto energy = [&](bool antialias) {
        settings.antialias = antialias;
        render::RenderTargets targets;
        REQUIRE(h->raster.render(camera, instances, settings, targets));
        const auto p = readColour(*gpu, targets);
        double sum = 0.0;
        for (size_t i = 0; i < p.size() / 4; ++i) {
            sum += static_cast<double>(p[i * 4 + 3]);
        }
        return sum;
    };
    const double with = energy(true);
    const double without = energy(false);
    CHECK(without > 0.0);
    CHECK(with < 0.6 * without);
}

TEST_CASE("a SplatEdit renders as the GPU reference renders it", "[render][gpu][edit]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(3000, 29);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    render::SplatEdit edit;
    edit.active = true;

    SECTION("keep what is inside a box") {
        edit.mode = render::SplatEdit::Mode::Keep;
        edit.centre = {0.5F, 0.0F, 0.0F};
        edit.size = {1.0F, 0.8F, 1.5F};
    }
    SECTION("remove what is inside a sphere, and the grade does nothing to the rest") {
        edit.mode = render::SplatEdit::Mode::Remove;
        edit.shape = render::SplatEdit::Shape::Sphere;
        edit.size = {1.2F, 0.0F, 0.0F};
        edit.tint = {0.2F, 1.0F, 1.0F};
    }
    SECTION("grade inside an inverted box: tint, saturation, brightness, opacity") {
        edit.mode = render::SplatEdit::Mode::Grade;
        edit.invert = true;
        edit.size = {1.0F, 1.0F, 1.0F};
        edit.tint = {1.0F, 0.6F, 0.3F};
        edit.saturation = 0.3F;
        edit.brightness = 1.4F;
        edit.opacity = 0.5F;
    }
    SECTION("the haze filters, wherever the volume is") {
        edit.mode = render::SplatEdit::Mode::Grade;
        edit.size = {100.0F, 100.0F, 100.0F};
        edit.minOpacity = 0.4F;
        edit.maxScale = 0.2F;
    }
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity(), edit}};
    const auto diff = compareToReference(*h, camera, instances, settings);
    CHECK(diff.p99 <= 2);

    // And the edit changed the picture.
    render::RenderTargets edited, plain;
    REQUIRE(h->raster.render(camera, instances, settings, edited));
    const std::vector<render::SplatInstance> untouched{{&*cloud, render::Mat4::identity()}};
    REQUIRE(h->raster.render(camera, untouched, settings, plain));
    auto changed = render::compareImages(*gpu->library, edited.colour, plain.colour, settings.width, settings.height);
    REQUIRE(changed);
    CHECK(changed->over2 > changed->pixels / 20);
}

TEST_CASE("a SplatEdit is in the cloud's own space and per instance", "[render][gpu][edit]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // One opaque white splat at the cloud's origin.
    CloudBuilder b;
    b.add(0.0F, 0.0F, 0.0F, 0.95F, 0.3F, 0.3F, 0.3F, {1, 0, 0, 0}, {1, 1, 1});
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 65;
    settings.height = 65;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0});
    // Two instances side by side; the edit's box is around the cloud origin,
    // which the left instance's transform moves with it.
    render::SplatEdit keepNothing;
    keepNothing.active = true;
    keepNothing.mode = render::SplatEdit::Mode::Keep;
    keepNothing.centre = {5.0F, 0.0F, 0.0F};   // far from the splat: nothing kept
    render::SplatEdit greenOnly;
    greenOnly.active = true;
    greenOnly.mode = render::SplatEdit::Mode::Grade;
    greenOnly.size = {1.0F, 1.0F, 1.0F};
    greenOnly.tint = {0.0F, 1.0F, 0.0F};
    const std::vector<render::SplatInstance> instances{
        {&*cloud, aofx::xform::translation({-1.0, 0.0, 0.0}), keepNothing},
        {&*cloud, aofx::xform::translation({1.0, 0.0, 0.0}), greenOnly}};
    render::RenderTargets targets;
    REQUIRE(h->raster.render(camera, instances, settings, targets));
    const auto p = readColour(*gpu, targets);
    // 65 px over a 24.576 aperture at 50mm is 132 px per unit at distance 1:
    // x = +-1 at 5 units lands 26 px either side of the centre.
    const auto at = [&](int x, int y) { return &p[static_cast<size_t>((y * 65 + x) * 4)]; };
    const float* left = at(32 - 26, 32);
    const float* right = at(32 + 26, 32);
    CHECK(left[3] < 0.01F);              // removed
    CHECK(right[3] > 0.5F);
    CHECK(right[0] < 0.01F);             // tinted green
    CHECK(right[2] < 0.01F);
    CHECK(right[1] > 0.3F);
}
