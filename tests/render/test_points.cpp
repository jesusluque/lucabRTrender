// Copyright (c) 2026 lucabRTrender contributors.
//
// Points: discs against the GPU reference, the raster route against the disc
// route, splats composited over rasterised points against one mixed sort.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "lrt/io/Exr.h"
#include "lrt/io/RawSplats.h"
#include "lrt/render/PointRasterizer.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

using namespace lrt;

namespace {

struct Lcg {
    uint64_t state = 0x2545F4914F6CDD1DULL;
    float next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0F;
    }
    float range(float lo, float hi) { return lo + (hi - lo) * next(); }
};

io::RawPoints randomPoints(uint32_t count, uint64_t seed) {
    io::RawPoints raw;
    raw.source = "synthetic points";
    raw.colourKind = 2;
    Lcg rng;
    rng.state ^= seed;
    for (uint32_t i = 0; i < count; ++i) {
        raw.records.insert(raw.records.end(),
                           {rng.range(-2, 2), rng.range(-1.5F, 1.5F), rng.range(-2, 2),
                            rng.range(0, 1), rng.range(0, 1), rng.range(0, 1)});
    }
    raw.count = count;
    return raw;
}

struct Harness {
    test::Gpu*                gpu;
    scene::CloudLoader        loader;
    render::TileRasterizer    raster;
    render::ReferenceRenderer reference;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto reference = render::ReferenceRenderer::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!reference) FAIL(reference.error().toString());
    return std::unique_ptr<Harness>(new Harness{gpu, std::move(*loader), std::move(*raster), std::move(*reference)});
}

render::ImageDifference difference(test::Gpu& gpu, const render::RenderTargets& a,
                                   const render::RenderTargets& b) {
    auto diff = render::compareImages(*gpu.library, a.colour, b.colour, a.width, a.height);
    if (!diff) FAIL(diff.error().toString());
    std::printf("  p99 %u, max %u, %llu pixels over 2 of %llu\n", diff->p99, diff->max,
                static_cast<unsigned long long>(diff->over2),
                static_cast<unsigned long long>(diff->pixels));
    return *diff;
}

}   // namespace

TEST_CASE("points drawn as discs match the GPU reference", "[render][points][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    auto cloud = h->loader.upload(randomPoints(4000, 3));
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 230;
    settings.height = 170;
    settings.linearise = false;
    render::Camera camera = render::Camera::lookingAt({1.0, 2.0, 6.0}, {0.0, 0.0, 0.0});

    for (const auto mode : {render::PointStyle::Size::World, render::PointStyle::Size::Pixels}) {
        render::PointInstance points{&*cloud, render::Mat4::identity(), {}};
        points.style.sizeMode = mode;
        points.style.size = mode == render::PointStyle::Size::World ? 0.06F : 5.0F;
        std::vector<render::PointInstance> list{points};
        render::RenderTargets ours, truth;
        REQUIRE(h->raster.render(camera, {}, settings, ours, list));
        auto overflow = h->reference.render(camera, {}, settings, truth, list);
        REQUIRE(overflow);
        CHECK(*overflow == 0);
        const auto diff = difference(*gpu, ours, truth);
        CHECK(diff.p99 <= 2);
    }
}

TEST_CASE("the raster route draws what the disc route draws", "[render][points][gpu][raster]") {
    LRT_REQUIRE_GPU(gpu);
    auto points = render::PointRasterizer::create(*gpu->library);
    if (!points) {
        SKIP(points.error().toString());
    }
    auto h = harness(gpu);
    auto cloud = h->loader.upload(randomPoints(20000, 11));
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 200;
    settings.linearise = false;
    render::Camera camera = render::Camera::lookingAt({-1.0, 1.0, 5.0}, {0.0, 0.0, 0.0});
    render::PointInstance instance{&*cloud, render::Mat4::identity(), {}};
    instance.style.size = 0.04F;
    std::vector<render::PointInstance> list{instance};

    render::RenderTargets rasterised, discs;
    REQUIRE(points->render(camera, list, settings, rasterised));
    REQUIRE(h->raster.render(camera, {}, settings, discs, list));
    if (std::getenv("LRT_TEST_DUMP") != nullptr) {
        {
            rhi::ComPtr<ISlangBlob> blob;
            rhi::SubresourceLayout layout;
            REQUIRE(SLANG_SUCCEEDED(gpu->device->rhi()->readTexture(points->colourTarget(), 0, 0, blob.writeRef(), &layout)));
            std::printf("texture layout: %u x %u, rowPitch %zu, bytes %zu\n", layout.size.width, layout.size.height,
                        static_cast<size_t>(layout.rowPitch), static_cast<size_t>(blob->getBufferSize()));
            std::vector<float> flipped(size_t{settings.width} * settings.height * 4);
            const auto* data = static_cast<const float*>(blob->getBufferPointer());
            for (uint32_t y = 0; y < settings.height; ++y) {
                for (uint32_t x = 0; x < settings.width; ++x) {
                    const size_t from = (size_t{y} * layout.rowPitch) / sizeof(float) + x * 4;
                    const size_t to = (size_t{settings.height - 1 - y} * settings.width + x) * 4;
                    for (int c = 0; c < 4; ++c) flipped[to + static_cast<size_t>(c)] = data[from + static_cast<size_t>(c)];
                }
            }
            REQUIRE((io::writeExr(std::string(std::getenv("LRT_TEST_DUMP")) + "/texture.exr", settings.width, settings.height, flipped)));
        }
        for (auto [name, t] : {std::pair{"raster", &rasterised}, std::pair{"discs", &discs}}) {
            auto c = t->colour.readAll<float>(*gpu->device);
            auto d = t->depth.readAll<float>(*gpu->device);
            REQUIRE((io::writeExr(std::string(std::getenv("LRT_TEST_DUMP")) + "/" + name + ".exr",
                                  t->width, t->height, *c, *d)));
        }
    }
    const auto diff = difference(*gpu, rasterised, discs);
    // Edge pixels may differ where two points at almost the same depth
    // overlap and the z-buffer and the sort break the tie differently.
    CHECK(diff.p99 <= 2);
}

TEST_CASE("splats over rasterised points match splats and discs in one sort",
          "[render][points][gpu][raster]") {
    LRT_REQUIRE_GPU(gpu);
    auto points = render::PointRasterizer::create(*gpu->library);
    if (!points) {
        SKIP(points.error().toString());
    }
    auto h = harness(gpu);
    auto pointCloud = h->loader.upload(randomPoints(8000, 5));
    REQUIRE(pointCloud);

    // A few large translucent splats in front of and among the points.
    io::RawSplats raw;
    raw.source = "fog";
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 14;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    e.colour = io::SplatEncoding::Colour::Linear;
    Lcg rng;
    for (int i = 0; i < 40; ++i) {
        raw.records.insert(raw.records.end(),
                           {rng.range(-2, 2), rng.range(-1, 1), rng.range(-2, 3), rng.range(0.1F, 0.6F),
                            rng.range(0.1F, 0.5F), rng.range(0.1F, 0.5F), rng.range(0.1F, 0.5F),
                            1, 0, 0, 0, rng.range(0, 1), rng.range(0, 1), rng.range(0, 1)});
        raw.count += 1;
    }
    auto splats = h->loader.upload(raw);
    REQUIRE(splats);

    render::RenderSettings settings;
    settings.width = 256;
    settings.height = 180;
    render::Camera camera = render::Camera::lookingAt({0.5, 0.5, 7.0}, {0.0, 0.0, 0.0});
    const std::vector<render::SplatInstance> splatList{{&*splats, render::Mat4::identity()}};
    render::PointInstance instance{&*pointCloud, render::Mat4::identity(), {}};
    instance.style.size = 0.05F;
    const std::vector<render::PointInstance> pointList{instance};

    render::RenderTargets layer, composited, mixed;
    REQUIRE(points->render(camera, pointList, settings, layer));
    REQUIRE(h->raster.render(camera, splatList, settings, composited, {}, &layer));
    REQUIRE(h->raster.render(camera, splatList, settings, mixed, pointList));
    const auto diff = difference(*gpu, composited, mixed);
    CHECK(diff.p99 <= 2);
}

TEST_CASE("eye-dome lighting darkens a depth edge and leaves a flat wall alone",
          "[render][points][gpu][raster]") {
    LRT_REQUIRE_GPU(gpu);
    auto points = render::PointRasterizer::create(*gpu->library);
    if (!points) {
        SKIP(points.error().toString());
    }
    auto h = harness(gpu);
    // A wall at z = 0 and a nearer square at z = 1 covering its left half.
    io::RawPoints raw;
    raw.colourKind = 0;
    for (int y = -50; y <= 50; ++y) {
        for (int x = -50; x <= 50; ++x) {
            raw.records.insert(raw.records.end(), {x * 0.04F, y * 0.04F, 0.0F, 1, 1, 1});
            if (x < 0) {
                raw.records.insert(raw.records.end(), {x * 0.02F, y * 0.02F, 1.0F, 1, 1, 1});
            }
            raw.count += x < 0 ? 2 : 1;
        }
    }
    auto cloud = h->loader.upload(raw);
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 200;
    settings.linearise = false;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 6.0}, {0.0, 0.0, 0.0});
    render::PointInstance instance{&*cloud, render::Mat4::identity(), {}};
    instance.style.size = 0.08F;
    instance.style.edlStrength = 1.0F;
    instance.style.edlRadius = 2.0F;
    render::RenderTargets targets;
    REQUIRE(points->render(camera, std::vector<render::PointInstance>{instance}, settings, targets));
    auto colour = targets.colour.readAll<float>(*gpu->device);
    auto depth = targets.depth.readAll<float>(*gpu->device);
    REQUIRE(colour);
    REQUIRE(depth);
    // Find, on the middle row, the wall pixel right beside the near square's
    // edge, and a wall pixel far from it.
    const int row = 100;
    int edge = -1;
    for (int x = 1; x < 199; ++x) {
        const float here = (*depth)[static_cast<size_t>(row * 200 + x)];
        const float left = (*depth)[static_cast<size_t>(row * 200 + x - 1)];
        if (here > 0.0F && left > 0.0F && here > left + 0.5F) {
            edge = x;
            break;
        }
    }
    REQUIRE(edge > 0);
    const float atEdge = (*colour)[static_cast<size_t>((row * 200 + edge) * 4)];
    const float flat = (*colour)[static_cast<size_t>((row * 200 + 170) * 4)];
    CHECK(flat > 0.95F);
    CHECK(atEdge < 0.9F * flat);
}

TEST_CASE("surface splatting blends points on one surface instead of picking one",
          "[render][points][gpu][raster]") {
    LRT_REQUIRE_GPU(gpu);
    auto points = render::PointRasterizer::create(*gpu->library);
    if (!points) {
        SKIP(points.error().toString());
    }
    auto h = harness(gpu);
    io::RawPoints raw;
    raw.colourKind = 2;
    raw.records = {-0.05F, 0.0F, 0.0F, 1, 0, 0,     // red
                   0.05F, 0.0F, 0.001F, 0, 0, 1};   // blue, a hair nearer
    raw.count = 2;
    auto cloud = h->loader.upload(raw);
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 64;
    settings.height = 64;
    settings.linearise = false;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 2.0}, {0.0, 0.0, 0.0});
    render::PointInstance instance{&*cloud, render::Mat4::identity(), {}};
    instance.style.size = 0.4F;
    const auto centre = [&](float offset) {
        instance.style.surfaceDepthOffset = offset;
        render::RenderTargets targets;
        REQUIRE(points->render(camera, std::vector<render::PointInstance>{instance}, settings, targets));
        auto colour = targets.colour.readAll<float>(*gpu->device);
        REQUIRE(colour);
        const size_t at = (32 * 64 + 32) * 4;
        return std::array<float, 3>{(*colour)[at], (*colour)[at + 1], (*colour)[at + 2]};
    };
    const auto hard = centre(0.0F);
    CHECK(hard[2] > 0.99F);   // opaque: the nearer (blue) wins outright
    const auto soft = centre(0.05F);
    CHECK(soft[0] > 0.3F);    // blended: red shows through
    CHECK(soft[2] > 0.3F);
}
