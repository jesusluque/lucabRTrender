// Copyright (c) 2026 lucabRTrender contributors.
//
// SOG files written by PlayCanvas's own converter render as the PLYs they came
// from. Renders, not splats: the converter reorders splats spatially and
// clusters harmonics into a palette, and a render is indifferent to order.
// Both decodes and the comparison run on the GPU.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <utility>
#include <vector>

#include "lrt/io/Readers.h"
#include "lrt/io/Sog.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

using namespace lrt;
namespace fs = std::filesystem;

// Measured and set just above. tiny: p99 5 (SOG's 8-bit colour and scale
// codebooks, the same allowance openFXplayer's own round-trip test gives).
// sh3: p99 1 without harmonics and 1 with them, its 64-entry palette being
// one entry a splat; with the palette's red and blue exchanged, 170.
constexpr uint32_t kFlatP99 = 6;
constexpr uint32_t kShadedP99 = 3;

namespace {

fs::path fixture(const char* name) {
    return fs::path(LRT_TEST_DATA_DIR) / "splats" / name;
}

struct Pair {
    scene::GpuSplats ply;
    scene::GpuSplats sog;
};

render::ImageDifference compareRenders(test::Gpu& gpu, const scene::GpuSplats& a, const scene::GpuSplats& b,
                                       uint32_t shDegree) {
    auto raster = render::TileRasterizer::create(*gpu.library);
    REQUIRE(raster);
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    settings.maxShDegree = shDegree;
    // Framed on the PLY's bounds: the cloud fills the picture.
    const auto& lo = a.bounds.min;
    const auto& hi = a.bounds.max;
    const render::Vec3 centre{(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5};
    const double extent = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    render::Camera camera = render::Camera::lookingAt(centre + render::Vec3{0.3 * extent, 0.2 * extent, 1.6 * extent},
                                                      centre);
    camera.lens.focal = 30.0;
    render::RenderTargets ta, tb;
    REQUIRE(raster->render(camera, std::vector<render::SplatInstance>{{&a, render::Mat4::identity()}}, settings, ta));
    REQUIRE(raster->render(camera, std::vector<render::SplatInstance>{{&b, render::Mat4::identity()}}, settings, tb));
    auto diff = render::compareImages(*gpu.library, ta.colour, tb.colour, settings.width, settings.height);
    REQUIRE(diff);
    return *diff;
}

}   // namespace

TEST_CASE("SOG from PlayCanvas's converter renders as the PLY it was made from", "[render][io][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    const auto load = [&](const char* name) {
        auto splats = scene::loadSplatFile(*loader, fixture(name), 3);
        if (!splats) {
            FAIL(splats.error().toString());
        }
        return std::move(*splats);
    };

    SECTION("version 2, no harmonics") {
        const scene::GpuSplats ply = load("tiny.ply");
        const scene::GpuSplats sog = load("tiny.sog");
        CHECK(sog.count == ply.count);
        const auto diff = compareRenders(*gpu, ply, sog, 0);
        std::printf("  tiny: p99 %u, max %u\n", diff.p99, diff.max);
        CHECK(diff.p99 <= kFlatP99);
    }

    SECTION("version 2, degree-3 harmonics through the palette") {
        const scene::GpuSplats ply = load("sh3.ply");
        const scene::GpuSplats sog = load("sh3.sog");
        CHECK(sog.count == ply.count);
        CHECK(sog.degree() == 3);
        const auto flat = compareRenders(*gpu, ply, sog, 0);
        const auto shaded = compareRenders(*gpu, ply, sog, 3);
        std::printf("  sh3: p99 %u (max %u) without harmonics, %u (max %u) with\n", flat.p99, flat.max,
                    shaded.p99, shaded.max);
        CHECK(flat.p99 <= kFlatP99);
        CHECK(shaded.p99 <= kShadedP99);

        // The control: the same palette with red and blue exchanged, which is
        // what reading its channels in the wrong order would do. The real
        // decode must be far closer than that.
        auto sogRaw = io::readSog(fixture("sh3.sog"));
        REQUIRE(sogRaw);
        for (uint32_t& texel : sogRaw->shCentroids.texels) {
            texel = (texel & 0xFF00FF00u) | ((texel >> 16) & 0xFFu) | ((texel & 0xFFu) << 16);
        }
        auto swapped = loader->upload(*sogRaw, 3);
        REQUIRE(swapped);
        const auto control = compareRenders(*gpu, ply, *swapped, 3);
        std::printf("  sh3 with the palette's red and blue exchanged: p99 %u\n", control.p99);
        CHECK(control.p99 >= 2 * shaded.p99 + 4);
    }
}
