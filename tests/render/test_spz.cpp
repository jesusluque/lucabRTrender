// Copyright (c) 2026 lucabRTrender contributors.
//
// An SPZ written by Niantic's own packer, from the same Gaussians a PLY
// carries, renders as that PLY does: the check that this engine's reading of
// SPZ's quantisation and axes is Niantic's, not only this repository's
// reading of the format. Niantic's code only writes the fixture; the
// comparison is two GPU renders compared on the GPU.
#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "load-spz.h"
#include "lrt/io/Readers.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

using namespace lrt;
namespace fs = std::filesystem;

// Measured on these 1500 splats and set just above: p99 3 without harmonics,
// then 4, 10 and 13 through degrees 1, 2 and 3 -- SPZ keeps degree 1 in 5 bits
// and the bands above in 4. With the signs of bands 2 and 3 deliberately
// wrong the same comparison gave 136 and 234, so a sign error cannot hide
// inside these bounds.
constexpr uint32_t kFlatP99 = 4;
constexpr uint32_t kShadedP99 = 16;

TEST_CASE("an SPZ from Niantic's packer renders as the PLY it was packed from", "[render][io][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    const uint32_t version = GENERATE(3u, 4u);   // gzip, zstd
    std::printf("  SPZ version %u\n", version);

    // Degree-3 records in the 3DGS PLY encoding: x y z, logit opacity, log
    // scales, w x y z, SH DC, 45 rest all red then green then blue. Degree 3
    // so every band's change of sign under the axis turn is exercised.
    test::CloudBuilder built = test::randomCloud(1500, 17, 0.02F, 0.4F);
    constexpr size_t kStride = 14 + 45;
    io::RawSplats raw;
    raw.source = "synthetic degree 3";
    raw.count = built.raw.count;
    raw.encoding = built.raw.encoding;
    raw.encoding.floatsPerRecord = kStride;
    raw.encoding.restPerColour = 15;
    raw.records.assign(size_t{raw.count} * kStride, 0.0F);
    test::Lcg rng;
    for (uint32_t i = 0; i < raw.count; ++i) {
        const float* from = built.raw.records.data() + i * built.raw.encoding.floatsPerRecord;
        float* to = raw.records.data() + i * kStride;
        std::copy(from, from + 14, to);
        for (size_t k = 0; k < 45; ++k) {
            // Large enough to survive 4-bit quantisation of the upper bands.
            to[14 + k] = rng.range(-0.6F, 0.6F);
        }
    }

    spz::GaussianCloud cloud;
    cloud.numPoints = static_cast<int32_t>(raw.count);
    cloud.shDegree = 3;
    for (uint32_t i = 0; i < raw.count; ++i) {
        const float* r = raw.records.data() + i * kStride;
        cloud.positions.insert(cloud.positions.end(), {r[0], r[1], r[2]});
        cloud.alphas.push_back(r[3]);
        cloud.scales.insert(cloud.scales.end(), {r[4], r[5], r[6]});
        cloud.rotations.insert(cloud.rotations.end(), {r[8], r[9], r[10], r[7]});   // x y z w
        cloud.colors.insert(cloud.colors.end(), {r[11], r[12], r[13]});
        for (int basis = 0; basis < 15; ++basis) {
            for (int colour = 0; colour < 3; ++colour) {
                cloud.sh.push_back(r[14 + colour * 15 + basis]);   // rgb per basis
            }
        }
    }
    spz::PackOptions options;
    options.version = version;
    options.from = spz::CoordinateSystem::RDF;   // what a 3DGS PLY is in
    const fs::path path = fs::temp_directory_path() / "lrt-tests" / ("packed-v" + std::to_string(version) + ".spz");
    fs::create_directories(path.parent_path());
    REQUIRE(spz::saveSpz(cloud, options, path.string()));

    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto fromPly = loader->upload(raw, 3);
    REQUIRE(fromPly);
    auto spzRaw = io::readSplats(path);
    if (!spzRaw) {
        FAIL(spzRaw.error().toString());
    }
    auto fromSpz = loader->upload(*spzRaw, 3);
    REQUIRE(fromSpz);
    CHECK(fromSpz->count == fromPly->count);
    CHECK(fromSpz->degree() == 3);

    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    const auto compare = [&](uint32_t shDegree) {
        settings.maxShDegree = shDegree;
        render::RenderTargets a, b;
        REQUIRE(raster->render(camera, std::vector<render::SplatInstance>{{&*fromPly, render::Mat4::identity()}},
                               settings, a));
        REQUIRE(raster->render(camera, std::vector<render::SplatInstance>{{&*fromSpz, render::Mat4::identity()}},
                               settings, b));
        auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
        REQUIRE(diff);
        std::printf("  PLY vs SPZ, harmonics to degree %u: p99 %u, max %u, %llu of %llu pixels over 2\n",
                    shDegree, diff->p99, diff->max, static_cast<unsigned long long>(diff->over2),
                    static_cast<unsigned long long>(diff->pixels));
        return *diff;
    };
    // Base colour, shape and position alone: 8-bit colour, 1/16-stop scales
    // and 12-bit fixed point cost next to nothing.
    const auto flat = compare(0);
    // Band by band, what their quantisation costs (5 bits for degree 1, 4
    // above). A wrong sign on any basis costs tens of code values.
    const auto first = compare(1);
    const auto second = compare(2);
    const auto third = compare(3);
    CHECK(flat.p99 <= kFlatP99);
    CHECK(first.p99 <= kShadedP99);
    CHECK(second.p99 <= kShadedP99);
    CHECK(third.p99 <= kShadedP99);
}
