// Copyright (c) 2026 lucabRTrender contributors.
//
// Loading: files written by hand with values whose decoded form is known
// exactly, decoded on the GPU, and unpacked on the GPU for comparison.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#ifdef LRT_TEST_SPZ
#include <zlib.h>
#endif

#include "lrt/io/Readers.h"
#include "lrt/scene/GpuClouds.h"

using namespace lrt;
using Catch::Approx;
namespace fs = std::filesystem;

namespace {

fs::path scratchFile(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "lrt-tests";
    fs::create_directories(dir);
    return dir / name;
}

/// One 3DGS record's properties, in file order, degree 1 (9 rest coefficients).
struct PlyRecord {
    float x, y, z, nx, ny, nz;
    float dc[3];
    float rest[9];
    float opacity;
    float scale[3];
    float rot[4];   // w x y z
};

void writeSplatPly(const fs::path& path, const std::vector<PlyRecord>& records) {
    std::ofstream out(path, std::ios::binary);
    out << "ply\nformat binary_little_endian 1.0\n"
        << "element vertex " << records.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property float nx\nproperty float ny\nproperty float nz\n"
        << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    for (int k = 0; k < 9; ++k) {
        out << "property float f_rest_" << k << "\n";
    }
    out << "property float opacity\n"
        << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        << "end_header\n";
    out.write(reinterpret_cast<const char*>(records.data()),
              static_cast<std::streamsize>(records.size() * sizeof(PlyRecord)));
}

std::array<float, 20> inspect(test::Gpu& gpu, const scene::GpuSplats& splats, uint32_t index) {
    static gpu::ComputeKernel kInspect = test::kernel(gpu, "lrt/test/splat_inspect");
    gpu::BufferDesc desc;
    desc.bytes = 20 * sizeof(float);
    desc.elementBytes = sizeof(float);
    auto values = gpu::Buffer::create(*gpu.device, desc);
    REQUIRE(values);
    gpu::CommandBatch batch(*gpu.device);
    kInspect.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(splats.positions.rhi());
        cursor["shape"].setBinding(splats.shape.rhi());
        cursor["sh"].setBinding(splats.sh.rhi());
        cursor["values"].setBinding(values->rhi());
        cursor["params"]["index"].setData(index);
        cursor["params"]["shWords"].setData(splats.shWords);
        cursor["params"]["hasSh"].setData(uint32_t{splats.restPerColour > 0 ? 1u : 0u});
    });
    REQUIRE(batch.submit(true));
    std::array<float, 20> out{};
    REQUIRE(values->read(*gpu.device, 0, sizeof(out), out.data()));
    return out;
}

}   // namespace

TEST_CASE("a 3DGS PLY decodes opacity, scale, rotation, colour and harmonics exactly",
          "[scene][io][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // logit(0) = 0.5; log scales 0, ln2, -ln2; quaternion (w=2, x=0, y=0, z=0)
    // unnormalised -> identity; f_dc 0 -> base 0.5; rest rrr ggg bbb.
    PlyRecord first{};
    first.x = 1; first.y = 2; first.z = 3;
    first.opacity = 0.0F;
    first.scale[0] = 0.0F; first.scale[1] = 0.6931472F; first.scale[2] = -0.6931472F;
    first.rot[0] = 2.0F;
    for (int k = 0; k < 9; ++k) {
        first.rest[k] = 0.1F * static_cast<float>(k + 1);   // r: .1 .2 .3  g: .4 .5 .6  b: .7 .8 .9
    }
    // Invisible: logit -10 is opacity 4.5e-5 < 1/255. Dropped.
    PlyRecord faint = first;
    faint.opacity = -10.0F;
    // Broken: a NaN position. Dropped.
    PlyRecord broken = first;
    broken.x = nan;
    // Kept, after two dropped ones: compaction must keep file order.
    PlyRecord last = first;
    last.x = -4; last.y = 5; last.z = -6;
    last.opacity = 1.0986123F;   // logit(0.75)
    last.rot[0] = 0.0F; last.rot[3] = -3.0F;   // 180 degrees about z, sign folded
    last.dc[0] = 1.0F / 0.28209479F;   // base colour 1.5

    const fs::path path = scratchFile("decode.ply");
    writeSplatPly(path, {first, faint, broken, last});

    auto raw = io::readSplats(path);
    REQUIRE(raw);
    CHECK(raw->count == 4);
    CHECK(raw->encoding.restPerColour == 3);
    auto loader = scene::CloudLoader::create(*gpu->library);
    if (!loader) {
        FAIL(loader.error().toString());
    }
    auto splats = loader->upload(*raw, 3);
    REQUIRE(splats);
    CHECK(splats->count == 2);
    CHECK(splats->degree() == 1);

    const auto a = inspect(*gpu, *splats, 0);
    CHECK(a[0] == 1.0F);
    CHECK(a[1] == 2.0F);
    CHECK(a[2] == 3.0F);
    CHECK(a[3] == Approx(0.5).margin(1e-6));
    CHECK(a[4] == Approx(1.0).epsilon(2e-3));
    CHECK(a[5] == Approx(2.0).epsilon(2e-3));
    CHECK(a[6] == Approx(0.5).epsilon(2e-3));
    CHECK(a[10] == Approx(1.0).margin(1e-3));   // w
    CHECK(a[11] == Approx(0.5).margin(1e-3));   // base colour r
    // rgb per basis: basis 1 is (r .1, g .4, b .7), basis 2 starts (r .2, ...)
    CHECK(a[14] == Approx(0.1).margin(1e-3));
    CHECK(a[15] == Approx(0.4).margin(1e-3));
    CHECK(a[16] == Approx(0.7).margin(1e-3));
    CHECK(a[17] == Approx(0.2).margin(1e-3));

    const auto b = inspect(*gpu, *splats, 1);
    CHECK(b[0] == -4.0F);
    CHECK(b[3] == Approx(0.75).margin(1e-5));
    CHECK(std::abs(b[9]) == Approx(1.0).margin(1e-3));   // |z| = 1
    CHECK(b[11] == Approx(1.5).margin(1e-3));

    CHECK(splats->bounds.min[0] == -4.0F);
    CHECK(splats->bounds.max[0] == 1.0F);
    CHECK(splats->bounds.min[2] == -6.0F);
    CHECK(splats->bounds.max[2] == 3.0F);

    // Degree capped at 0: no harmonics kept.
    auto flat = loader->upload(*raw, 0);
    REQUIRE(flat);
    CHECK(flat->degree() == 0);
}

TEST_CASE("a .splat file decodes byte-encoded opacity, colour and rotation", "[scene][io][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    struct Record {
        float pos[3];
        float scale[3];
        unsigned char rgba[4];
        unsigned char rot[4];
    };
    Record r{{0.5F, -0.5F, 2.0F}, {0.25F, 0.5F, 1.0F}, {255, 0, 51, 255}, {255, 128, 128, 128}};
    const fs::path path = scratchFile("decode.splat");
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&r), sizeof r);
    }
    auto raw = io::readSplats(path);
    REQUIRE(raw);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(*raw);
    REQUIRE(splats);
    REQUIRE(splats->count == 1);
    const auto v = inspect(*gpu, *splats, 0);
    CHECK(v[3] == Approx(1.0).margin(1e-6));
    CHECK(v[4] == Approx(0.25).epsilon(2e-3));
    CHECK(v[6] == Approx(1.0).epsilon(2e-3));
    CHECK(v[10] == Approx(1.0).margin(1e-2));   // w from byte 255 -> 0.992, normalised
    CHECK(v[11] == Approx(1.0).margin(1e-3));
    CHECK(v[13] == Approx(0.2).margin(1e-3));
}

#ifdef LRT_TEST_SPZ
namespace {

/// A legacy SPZ (v2/v3): a 16-byte header and the attribute streams, gzipped
/// together -- written here byte by byte, so the test knows every value.
void writeSpz(const fs::path& path, uint32_t version, uint32_t points, uint8_t shDegree,
              uint8_t fractionalBits, const std::vector<uint8_t>& streams) {
    std::vector<uint8_t> bytes(16, 0);
    const uint32_t magic = 0x5053474e;
    std::memcpy(bytes.data(), &magic, 4);
    std::memcpy(bytes.data() + 4, &version, 4);
    std::memcpy(bytes.data() + 8, &points, 4);
    bytes[12] = shDegree;
    bytes[13] = fractionalBits;
    bytes.insert(bytes.end(), streams.begin(), streams.end());
    gzFile out = gzopen(path.string().c_str(), "wb");
    REQUIRE(out != nullptr);
    REQUIRE(gzwrite(out, bytes.data(), static_cast<unsigned>(bytes.size())) == static_cast<int>(bytes.size()));
    gzclose(out);
}

void appendInt24(std::vector<uint8_t>& out, int32_t v) {
    const auto u = static_cast<uint32_t>(v);
    out.push_back(static_cast<uint8_t>(u & 0xFF));
    out.push_back(static_cast<uint8_t>((u >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((u >> 16) & 0xFF));
}

}   // namespace

TEST_CASE("an SPZ file dequantises on the GPU and turns right-up-back into the PLY's axes",
          "[scene][io][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);

    SECTION("version 3: smallest-three rotation, degree-1 harmonics") {
        // Two points; the second is transparent (alpha byte 0) and dropped.
        std::vector<uint8_t> s;
        appendInt24(s, 4096); appendInt24(s, 8192); appendInt24(s, -2048);   // (1, 2, -0.5) at 12 bits
        appendInt24(s, 0); appendInt24(s, 0); appendInt24(s, 0);
        s.push_back(191); s.push_back(0);                                  // alphas
        s.insert(s.end(), {255, 128, 0, 128, 128, 128});                   // colours
        s.insert(s.end(), {160, 176, 144, 160, 160, 160});                 // log scales 0, 1, -1
        // Largest component w (index 3), y = +1/sqrt2, x = z = 0.
        const uint32_t comp = (3u << 30) | (511u << 10);
        for (int k = 0; k < 2; ++k) {
            for (int b = 0; b < 4; ++b) {
                s.push_back(static_cast<uint8_t>((comp >> (8 * b)) & 0xFF));
            }
        }
        // Harmonics, rgb per basis: basis 0 (y) +0.5, basis 1 (z) -0.5, basis 2 (x) mixed.
        const std::vector<uint8_t> sh{192, 192, 192, 64, 64, 64, 192, 64, 128};
        s.insert(s.end(), sh.begin(), sh.end());
        s.insert(s.end(), sh.begin(), sh.end());
        const fs::path path = scratchFile("decode-v3.spz");
        writeSpz(path, 3, 2, 1, 12, s);

        auto raw = io::readSplats(path);
        if (!raw) {
            FAIL(raw.error().toString());
        }
        auto splats = loader->upload(*raw, 3);
        REQUIRE(splats);
        REQUIRE(splats->count == 1);
        CHECK(splats->degree() == 1);
        const auto v = inspect(*gpu, *splats, 0);
        // Right-up-back (1, 2, -0.5) is right-down-front (1, -2, 0.5).
        CHECK(v[0] == 1.0F);
        CHECK(v[1] == -2.0F);
        CHECK(v[2] == 0.5F);
        CHECK(v[3] == Approx(191.0 / 255.0).margin(1e-6));
        CHECK(v[4] == Approx(1.0).epsilon(2e-3));
        CHECK(v[5] == Approx(std::exp(1.0)).epsilon(2e-3));
        CHECK(v[6] == Approx(std::exp(-1.0)).epsilon(2e-3));
        // A quarter turn about y becomes one about -y: y and w of opposite
        // sign (the packing may negate the whole quaternion).
        CHECK(std::abs(v[8]) == Approx(0.70710678).margin(3e-3));
        CHECK(std::abs(v[10]) == Approx(0.70710678).margin(3e-3));
        CHECK(v[8] * v[10] < 0.0F);
        CHECK(std::abs(v[7]) < 3e-3F);
        CHECK(std::abs(v[9]) < 3e-3F);
        // Colour through SPZ's DC scale (0.15), then 0.5 + SH0 * dc.
        const auto base = [](double byte) { return 0.5 + 0.28209479 * ((byte / 255.0 - 0.5) / 0.15); };
        CHECK(v[11] == Approx(base(255)).margin(1e-3));
        CHECK(v[12] == Approx(base(128)).margin(1e-3));
        CHECK(v[13] == Approx(base(0)).margin(1e-3));
        // Basis 0 is odd in y and basis 1 in z: both change sign.
        CHECK(v[14] == Approx(-0.5).margin(1e-3));
        CHECK(v[15] == Approx(-0.5).margin(1e-3));
        CHECK(v[17] == Approx(0.5).margin(1e-3));
        CHECK(v[19] == Approx(0.5).margin(1e-3));
    }

    SECTION("version 2: first-three rotation, no harmonics") {
        std::vector<uint8_t> s;
        appendInt24(s, -256); appendInt24(s, 256); appendInt24(s, 512);   // (-1, 1, 2) at 8 bits
        s.push_back(255);
        s.insert(s.end(), {128, 128, 128});
        s.insert(s.end(), {160, 160, 160});
        s.insert(s.end(), {128, 128, 255});   // x, y ~ 0, z = 1: a half turn about z
        const fs::path path = scratchFile("decode-v2.spz");
        writeSpz(path, 2, 1, 0, 8, s);
        auto raw = io::readSplats(path);
        if (!raw) {
            FAIL(raw.error().toString());
        }
        auto splats = loader->upload(*raw, 3);
        REQUIRE(splats);
        REQUIRE(splats->count == 1);
        CHECK(splats->degree() == 0);
        const auto v = inspect(*gpu, *splats, 0);
        CHECK(v[0] == -1.0F);
        CHECK(v[1] == -1.0F);
        CHECK(v[2] == -2.0F);
        CHECK(v[3] == Approx(1.0).margin(1e-6));
        CHECK(std::abs(v[9]) == Approx(1.0).margin(1e-2));
    }
}
#endif

TEST_CASE("point files of every kind load, with 8-bit colour linearised on the GPU",
          "[scene][io][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);

    const fs::path ascii = scratchFile("points.ply");
    {
        std::ofstream out(ascii);
        out << "ply\nformat ascii 1.0\nelement vertex 3\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
               "0 0 0 255 0 0\n1 2 3 0 255 0\n-1 -2 -3 0 0 255\n";
    }
    auto rawAscii = io::readPoints(ascii);
    REQUIRE(rawAscii);
    CHECK(rawAscii->count == 3);
    CHECK(rawAscii->colourKind == 1);
    auto cloud = loader->upload(*rawAscii);
    REQUIRE(cloud);
    CHECK(cloud->count == 3);
    CHECK(cloud->bounds.min[1] == -2.0F);
    CHECK(cloud->bounds.max[2] == 3.0F);

    const fs::path colmap = scratchFile("points3D.txt");
    {
        std::ofstream out(colmap);
        out << "# 3D point list\n1 0.5 0.5 0.5 128 128 128 0.1 1 2\n2 1.5 0.5 0.5 10 20 30 0.2\n";
    }
    auto rawColmap = io::readPoints(colmap);
    REQUIRE(rawColmap);
    CHECK(rawColmap->count == 2);
    CHECK(rawColmap->records[0] == 0.5F);   // the id was skipped

    // detail keeps a fraction, the same subset every time.
    std::vector<float> many;
    const fs::path xyz = scratchFile("many.xyz");
    {
        std::ofstream out(xyz);
        for (int i = 0; i < 20000; ++i) {
            out << i << " 0 0\n";
        }
    }
    auto rawMany = io::readPoints(xyz);
    REQUIRE(rawMany);
    CHECK(rawMany->count == 20000);
    auto half = loader->upload(*rawMany, 0.5F);
    REQUIRE(half);
    CHECK(half->count > 9000);
    CHECK(half->count < 11000);
    auto again = loader->upload(*rawMany, 0.5F);
    REQUIRE(again);
    CHECK(again->count == half->count);
}

TEST_CASE("a trained cloud from openFXplayer's examples loads", "[scene][io][gpu][data]") {
    const fs::path train = fs::path(std::getenv("HOME") != nullptr ? std::getenv("HOME") : "") /
                           "openFXplayer/examples/media/train_7k.ply";
    std::error_code ignored;
    if (!fs::exists(train, ignored)) {
        SKIP("no " + train.string());
    }
    LRT_REQUIRE_GPU(gpu);
    auto raw = io::readSplats(train);
    REQUIRE(raw);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(*raw);
    REQUIRE(splats);
    CHECK(splats->count > splats->declared / 2);
    CHECK(splats->degree() == 3);
    CHECK(splats->bounds.min[0] < splats->bounds.max[0]);
    std::printf("%s: %u of %u splats, bounds x %.2f..%.2f\n", train.c_str(), splats->count,
                splats->declared, static_cast<double>(splats->bounds.min[0]),
                static_cast<double>(splats->bounds.max[0]));
}
