// Copyright (c) 2026 lucabRTrender contributors.
//
// Volumes (M9): a .vdb written from boxes of known extent, read and laid
// out as NanoVDB, uploaded, and checked by kernels -- the active voxels
// counted through the leaf masks against the count the boxes make, the
// bounds against the boxes' extent, each leaf's maximum recomputed voxel by
// voxel against the prepare kernel's, the majorant against the largest
// value filled. Then the medium on it: Beer-Lambert through the dense box
// by ratio and delta tracking within the standard error the kernel
// measures, no density over its majorant, and the leaf walk visiting the
// leaves a voxel walk passes through. Nothing about the values is read on
// the host.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "lrt/io/Vdb.h"
#include "lrt/world/VolumeSet.h"

using namespace lrt;
namespace fs = std::filesystem;

namespace {

/// The fixture: 32^3 voxels at 0.5 from the origin, 8^3 at 2.0 apart, voxel 0.1.
fs::path boxesFixture() {
    const fs::path dir = fs::temp_directory_path() / "lrt-tests" / "volume";
    fs::create_directories(dir);
    const fs::path path = dir / "boxes.vdb";
    const io::VdbBox boxes[] = {{{0, 0, 0}, {32, 32, 32}, 0.5F}, {{40, 0, 0}, {48, 8, 8}, 2.0F}};
    if (auto written = io::writeVdbBoxes(path, "density", 0.1, boxes); !written) FAIL(written.error().toString());
    return path;
}

}   // namespace

TEST_CASE("a VDB grid laid out as NanoVDB holds the voxels it was filled with, and its bounds and maxima are the "
          "kernels'",
          "[volume][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path path = boxesFixture();
    auto names = io::vdbGridNames(path);
    if (!names) FAIL(names.error().toString());
    REQUIRE(*names == std::vector<std::string>{"density"});
    auto grid = io::readVdbGrid(path, "density");
    if (!grid) FAIL(grid.error().toString());
    std::printf("  %s: %zu words, %u leaves, voxel %.3f\n", grid->name.c_str(), grid->words.size(), grid->leafCount,
                grid->voxelSize);
    CHECK(grid->leafCount == 64 + 1);
    CHECK(grid->voxelSize == 0.1);

    auto set = world::VolumeSet::create(*gpu->library);
    if (!set) FAIL(set.error().toString());
    world::VolumeInput input;
    input.grid = &*grid;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
        REQUIRE(batch.submit(true));
    }
    auto countKernel = gpu::ComputeKernel::create(*gpu->library, "lrt/test/volume_check", "volumeCountActive");
    auto maxKernel = gpu::ComputeKernel::create(*gpu->library, "lrt/test/volume_check", "volumeLeafMaxCheck");
    if (!countKernel) FAIL(countKernel.error().toString());
    if (!maxKernel) FAIL(maxKernel.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "volume.counts");
    const world::VolumeGridPlacement& p = set->placement(0);
    {
        gpu::CommandBatch batch(*gpu->device);
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(set->words().rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["check"]["gridWord"].setData(p.gridWord);
            cursor["check"]["leafCount"].setData(p.leafCount);
            cursor["check"]["leafBase"].setData(p.leafBase);
        };
        countKernel->dispatch(batch, {p.leafCount, 1, 1}, bind);
        maxKernel->dispatch(batch, {p.leafCount, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    uint32_t c[3] = {};
    int32_t bounds[6] = {};
    uint32_t majorantBits = 0;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(set->bounds().read(*gpu->device, 0, sizeof(bounds), bounds));
    REQUIRE(set->majorant().read(*gpu->device, 0, 4, &majorantBits));
    float majorant = 0.0F;
    std::memcpy(&majorant, &majorantBits, 4);
    std::printf("  %u active voxels (32^3 + 8^3 = %u); bounds [%d %d %d, %d %d %d); %u of %u leaf maxima off the "
                "voxel-by-voxel recount; majorant %.3f\n",
                c[0], 32u * 32u * 32u + 8u * 8u * 8u, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
                c[1], c[2], static_cast<double>(majorant));
    CHECK(c[0] == 32u * 32u * 32u + 8u * 8u * 8u);
    CHECK(bounds[0] == 0);
    CHECK(bounds[1] == 0);
    CHECK(bounds[2] == 0);
    CHECK(bounds[3] == 48);
    CHECK(bounds[4] == 32);
    CHECK(bounds[5] == 32);
    CHECK(c[2] == 65);
    CHECK(c[1] == 0);
    CHECK(majorant == 2.0F);
}

TEST_CASE("a medium on the grid is Beer-Lambert's within its own standard error, never over its majorant, and its "
          "leaf walk visits the leaves a voxel walk does",
          "[volume][gpu][medium]") {
    LRT_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    auto grid = io::readVdbGrid(boxesFixture(), "density");
    if (!grid) FAIL(grid.error().toString());
    auto set = world::VolumeSet::create(*gpu->library);
    if (!set) FAIL(set.error().toString());
    // The dense box is 32 voxels of 0.1: 3.2 world units at density 0.5,
    // so sigma L = 1.6 along an axis through it; a scale of 1.
    world::VolumeInput input;
    input.grid = &*grid;
    input.densityScale = 1.0F;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
        REQUIRE(batch.submit(true));
    }
    auto rays = gpu::ComputeKernel::create(*gpu->library, "lrt/test/medium_check", "mediumRays");
    auto reduce = gpu::ComputeKernel::create(*gpu->library, "lrt/test/medium_check", "mediumReduce");
    auto majorant = gpu::ComputeKernel::create(*gpu->library, "lrt/test/medium_check", "mediumMajorant");
    auto steps = gpu::ComputeKernel::create(*gpu->library, "lrt/test/medium_check", "mediumSteps");
    if (!rays) FAIL(rays.error().toString());
    if (!reduce) FAIL(reduce.error().toString());
    if (!majorant) FAIL(majorant.error().toString());
    if (!steps) FAIL(steps.error().toString());
    constexpr uint32_t kRays = 16384;
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 6, "medium.counts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 4, "medium.sums");
    gpu::Buffer perRay = test::uintBuffer(*gpu->device, uint64_t{kRays} * 2, "medium.perRay");
    struct Ray {
        float origin[3];
        float direction[3];
        float tMin, tMax;
        float expectedT;
        const char* name;
        uint32_t leaves;   // the leaf walk's expected count; 0: judge against the voxel walk
    };
    // The chord a ray cuts through the dense box [0, 3.2)^3, for the
    // expectation of an oblique ray (the fixture's geometry, by construction).
    const auto chord = [](const float* o, const float* d) {
        float t0 = -1e9F, t1 = 1e9F;
        for (int a = 0; a < 3; ++a) {
            float ta = (0.0F - o[a]) / d[a];
            float tb = (3.2F - o[a]) / d[a];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
        }
        return std::max(t1 - t0, 0.0F);
    };
    const float oblique[3] = {-0.5F, -0.3F, -0.2F};
    const float obliqueDir[3] = {0.68599F, 0.51449F, 0.51449F};   // (0.8, 0.6, 0.6) normalised
    // Through the middle of the dense box along x: 4 leaves, 6 leaf cells
    // with the empty ones past it; obliquely across it, judged against the
    // voxel walk; and along x through the small box from beyond the dense
    // one: 1 leaf.
    const Ray cases[] = {
        {{-1.0F, 1.6F, 1.6F}, {1.0F, 0.0F, 0.0F}, 0.0F, 10.0F, std::exp(-1.6F), "x through the dense box", 4},
        {{oblique[0], oblique[1], oblique[2]}, {obliqueDir[0], obliqueDir[1], obliqueDir[2]}, 0.0F, 20.0F,
         std::exp(-0.5F * chord(oblique, obliqueDir)), "obliquely across the dense box", 0},
        {{3.5F, 0.4F, 0.4F}, {1.0F, 0.0F, 0.0F}, 0.0F, 10.0F, std::exp(-2.0F * 0.8F), "x through the small box", 1},
    };
    for (const Ray& ray : cases) {
        const auto bind = [&](rhi::ShaderCursor c) {
            c["medium"]["rays"].setData(kRays);
            c["medium"]["volume"].setData(uint32_t{0});
            c["medium"]["expectedT"].setData(ray.expectedT);
            c["medium"]["origin"].setData(ray.origin, sizeof(ray.origin));
            c["medium"]["tMin"].setData(ray.tMin);
            c["medium"]["direction"].setData(ray.direction, sizeof(ray.direction));
            c["medium"]["tMax"].setData(ray.tMax);
            const float boxMin[3] = {0.0F, 0.0F, 0.0F};
            const float boxMax[3] = {4.8F, 3.2F, 3.2F};
            c["medium"]["boxMin"].setData(boxMin, sizeof(boxMin));
            c["medium"]["boxMax"].setData(boxMax, sizeof(boxMax));
            c["words"].setBinding(set->words().rhi());
            c["counts"].setBinding(counts.rhi());
            c["sums"].setBinding(sums.rhi());
            c["perRay"].setBinding(perRay.rhi());
        };
        const uint32_t zeros[6] = {};
        REQUIRE(counts.write(*gpu->device, 0, sizeof(zeros), zeros));
        {
            gpu::CommandBatch batch(*gpu->device);
            rays->dispatch(batch, {kRays, 1, 1}, bind);
            reduce->dispatch(batch, {1, 1, 1}, bind);
            majorant->dispatch(batch, {kRays, 1, 1}, bind);
            steps->dispatch(batch, {1, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        uint32_t c[6] = {};
        float s[4] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
        const double n = kRays;
        const double meanT = s[0] / n;
        const double varT = std::max(s[1] / n - meanT * meanT, 0.0);
        const double zT = (meanT - ray.expectedT) / std::max(std::sqrt(varT / n), 1e-9);
        const double meanS = s[2] / n;
        const double varS = std::max(s[3] / n - meanS * meanS, 0.0);
        const double zS = (meanS - (1.0 - ray.expectedT)) / std::max(std::sqrt(varS / n), 1e-9);
        std::printf("  %s: T %.4f by ratio tracking (expected %.4f, z %.2f); scattered %.4f by delta tracking "
                    "(expected %.4f, z %.2f); %u of %u probes over their majorant; leaf walk %u steps, %u with a "
                    "leaf, voxel walk %u distinct leaves\n",
                    ray.name, meanT, static_cast<double>(ray.expectedT), zT, meanS, 1.0 - ray.expectedT, zS, c[1],
                    c[5], c[2], c[3], c[4]);
        CHECK(std::abs(zT) < 3.5);
        CHECK(std::abs(zS) < 3.5);
        CHECK(c[1] == 0);
        CHECK(c[2] == c[4]);   // every leaf cell the ray crosses, once
        if (ray.leaves != 0) {
            CHECK(c[3] == ray.leaves);
        }
    }
}
