// Copyright (c) 2026 lucabRTrender contributors.
//
// Volumes (M9): a .vdb written from boxes of known extent, read and laid
// out as NanoVDB, uploaded, and checked by kernels -- the active voxels
// counted through the leaf masks against the count the boxes make, the
// bounds against the boxes' extent, each leaf's maximum recomputed voxel by
// voxel against the prepare kernel's, the majorant against the largest
// value filled. Nothing about the values is read on the host.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <filesystem>

#include "lrt/io/Vdb.h"
#include "lrt/world/VolumeSet.h"

using namespace lrt;
namespace fs = std::filesystem;

TEST_CASE("a VDB grid laid out as NanoVDB holds the voxels it was filled with, and its bounds and maxima are the "
          "kernels'",
          "[volume][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path dir = fs::temp_directory_path() / "lrt-tests" / "volume";
    fs::create_directories(dir);
    const fs::path path = dir / "boxes.vdb";
    // Two leaf-aligned boxes: 32^3 at 0.5, and 8^3 apart at 2.0.
    const io::VdbBox boxes[] = {{{0, 0, 0}, {32, 32, 32}, 0.5F}, {{40, 0, 0}, {48, 8, 8}, 2.0F}};
    if (auto written = io::writeVdbBoxes(path, "density", 0.1, boxes); !written) FAIL(written.error().toString());
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
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, std::span<const io::NanoGrid>(&*grid, 1)));
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
            cursor["leafMax"].setBinding(set->leafMax().rhi());
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
