// Copyright (c) 2026 lucabRTrender contributors.
//
// Skinning on the device against closed forms a check kernel evaluates on
// its own: one joint of weight 1 is a matrix chain; two joints turning
// about one axis at equal weight blend, under dual quaternions, to the mean
// angle exactly (linear blending would pull the point inward); blend
// shapes add their offsets by their sub-shape weights; constant influences
// bind every point the same way.
#include "../gpu/GpuTest.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "lrt/geom/Skinner.h"

using namespace lrt;

namespace {

/// GfMatrix4f's layout: row-major with the translation in the last row,
/// vectors on the left. A rotation about z by `degrees` then a translation.
std::array<float, 16> usdMatrix(float degrees, float tx, float ty, float tz) {
    const float a = degrees * 3.14159265F / 180.0F;
    const float c = std::cos(a);
    const float s = std::sin(a);
    return {c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1};
}

/// The same as rows for M * v: the transpose.
std::array<float, 16> rowsOf(const std::array<float, 16>& usd) {
    std::array<float, 16> t{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3 + 1; ++c) {
            t[static_cast<size_t>(r * 4 + c)] = usd[static_cast<size_t>(c * 4 + r)];
        }
    }
    return t;
}

std::vector<float> gridPoints(int n) {
    std::vector<float> points;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            points.insert(points.end(), {x * 0.3F - 1.0F, y * 0.25F - 0.7F, 0.1F * (x - y)});
        }
    }
    return points;
}

struct Check {
    uint32_t checked = 0;
    uint32_t beyond = 0;
    float    worst = 0.0F;
};

Check checkSkinned(test::Gpu& gpu, const std::vector<float>& rest, const gpu::Buffer& skinned, uint32_t mode,
                   float angle, const std::array<float, 16>& geomBind, const std::array<float, 16>& joint,
                   const std::array<float, 16>& skelToWorld, const std::array<float, 16>& worldToPrim,
                   const gpu::Buffer* offsets = nullptr, const gpu::Buffer* ranges = nullptr,
                   const gpu::Buffer* weights = nullptr, uint32_t ranged = 0) {
    auto made = gpu::ComputeKernel::create(*gpu.library, "lrt/test/skin_check", "skinCheck");
    if (!made) FAIL(made.error().toString());
    auto restWords = gpu::Buffer::fromSpan<float>(*gpu.device, rest, "check.rest");
    REQUIRE(restWords);
    gpu::Buffer none = test::uintBuffer(*gpu.device, 4, "check.none");
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 2, "check.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu.device, 1, "check.worst");
    const uint32_t count = static_cast<uint32_t>(rest.size() / 3);
    gpu::CommandBatch batch(*gpu.device);
    made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["restWords"].setBinding(restWords->rhi());
        cursor["skinned"].setBinding(skinned.rhi());
        cursor["offsets"].setBinding(offsets != nullptr ? offsets->rhi() : none.rhi());
        cursor["ranges"].setBinding(ranges != nullptr ? ranges->rhi() : none.rhi());
        cursor["weights"].setBinding(weights != nullptr ? weights->rhi() : none.rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst.rhi());
        rhi::ShaderCursor c = cursor["check"];
        c["count"].setData(count);
        c["mode"].setData(mode);
        c["angle"].setData(angle);
        c["tolerance"].setData(1.0e-5F);
        c["ranged"].setData(ranged);
        const std::array<float, 16> g = rowsOf(geomBind);
        const std::array<float, 16> j = rowsOf(joint);
        const std::array<float, 16> s = rowsOf(skelToWorld);
        const std::array<float, 16> w = rowsOf(worldToPrim);
        static constexpr const char* kRows[4][4] = {{"geomBind0", "geomBind1", "geomBind2", "geomBind3"},
                                                    {"joint0", "joint1", "joint2", "joint3"},
                                                    {"skelToWorld0", "skelToWorld1", "skelToWorld2", "skelToWorld3"},
                                                    {"worldToPrim0", "worldToPrim1", "worldToPrim2", "worldToPrim3"}};
        const std::array<float, 16>* matrices[4] = {&g, &j, &s, &w};
        for (size_t m = 0; m < 4; ++m) {
            for (size_t r = 0; r < 4; ++r) {
                c[kRows[m][r]].setData(matrices[m]->data() + r * 4, sizeof(float) * 4);
            }
        }
    });
    REQUIRE(batch.submit(true));
    Check out;
    uint32_t n[2] = {0, 0};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(n), n));
    REQUIRE(worst.read(*gpu.device, 0, sizeof(out.worst), &out.worst));
    out.checked = n[0];
    out.beyond = n[1];
    return out;
}

/// A unit dual quaternion for a rotation about z by `degrees` and a
/// translation, as GfDualQuatf lays it out: real (x, y, z, w), dual.
std::array<float, 8> dualQuat(float degrees, float tx, float ty, float tz) {
    const float half = degrees * 3.14159265F / 360.0F;
    const float rx = 0.0F, ry = 0.0F, rz = std::sin(half), rw = std::cos(half);
    // dual = 0.5 * t * real, with t a pure quaternion.
    const float dx = 0.5F * (tx * rw + ty * rz - tz * ry);
    const float dy = 0.5F * (ty * rw + tz * rx - tx * rz);
    const float dz = 0.5F * (tz * rw + tx * ry - ty * rx);
    const float dw = -0.5F * (tx * rx + ty * ry + tz * rz);
    return {rx, ry, rz, rw, dx, dy, dz, dw};
}

}   // namespace

TEST_CASE("linear blend skinning under one joint is the matrix chain, and constant influences bind every point",
          "[geom][skinning]") {
    LRT_REQUIRE_GPU(gpu);
    auto skinner = geom::Skinner::create(*gpu->library);
    if (!skinner) FAIL(skinner.error().toString());
    const std::vector<float> rest = gridPoints(8);
    const uint32_t count = 64;
    const std::array<float, 16> geomBind = usdMatrix(15.0F, 0.2F, -0.1F, 0.3F);
    const std::array<float, 16> joint = usdMatrix(40.0F, 1.0F, 0.5F, -0.25F);
    const std::array<float, 16> still = usdMatrix(0.0F, 0.0F, 0.0F, 0.0F);
    const std::array<float, 16> skelToWorld = usdMatrix(-30.0F, 0.0F, 2.0F, 0.0F);
    const std::array<float, 16> worldToPrim = usdMatrix(5.0F, -0.4F, 0.0F, 1.0F);
    // Two joints: the first still, the second the one every point is bound to.
    std::vector<float> xforms(still.begin(), still.end());
    xforms.insert(xforms.end(), joint.begin(), joint.end());
    geom::SkinningInput in;
    in.points = count;
    in.restPoints = std::as_bytes(std::span<const float>(rest));
    in.numInfluencesPerPoint = 2;
    in.skinningXforms = std::as_bytes(std::span<const float>(xforms));
    in.geomBindXform = geomBind;
    in.skelLocalToWorld = skelToWorld;
    in.primWorldToLocal = worldToPrim;
    // Per point: (joint 1, weight 1) and (joint 0, weight 0).
    std::vector<float> influences;
    for (uint32_t i = 0; i < count; ++i) {
        influences.insert(influences.end(), {1.0F, 1.0F, 0.0F, 0.0F});
    }
    in.influences = std::as_bytes(std::span<const float>(influences));
    auto skinned = skinner->skin(in);
    if (!skinned) FAIL(skinned.error().toString());
    const Check perPoint = checkSkinned(*gpu, rest, *skinned, 0, 0.0F, geomBind, joint, skelToWorld, worldToPrim);
    std::printf("  one joint, per-point influences: %u of %u points beyond 1e-5 (worst %.2e)\n", perPoint.beyond,
                perPoint.checked, static_cast<double>(perPoint.worst));
    CHECK(perPoint.checked == count);
    CHECK(perPoint.beyond == 0);

    // The same binding once, for every point.
    const std::vector<float> constant{1.0F, 1.0F, 0.0F, 0.0F};
    in.influences = std::as_bytes(std::span<const float>(constant));
    in.constantInfluences = true;
    auto shared = skinner->skin(in);
    if (!shared) FAIL(shared.error().toString());
    const Check once = checkSkinned(*gpu, rest, *shared, 0, 0.0F, geomBind, joint, skelToWorld, worldToPrim);
    std::printf("  one joint, constant influences: %u of %u beyond (worst %.2e)\n", once.beyond, once.checked,
                static_cast<double>(once.worst));
    CHECK(once.beyond == 0);
}

TEST_CASE("dual quaternion skinning blends two turns about one axis to the mean angle, which linear blending does not",
          "[geom][skinning][dqs]") {
    LRT_REQUIRE_GPU(gpu);
    auto skinner = geom::Skinner::create(*gpu->library);
    if (!skinner) FAIL(skinner.error().toString());
    const std::vector<float> rest = gridPoints(8);
    const uint32_t count = 64;
    const std::array<float, 16> identity = usdMatrix(0.0F, 0.0F, 0.0F, 0.0F);
    const float a = 20.0F;
    const float b = 80.0F;
    const std::array<float, 8> qa = dualQuat(a, 0.0F, 0.0F, 0.0F);
    const std::array<float, 8> qb = dualQuat(b, 0.0F, 0.0F, 0.0F);
    std::vector<float> quats(qa.begin(), qa.end());
    quats.insert(quats.end(), qb.begin(), qb.end());
    std::vector<float> influences;
    for (uint32_t i = 0; i < count; ++i) {
        influences.insert(influences.end(), {0.0F, 0.5F, 1.0F, 0.5F});
    }
    geom::SkinningInput in;
    in.points = count;
    in.restPoints = std::as_bytes(std::span<const float>(rest));
    in.numInfluencesPerPoint = 2;
    in.influences = std::as_bytes(std::span<const float>(influences));
    in.method = geom::SkinningMethod::DualQuaternion;
    in.skinningDualQuats = std::as_bytes(std::span<const float>(quats));
    auto dqs = skinner->skin(in);
    if (!dqs) FAIL(dqs.error().toString());
    const float mean = 0.5F * (a + b) * 3.14159265F / 180.0F;
    const Check blended = checkSkinned(*gpu, rest, *dqs, 1, mean, identity, identity, identity, identity);
    std::printf("  dual quaternions, %g and %g degrees at equal weight: %u of %u beyond 1e-5 of the %g degree turn "
                "(worst %.2e)\n",
                a, b, blended.beyond, blended.checked, 0.5F * (a + b), static_cast<double>(blended.worst));
    CHECK(blended.beyond == 0);

    // Linear blending of the same two turns pulls the points inward.
    const std::array<float, 16> ma = usdMatrix(a, 0.0F, 0.0F, 0.0F);
    const std::array<float, 16> mb = usdMatrix(b, 0.0F, 0.0F, 0.0F);
    std::vector<float> xforms(ma.begin(), ma.end());
    xforms.insert(xforms.end(), mb.begin(), mb.end());
    in.method = geom::SkinningMethod::LinearBlend;
    in.skinningXforms = std::as_bytes(std::span<const float>(xforms));
    auto lbs = skinner->skin(in);
    if (!lbs) FAIL(lbs.error().toString());
    const Check linear = checkSkinned(*gpu, rest, *lbs, 1, mean, identity, identity, identity, identity);
    std::printf("  linear blending of the same: %u of %u beyond (worst %.2e)\n", linear.beyond, linear.checked,
                static_cast<double>(linear.worst));
    CHECK(linear.beyond > count / 2);
}

TEST_CASE("blend shapes add each sub-shape's offsets by its weight before the joints", "[geom][skinning][blendshapes]") {
    LRT_REQUIRE_GPU(gpu);
    auto skinner = geom::Skinner::create(*gpu->library);
    if (!skinner) FAIL(skinner.error().toString());
    const std::vector<float> rest = gridPoints(8);
    const uint32_t count = 64;
    // Three sub-shapes (a shape and two inbetweens, say) touching the first
    // 40 points: offsets listed per point, each with its sub-shape.
    std::vector<float> offsets;
    std::vector<uint32_t> ranges;
    for (uint32_t i = 0; i < 40; ++i) {
        const uint32_t start = static_cast<uint32_t>(offsets.size() / 4);
        offsets.insert(offsets.end(), {0.1F * i, 0.0F, 0.02F, 0.0F});
        offsets.insert(offsets.end(), {0.0F, -0.05F * i, 0.0F, 1.0F});
        if (i % 2 == 0) {
            offsets.insert(offsets.end(), {0.3F, 0.3F, -0.1F * i, 2.0F});
        }
        ranges.insert(ranges.end(), {start, static_cast<uint32_t>(offsets.size() / 4)});
    }
    const std::vector<float> weights{0.25F, 0.5F, -0.75F};
    geom::SkinningInput in;
    in.points = count;
    in.restPoints = std::as_bytes(std::span<const float>(rest));
    in.blendShapeOffsets = std::as_bytes(std::span<const float>(offsets));
    in.blendShapeOffsetRanges = std::as_bytes(std::span<const uint32_t>(ranges));
    in.blendShapeWeights = std::as_bytes(std::span<const float>(weights));
    auto shaped = skinner->skin(in);
    if (!shaped) FAIL(shaped.error().toString());
    auto offsetsBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, offsets, "shapes.offsets");
    auto rangesBuffer = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, ranges, "shapes.ranges");
    auto weightsBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, weights, "shapes.weights");
    REQUIRE(offsetsBuffer);
    REQUIRE(rangesBuffer);
    REQUIRE(weightsBuffer);
    const std::array<float, 16> identity = usdMatrix(0.0F, 0.0F, 0.0F, 0.0F);
    const Check c = checkSkinned(*gpu, rest, *shaped, 2, 0.0F, identity, identity, identity, identity, &*offsetsBuffer,
                                 &*rangesBuffer, &*weightsBuffer, static_cast<uint32_t>(ranges.size() / 2));
    std::printf("  three sub-shapes over 40 of 64 points: %u beyond 1e-5 (worst %.2e)\n", c.beyond,
                static_cast<double>(c.worst));
    CHECK(c.checked == count);
    CHECK(c.beyond == 0);
}
