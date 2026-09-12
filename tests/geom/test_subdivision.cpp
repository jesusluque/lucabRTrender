// Copyright (c) 2026 lucabRTrender contributors.
//
// Subdivision on the device against what must hold exactly: the counts a
// level has (Euler's bookkeeping, integer and exact), the limit a vertex
// reaches (the closed form from the coarse mesh, the same at every level --
// converging, not drifting), a crease holding its plane, and a square's
// face-varying grid staying a grid.
#include "../gpu/GpuTest.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "lrt/geom/Subdivision.h"

using namespace lrt;

namespace {

struct Cube {
    std::vector<float>   points{-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1};
    std::vector<int32_t> counts{4, 4, 4, 4, 4, 4};
    // Each face wound outward.
    std::vector<int32_t> indices{0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3};
};

struct Octahedron {
    std::vector<float>   points{1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1};
    std::vector<int32_t> counts{3, 3, 3, 3, 3, 3, 3, 3};
    std::vector<int32_t> indices{0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
};

struct LimitOutcome {
    uint32_t off = 0;
    float    gap = 0.0F;
    std::array<float, 3> expected{};
};

LimitOutcome limitCheck(test::Gpu& gpu, const std::vector<float>& coarse, const geom::Refined& refined,
                        uint32_t vertex, const std::vector<uint32_t>& neighbours,
                        const std::vector<uint32_t>& faceCorners, uint32_t scheme) {
    auto made = gpu::ComputeKernel::create(*gpu.library, "lrt/test/subdiv_check", "subdivLimitCheck");
    if (!made) FAIL(made.error().toString());
    std::vector<float> coarse4;
    for (size_t i = 0; i < coarse.size() / 3; ++i) {
        coarse4.insert(coarse4.end(), {coarse[i * 3], coarse[i * 3 + 1], coarse[i * 3 + 2], 1.0F});
    }
    auto coarseBuffer = gpu::Buffer::fromSpan<float>(*gpu.device, coarse4, "check.coarse");
    auto neighbourBuffer = gpu::Buffer::fromSpan<uint32_t>(*gpu.device, neighbours, "check.neighbours");
    std::vector<uint32_t> cornersPadded = faceCorners.empty() ? std::vector<uint32_t>{0} : faceCorners;
    auto cornerBuffer = gpu::Buffer::fromSpan<uint32_t>(*gpu.device, cornersPadded, "check.corners");
    REQUIRE(coarseBuffer);
    REQUIRE(neighbourBuffer);
    REQUIRE(cornerBuffer);
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 2, "check.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu.device, 4, "check.worst");
    gpu::Buffer none = test::uintBuffer(*gpu.device, 4, "check.none");
    gpu::CommandBatch batch(*gpu.device);
    made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["coarse"].setBinding(coarseBuffer->rhi());
        cursor["neighbours"].setBinding(neighbourBuffer->rhi());
        cursor["centroids"].setBinding(none.rhi());
        cursor["faceCorners"].setBinding(cornerBuffer->rhi());
        cursor["refined"].setBinding(refined.positions.rhi());
        cursor["refinedCounts"].setBinding(none.rhi());
        cursor["refinedStarts"].setBinding(none.rhi());
        cursor["fvar"].setBinding(none.rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst.rhi());
        rhi::ShaderCursor c = cursor["check"];
        c["count"].setData(refined.points);
        c["vertex"].setData(vertex);
        c["valence"].setData(static_cast<uint32_t>(neighbours.size()));
        c["scheme"].setData(scheme);
        c["tolerance"].setData(1.0e-5F);
        c["planeY"].setData(0.0F);
    });
    REQUIRE(batch.submit(true));
    LimitOutcome out;
    float w[4] = {};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(out.off), &out.off));
    REQUIRE(worst.read(*gpu.device, 0, sizeof(w), w));
    out.gap = w[0];
    out.expected = {w[1], w[2], w[3]};
    return out;
}

}   // namespace

TEST_CASE("Catmull-Clark on a cube: every level's counts are Euler's, and the corner's limit is the closed form at "
          "every level",
          "[geom][subdivision][catmullClark]") {
    LRT_REQUIRE_GPU(gpu);
    auto subdivider = geom::Subdivider::create(*gpu->library);
    if (!subdivider) FAIL(subdivider.error().toString());
    const Cube cube;
    // Vertex 0 (-1,-1,-1): neighbours 1, 3, 4; its three faces.
    const std::vector<uint32_t> neighbours{1, 3, 4};
    const std::vector<uint32_t> faceCorners{0, 3, 2, 1, 0, 1, 5, 4, 0, 4, 7, 3};
    uint32_t vertices = 8;
    uint32_t edges = 12;
    uint32_t faces = 6;
    for (uint32_t levels = 1; levels <= 3; ++levels) {
        geom::SubdivisionInput in;
        in.source = "cube";
        in.points = {std::as_bytes(std::span<const float>(cube.points)), false};
        in.faceVertexCounts = cube.counts;
        in.faceVertexIndices = cube.indices;
        in.scheme = geom::SubdivisionScheme::CatmullClark;
        in.levels = levels;
        auto refined = subdivider->refine(in);
        if (!refined) FAIL(refined.error().toString());
        // Euler's bookkeeping, a level at a time: V' = V + E + F, F' = sum of
        // the faces' corners, E' = 2E + sum of corners.
        const uint32_t cornersSum = faces * 4;
        vertices = vertices + edges + faces;
        edges = 2 * edges + cornersSum;
        faces = cornersSum;
        const LimitOutcome limit = limitCheck(*gpu, cube.points, *refined, 0, neighbours, faceCorners, 1);
        std::printf("  level %u: %u points (%u expected), %u faces (%u), %u edges (%u); corner limit %.6f off the "
                    "closed form (%.4f, %.4f, %.4f)\n",
                    levels, refined->points, vertices, static_cast<uint32_t>(refined->faceVertexCounts.size()), faces,
                    refined->edges, edges, static_cast<double>(limit.gap), static_cast<double>(limit.expected[0]),
                    static_cast<double>(limit.expected[1]), static_cast<double>(limit.expected[2]));
        CHECK(refined->points == vertices);
        CHECK(refined->faceVertexCounts.size() == faces);
        CHECK(refined->edges == edges);
        CHECK(limit.off == 0);
        CHECK(std::abs(limit.expected[0] + 0.5F) < 1e-6F);
    }
}

TEST_CASE("Loop on an octahedron: Euler's counts and the vertex limit at every level", "[geom][subdivision][loop]") {
    LRT_REQUIRE_GPU(gpu);
    auto subdivider = geom::Subdivider::create(*gpu->library);
    if (!subdivider) FAIL(subdivider.error().toString());
    const Octahedron octa;
    const std::vector<uint32_t> neighbours{2, 3, 4, 5};   // vertex 0's ring
    uint32_t vertices = 6;
    uint32_t edges = 12;
    uint32_t faces = 8;
    for (uint32_t levels = 1; levels <= 3; ++levels) {
        geom::SubdivisionInput in;
        in.source = "octahedron";
        in.points = {std::as_bytes(std::span<const float>(octa.points)), false};
        in.faceVertexCounts = octa.counts;
        in.faceVertexIndices = octa.indices;
        in.scheme = geom::SubdivisionScheme::Loop;
        in.levels = levels;
        auto refined = subdivider->refine(in);
        if (!refined) FAIL(refined.error().toString());
        vertices = vertices + edges;
        edges = 2 * edges + 3 * faces;
        faces = 4 * faces;
        const LimitOutcome limit = limitCheck(*gpu, octa.points, *refined, 0, neighbours, {}, 2);
        std::printf("  level %u: %u points (%u), %u faces (%u), %u edges (%u); limit %.6f off the closed form "
                    "(%.4f, %.4f, %.4f)\n",
                    levels, refined->points, vertices, static_cast<uint32_t>(refined->faceVertexCounts.size()), faces,
                    refined->edges, edges, static_cast<double>(limit.gap), static_cast<double>(limit.expected[0]),
                    static_cast<double>(limit.expected[1]), static_cast<double>(limit.expected[2]));
        CHECK(refined->points == vertices);
        CHECK(refined->faceVertexCounts.size() == faces);
        CHECK(refined->edges == edges);
        CHECK(limit.off == 0);
    }
}

TEST_CASE("a sharp crease around a cube's top holds the top's plane, and without it the top sinks",
          "[geom][subdivision][creases]") {
    LRT_REQUIRE_GPU(gpu);
    auto subdivider = geom::Subdivider::create(*gpu->library);
    if (!subdivider) FAIL(subdivider.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/subdiv_check", "subdivPlaneCheck");
    if (!made) FAIL(made.error().toString());
    const Cube cube;
    // The top face's loop: vertices 2, 3, 7, 6 (y = 1), as one crease of five indices.
    const std::vector<int32_t> creaseIndices{2, 3, 7, 6, 2};
    const std::vector<int32_t> creaseLengths{5};
    const std::vector<float> sharp{10.0F};
    const auto run = [&](bool creased, uint32_t levels) {
        geom::SubdivisionInput in;
        in.source = "cube";
        in.points = {std::as_bytes(std::span<const float>(cube.points)), false};
        in.faceVertexCounts = cube.counts;
        in.faceVertexIndices = cube.indices;
        in.scheme = geom::SubdivisionScheme::CatmullClark;
        in.levels = levels;
        if (creased) {
            in.creaseIndices = creaseIndices;
            in.creaseLengths = creaseLengths;
            in.creaseSharpnesses = sharp;
        }
        auto refined = subdivider->refine(in);
        if (!refined) FAIL(refined.error().toString());
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "plane.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 4, "plane.worst");
        gpu::Buffer none = test::uintBuffer(*gpu->device, 4, "plane.none");
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["coarse"].setBinding(none.rhi());
            cursor["neighbours"].setBinding(none.rhi());
            cursor["centroids"].setBinding(none.rhi());
            cursor["faceCorners"].setBinding(none.rhi());
            cursor["refined"].setBinding(refined->positions.rhi());
            cursor["refinedCounts"].setBinding(none.rhi());
            cursor["refinedStarts"].setBinding(none.rhi());
            cursor["fvar"].setBinding(none.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            rhi::ShaderCursor c = cursor["check"];
            c["count"].setData(refined->points);
            c["vertex"].setData(uint32_t{0});
            c["valence"].setData(uint32_t{0});
            c["scheme"].setData(uint32_t{1});
            c["tolerance"].setData(1.0e-6F);
            c["planeY"].setData(1.0F);
        });
        REQUIRE(batch.submit(true));
        uint32_t c[2] = {0, 0};
        float highest = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(highest), &highest));
        std::printf("  %s, %u levels: %u points above y = 1, %u on it, highest %.6f (%u points)\n",
                    creased ? "creased" : "smooth", levels, c[0], c[1], static_cast<double>(highest),
                    refined->points);
        return std::array<uint32_t, 2>{c[0], c[1]};
    };
    for (uint32_t levels = 1; levels <= 2; ++levels) {
        const auto creased = run(true, levels);
        CHECK(creased[0] == 0);
        CHECK(creased[1] >= 9);   // the loop's corners, its edge points and the face point, at least
        const auto smooth = run(false, levels);
        CHECK(smooth[1] == 0);
    }
}

TEST_CASE("a cube's face-varying square stays a grid under subdivision, its seams as boundaries",
          "[geom][subdivision][faceVarying]") {
    LRT_REQUIRE_GPU(gpu);
    auto subdivider = geom::Subdivider::create(*gpu->library);
    if (!subdivider) FAIL(subdivider.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/subdiv_check", "subdivRectCheck");
    if (!made) FAIL(made.error().toString());
    const Cube cube;
    // Every face its own unit square: four values of its own a face (a first
    // version gave the six faces the same four, which made one face-varying
    // "vertex" of each square corner across the cube and no seam at all).
    std::vector<float> st;
    std::vector<int32_t> stIndices;
    for (int f = 0; f < 6; ++f) {
        st.insert(st.end(), {0, 0, 1, 0, 1, 1, 0, 1});
        stIndices.insert(stIndices.end(), {f * 4, f * 4 + 1, f * 4 + 2, f * 4 + 3});
    }
    geom::PrimvarInput primvar;
    primvar.name = "st";
    primvar.interpolation = geom::Interpolation::FaceVarying;
    primvar.components = 2;
    primvar.values = {std::as_bytes(std::span<const float>(st)), false};
    primvar.indices = stIndices;
    geom::SubdivisionInput in;
    in.source = "cube";
    in.points = {std::as_bytes(std::span<const float>(cube.points)), false};
    in.faceVertexCounts = cube.counts;
    in.faceVertexIndices = cube.indices;
    in.scheme = geom::SubdivisionScheme::CatmullClark;
    in.levels = 2;
    in.primvars = std::span<const geom::PrimvarInput>(&primvar, 1);
    auto refined = subdivider->refine(in);
    if (!refined) FAIL(refined.error().toString());
    const geom::RefinedPrimvar* refinedSt = nullptr;
    for (const geom::RefinedPrimvar& r : refined->primvars) {
        if (r.name == "st") refinedSt = &r;
    }
    REQUIRE(refinedSt != nullptr);
    REQUIRE(refinedSt->count == refined->faceVertexIndices.size());
    std::vector<uint32_t> counts(refined->faceVertexCounts.begin(), refined->faceVertexCounts.end());
    std::vector<uint32_t> starts;
    uint32_t at = 0;
    for (const uint32_t n : counts) {
        starts.push_back(at);
        at += n;
    }
    auto countBuffer = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, counts, "rect.counts");
    auto startBuffer = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, starts, "rect.starts");
    REQUIRE(countBuffer);
    REQUIRE(startBuffer);
    gpu::Buffer out = test::uintBuffer(*gpu->device, 2, "rect.bad");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 4, "rect.worst");
    gpu::Buffer none = test::uintBuffer(*gpu->device, 4, "rect.none");
    gpu::CommandBatch batch(*gpu->device);
    made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["coarse"].setBinding(none.rhi());
        cursor["neighbours"].setBinding(none.rhi());
        cursor["centroids"].setBinding(none.rhi());
        cursor["faceCorners"].setBinding(none.rhi());
        cursor["refined"].setBinding(refined->positions.rhi());
        cursor["refinedCounts"].setBinding(countBuffer->rhi());
        cursor["refinedStarts"].setBinding(startBuffer->rhi());
        cursor["fvar"].setBinding(refinedSt->values.rhi());
        cursor["counts"].setBinding(out.rhi());
        cursor["worst"].setBinding(worst.rhi());
        rhi::ShaderCursor c = cursor["check"];
        c["count"].setData(static_cast<uint32_t>(counts.size()));
        c["vertex"].setData(uint32_t{0});
        c["valence"].setData(uint32_t{0});
        c["scheme"].setData(uint32_t{1});
        c["tolerance"].setData(1.0e-5F);
        c["planeY"].setData(0.0F);
    });
    REQUIRE(batch.submit(true));
    uint32_t bad = 0;
    REQUIRE(out.read(*gpu->device, 0, sizeof(bad), &bad));
    std::printf("  %zu refined faces, %u whose st corners are not an axis-aligned rectangle\n", counts.size(), bad);
    CHECK(counts.size() == 96);
    CHECK(bad == 0);
}
