// Copyright (c) 2026 lucabRTrender contributors.
//
// Meshes onto the device: triangulation in Hydra's order and smooth normals,
// each checked by a kernel that reaches the same answer another way.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "lrt/geom/Mesh.h"

using namespace lrt;

namespace {

struct Authored {
    std::vector<float>   points;   // xyz
    std::vector<int32_t> counts;
    std::vector<int32_t> indices;
    std::vector<int32_t> holes;
};

geom::MeshInput inputOf(const Authored& a, bool leftHanded = false) {
    geom::MeshInput in;
    in.source = "test mesh";
    in.points = {std::as_bytes(std::span<const float>(a.points)), false};
    in.faceVertexCounts = a.counts;
    in.faceVertexIndices = a.indices;
    in.holeIndices = a.holes;
    in.leftHanded = leftHanded;
    return in;
}

struct Checked {
    float    triangleArea = 0.0F;
    float    faceArea = 0.0F;
    uint32_t badCorners = 0;
    uint32_t facingUp = 0;
    uint32_t facingDown = 0;
    float    worstAngle = 0.0F;
};

Checked check(test::Gpu& gpu, const Authored& a, const geom::GpuMesh& mesh, const char* normalsCheck) {
    const auto kernel = [&](const char* entry) {
        auto made = gpu::ComputeKernel::create(*gpu.library, "lrt/test/mesh_check", entry);
        if (!made) FAIL(made.error().toString());
        return *made;
    };
    auto counts = gpu::Buffer::fromSpan<int32_t>(*gpu.device, a.counts, "counts");
    auto indices = gpu::Buffer::fromSpan<int32_t>(*gpu.device, a.indices, "indices");
    std::vector<int32_t> holes = a.holes.empty() ? std::vector<int32_t>{-1} : a.holes;
    auto holeBuffer = gpu::Buffer::fromSpan<int32_t>(*gpu.device, holes, "holes");
    REQUIRE(counts);
    REQUIRE(indices);
    REQUIRE(holeBuffer);
    gpu::BufferDesc floats;
    floats.bytes = 8;
    floats.elementBytes = 4;
    auto result = gpu::Buffer::create(*gpu.device, floats);
    REQUIRE(result);
    gpu::Buffer countsOut = test::uintBuffer(*gpu.device, 2, "counts.out");
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["faceVertexCounts"].setBinding(counts->rhi());
        cursor["faceVertexIndices"].setBinding(indices->rhi());
        cursor["holeIndices"].setBinding(holeBuffer->rhi());
        cursor["positions"].setBinding(mesh.positions.rhi());
        cursor["triangles"].setBinding(mesh.indices.rhi());
        cursor["triangleCorners"].setBinding(mesh.triangleCorners.rhi());
        cursor["triangleFaces"].setBinding(mesh.triangleFaces.rhi());
        const geom::GpuPrimvar* normals = mesh.primvar("normals");
        cursor["normals"].setBinding(normals != nullptr ? normals->values.rhi() : mesh.positions.rhi());
        cursor["result"].setBinding(result->rhi());
        cursor["counts"].setBinding(countsOut.rhi());
        cursor["params"]["faces"].setData(mesh.faces);
        cursor["params"]["indices"].setData(static_cast<uint32_t>(a.indices.size()));
        cursor["params"]["triangles"].setData(mesh.triangles);
        cursor["params"]["points"].setData(mesh.points);
        cursor["params"]["holes"].setData(static_cast<uint32_t>(a.holes.size()));
    };
    Checked out;
    const auto run = [&](const char* entry) {
        gpu::ComputeKernel k = kernel(entry);
        gpu::CommandBatch batch(*gpu.device);
        k.dispatch(batch, {1, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    };
    float floatsRead[2] = {0, 0};
    uint32_t uints[2] = {0, 0};
    run("areas");
    REQUIRE(result->read(*gpu.device, 0, 8, floatsRead));
    out.triangleArea = floatsRead[0];
    out.faceArea = floatsRead[1];
    run("corners");
    REQUIRE(countsOut.read(*gpu.device, 0, 4, uints));
    out.badCorners = uints[0];
    run("facing");
    REQUIRE(countsOut.read(*gpu.device, 0, 8, uints));
    out.facingUp = uints[0];
    out.facingDown = uints[1];
    if (normalsCheck != nullptr) {
        run(normalsCheck);
        REQUIRE(result->read(*gpu.device, 0, 4, floatsRead));
        out.worstAngle = floatsRead[0];
    }
    return out;
}

/// A planar mesh of every kind of face: a quad, a pentagon, a triangle, a
/// face of two vertices, a face of none, a hole, a hexagon.
Authored mixedFaces() {
    Authored a;
    const auto add = [&](float x, float y) {
        a.points.insert(a.points.end(), {x, y, 0.0F});
        return static_cast<int32_t>(a.points.size() / 3 - 1);
    };
    const auto polygon = [&](float cx, float cy, float r, int n) {
        a.counts.push_back(n);
        for (int k = 0; k < n; ++k) {
            const float angle = 6.2831853F * static_cast<float>(k) / static_cast<float>(n);
            a.indices.push_back(add(cx + r * std::cos(angle), cy + r * std::sin(angle)));
        }
    };
    polygon(0, 0, 1, 4);
    polygon(3, 0, 1, 5);
    polygon(6, 0, 1, 3);
    a.counts.push_back(2);
    a.indices.push_back(0);
    a.indices.push_back(1);
    a.counts.push_back(0);
    polygon(0, 3, 1, 4);
    a.holes.push_back(static_cast<int32_t>(a.counts.size() - 1));
    polygon(3, 3, 1.5F, 6);
    return a;
}

}   // namespace

TEST_CASE("polygons triangulate into their own area, corner by corner, facing their handedness", "[geom][mesh]") {
    LRT_REQUIRE_GPU(gpu);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    const Authored authored = mixedFaces();
    for (const bool leftHanded : {false, true}) {
        auto mesh = builder->build(inputOf(authored, leftHanded));
        if (!mesh) FAIL(mesh.error().toString());
        const Checked c = check(*gpu, authored, *mesh, nullptr);
        std::printf("  %s: %u triangles, area %.6f (faces %.6f), %u bad corners, %u up %u down\n",
                    leftHanded ? "left-handed" : "right-handed", mesh->triangles,
                    static_cast<double>(c.triangleArea), static_cast<double>(c.faceArea), c.badCorners, c.facingUp,
                    c.facingDown);
        // quad 2 + pentagon 3 + triangle 1 + hexagon 4; the two-vertex face,
        // the empty face and the hole make none.
        CHECK(mesh->triangles == 10);
        CHECK(mesh->faces == 7);
        CHECK(std::abs(c.triangleArea - c.faceArea) <= 1e-5F * c.faceArea);
        CHECK(c.badCorners == 0);
        CHECK(c.facingUp == (leftHanded ? 0u : 10u));
        CHECK(c.facingDown == (leftHanded ? 10u : 0u));
    }
}

TEST_CASE("smooth normals match every touching triangle's area-weighted normal", "[geom][mesh][normals]") {
    LRT_REQUIRE_GPU(gpu);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    // A bumpy height field of triangles.
    Authored a;
    const int n = 24;
    for (int y = 0; y <= n; ++y) {
        for (int x = 0; x <= n; ++x) {
            const float fx = static_cast<float>(x) / n;
            const float fy = static_cast<float>(y) / n;
            a.points.insert(a.points.end(), {fx, fy, 0.1F * std::sin(9.0F * fx) * std::cos(7.0F * fy)});
        }
    }
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int p = y * (n + 1) + x;
            a.counts.insert(a.counts.end(), {3, 3});
            a.indices.insert(a.indices.end(), {p, p + 1, p + n + 2, p, p + n + 2, p + n + 1});
        }
    }
    auto mesh = builder->build(inputOf(a));
    if (!mesh) FAIL(mesh.error().toString());
    REQUIRE(mesh->primvar("normals") != nullptr);
    const Checked c = check(*gpu, a, *mesh, "bruteNormals");
    std::printf("  height field: %u triangles, worst normal %.2e rad from the brute-force sum\n", mesh->triangles,
                static_cast<double>(c.worstAngle));
    CHECK(c.worstAngle < 1e-4F);
}

TEST_CASE("a quad sphere's smooth normals point away from its centre", "[geom][mesh][normals]") {
    LRT_REQUIRE_GPU(gpu);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    Authored a;
    const int rings = 48;
    const int segments = 96;
    // Poles, then rings of points.
    a.points.insert(a.points.end(), {0, 0, 1, 0, 0, -1});
    for (int r = 1; r < rings; ++r) {
        const float theta = 3.14159265F * static_cast<float>(r) / rings;
        for (int s = 0; s < segments; ++s) {
            const float phi = 6.2831853F * static_cast<float>(s) / segments;
            a.points.insert(a.points.end(), {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                                             std::cos(theta)});
        }
    }
    const auto at = [&](int r, int s) { return 2 + (r - 1) * segments + (s % segments); };
    for (int s = 0; s < segments; ++s) {
        a.counts.push_back(3);
        a.indices.insert(a.indices.end(), {0, at(1, s), at(1, s + 1)});
        a.counts.push_back(3);
        a.indices.insert(a.indices.end(), {1, at(rings - 1, s + 1), at(rings - 1, s)});
    }
    for (int r = 1; r < rings - 1; ++r) {
        for (int s = 0; s < segments; ++s) {
            a.counts.push_back(4);
            a.indices.insert(a.indices.end(), {at(r, s), at(r + 1, s), at(r + 1, s + 1), at(r, s + 1)});
        }
    }
    auto mesh = builder->build(inputOf(a));
    if (!mesh) FAIL(mesh.error().toString());
    const Checked c = check(*gpu, a, *mesh, "sphereNormals");
    std::printf("  sphere: %u triangles, worst normal %.2e rad from radial\n", mesh->triangles,
                static_cast<double>(c.worstAngle));
    CHECK(c.worstAngle < 1e-3F);
    CHECK(c.badCorners == 0);
}
