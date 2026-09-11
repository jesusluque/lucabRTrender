// Copyright (c) 2026 lucabRTrender contributors.
//
// Meshes through the visibility pass and their first shading, checked against
// what the geometry analytically is and against itself arranged another way.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <vector>

#include "lrt/geom/Mesh.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/GpuScene.h"

using namespace lrt;

namespace {

struct Renderer {
    geom::MeshBuilder           builder;
    world::GpuScene             scene;
    technique::VisibilityRaster raster;
    technique::HeadlightShading shading;
    technique::VisibilityTargets visibility;
};

std::unique_ptr<Renderer> renderer(test::Gpu& gpu) {
    auto builder = geom::MeshBuilder::create(*gpu.library);
    auto scene = world::GpuScene::create(*gpu.library);
    auto raster = technique::VisibilityRaster::create(*gpu.library);
    auto shading = technique::HeadlightShading::create(*gpu.library);
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!shading) FAIL(shading.error().toString());
    return std::unique_ptr<Renderer>(
        new Renderer{std::move(*builder), std::move(*scene), std::move(*raster), std::move(*shading), {}});
}

/// A unit square in z = 0, two triangles, facing +z.
std::shared_ptr<const geom::GpuMesh> square(Renderer& r, float half) {
    static std::vector<float> points;
    points = {-half, -half, 0, half, -half, 0, half, half, 0, -half, half, 0};
    static const std::vector<int32_t> counts{4};
    static const std::vector<int32_t> indices{0, 1, 2, 3};
    geom::MeshInput in;
    in.source = "square";
    in.points = {std::as_bytes(std::span<const float>(points)), false};
    in.faceVertexCounts = counts;
    in.faceVertexIndices = indices;
    in.smoothNormals = false;
    auto mesh = r.builder.build(in);
    if (!mesh) FAIL(mesh.error().toString());
    return std::make_shared<const geom::GpuMesh>(std::move(*mesh));
}

render::RenderTargets draw(test::Gpu& gpu, Renderer& r, std::span<const world::MeshInstance> instances,
                           const render::Projection& projection, uint32_t w, uint32_t h) {
    REQUIRE(r.scene.update(instances, projection));
    render::RenderTargets out;
    gpu::CommandBatch batch(*gpu.device);
    REQUIRE(r.raster.render(batch, r.scene, projection, w, h, r.visibility));
    REQUIRE(r.shading.shade(batch, r.scene, r.visibility, projection, out));
    REQUIRE(batch.submit(true));
    return out;
}

}   // namespace

TEST_CASE("a square drawn through visibility and shading is where, how deep and how lit it analytically is",
          "[technique][visibility]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    auto r = renderer(*gpu);
    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    world::MeshInstance instance;
    instance.mesh = square(*r, 1.0F);
    instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    instance.displayColor = {0.8F, 0.4F, 0.2F};
    const render::RenderTargets out = draw(*gpu, *r, std::span<const world::MeshInstance>(&instance, 1), projection, w, h);

    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/visibility_check", "planeCheck");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "counts");
    gpu::BufferDesc one;
    one.bytes = 4;
    one.elementBytes = 4;
    auto worst = gpu::Buffer::create(*gpu->device, one);
    REQUIRE(worst);
    gpu::CommandBatch batch(*gpu->device);
    check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst->rhi());
        technique::setCamera(cursor["camera"], projection, w, h);
        cursor["plane"]["z"].setData(5.0F);
        cursor["plane"]["half"].setData(1.0F);
        cursor["plane"]["colourR"].setData(0.8F);
        cursor["plane"]["colourG"].setData(0.4F);
        cursor["plane"]["colourB"].setData(0.2F);
    });
    REQUIRE(batch.submit(true));
    uint32_t c[3] = {0, 0, 0};
    float depthError = 0.0F;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(depthError), &depthError));
    std::printf("  square: %u pixels covered, %u coverage and %u colour mismatches, depth off by %.2e\n", c[2], c[0],
                c[1], static_cast<double>(depthError));
    CHECK(c[2] > 1000);
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
    CHECK(depthError < 1e-4F);
}

TEST_CASE("instances of one mesh draw as the same meshes built apart", "[technique][visibility][instancing]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    auto r = renderer(*gpu);
    const uint32_t w = 200;
    const uint32_t h = 150;
    render::Camera camera = render::Camera::lookingAt({1.0, 2.0, 9.0}, {0.0, 0.0, 0.0});
    const render::Projection projection = render::projectionFor(camera, w, h);
    const auto shared = square(*r, 0.7F);
    std::vector<world::MeshInstance> instanced;
    std::vector<world::MeshInstance> apart;
    for (int k = 0; k < 6; ++k) {
        world::MeshInstance i;
        i.objectToWorld = aofx::xform::translation({-3.0 + k * 1.2, 0.3 * k, -0.5 * k}) *
                          aofx::xform::rotationY(20.0 * k) * aofx::xform::rotationX(-10.0 * k);
        i.displayColor = {0.2F + 0.1F * k, 0.5F, 0.9F - 0.1F * k};
        i.mesh = shared;
        instanced.push_back(i);
        i.mesh = square(*r, 0.7F);   // the same data, a mesh of its own
        apart.push_back(i);
    }
    const render::RenderTargets a = draw(*gpu, *r, instanced, projection, w, h);
    const render::RenderTargets b = draw(*gpu, *r, apart, projection, w, h);
    CHECK(r->scene.meshCount() == 6);
    auto colour = render::compareHdr(*gpu->library, a.colour, b.colour, w, h);
    REQUIRE(colour);
    auto depth = render::countDifferent(*gpu->library, a.depth, b.depth, w * h);   // float bits compared as words
    REQUIRE(depth);
    std::printf("  six instances against six meshes: colour max relative %.2e, %llu depths differ\n",
                colour->maxRelative, static_cast<unsigned long long>(*depth));
    CHECK(colour->maxRelative == 0.0);
    CHECK(*depth == 0);
}
