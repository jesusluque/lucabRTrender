// Copyright (c) 2026 lucabRTrender contributors.
//
// Meshes shaded by MaterialX materials through the visibility buffer: a
// diffuse material and the displayColor fallback lit by the headlight are
// where and how lit a square analytically is (visibility_check.slang), as
// HeadlightShading's squares are.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <vector>

#include "lrt/geom/Mesh.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"
#include "lrt/technique/MaterialShading.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/GpuScene.h"

using namespace lrt;

namespace {

std::shared_ptr<const geom::GpuMesh> square(geom::MeshBuilder& builder, float half) {
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
    auto mesh = builder.build(in);
    if (!mesh) FAIL(mesh.error().toString());
    return std::make_shared<const geom::GpuMesh>(std::move(*mesh));
}

}   // namespace

TEST_CASE("a square shaded by a MaterialX diffuse material, and by the displayColor fallback, is lit as analytically",
          "[technique][materials]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu->library);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    auto scene = world::GpuScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto shading = technique::MaterialShading::create(*gpu->library);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!shading) FAIL(shading.error().toString());
    auto diffuse = (*compiler)->compileXml(
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
        "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\">\n"
        "    <input name=\"color\" type=\"color3\" value=\"0.8, 0.4, 0.2\" />\n"
        "  </oren_nayar_diffuse_bsdf>\n"
        "  <surface name=\"shader\" type=\"surfaceshader\"><input name=\"bsdf\" type=\"BSDF\" nodename=\"d\" /></surface>\n"
        "  <surfacematerial name=\"material\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
        "  </surfacematerial>\n</materialx>\n");
    if (!diffuse) FAIL(diffuse.error().toString());
    REQUIRE(programs->setModules(std::span<const material::CompiledMaterial>(&*diffuse, 1)));
    REQUIRE(shading->setPrograms(*programs));
    // Rows: 0 the fallback, 1 the diffuse material (function 1, blob from 0).
    const std::vector<float> blob =
        material::MaterialCompiler::parameters(*diffuse, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu->device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/visibility_check", "planeCheck");
    if (!check) FAIL(check.error().toString());

    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    for (uint32_t row : {1u, 0u}) {
        world::MeshInstance instance;
        instance.mesh = square(*builder, 1.0F);
        instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
        instance.displayColor = {0.8F, 0.4F, 0.2F};
        instance.material = row;
        REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection));
        technique::VisibilityTargets visibility;
        render::RenderTargets out;
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
            technique::MaterialFrame frame;
            frame.programs = &*programs;
            frame.scene = &*scene;
            frame.records = &*records;
            frame.blob = &*blobBuffer;
            frame.textures = &**textures;
            REQUIRE(shading->shade(batch, visibility, projection, frame, out));
            REQUIRE(batch.submit(true));
        }
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "counts");
        gpu::BufferDesc one;
        one.bytes = 4;
        one.elementBytes = 4;
        auto worst = gpu::Buffer::create(*gpu->device, one);
        REQUIRE(worst);
        {
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
        }
        uint32_t c[3] = {0, 0, 0};
        float depthError = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst->read(*gpu->device, 0, sizeof(depthError), &depthError));
        std::printf("  %s: %u pixels covered, %u coverage and %u colour mismatches, depth off by %.2e\n",
                    row == 1 ? "oren_nayar material" : "displayColor fallback", c[2], c[0], c[1],
                    static_cast<double>(depthError));
        CHECK(c[2] > 1000);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
        CHECK(depthError < 1e-4F);
    }
}
