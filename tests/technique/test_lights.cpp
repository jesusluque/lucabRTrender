// Copyright (c) 2026 lucabRTrender contributors.
//
// Lights against what they analytically put on a Lambert plane: a sphere, a
// disk and a rectangle, each sampled by the shading kernel and compared with
// the closed-form irradiance (lambert_irradiance.slang), which shares no code
// with the renderer.
#include "../gpu/GpuTest.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "lrt/geom/Mesh.h"
#include "lrt/light/LightTable.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"
#include "lrt/technique/MaterialShading.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/RayTracingScene.h"

using namespace lrt;

namespace {

std::shared_ptr<const geom::GpuMesh> lambertSquare(geom::MeshBuilder& builder, float half) {
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

TEST_CASE("a Lambert plane under a sphere, a disk and a rectangle is lit as the closed form says",
          "[technique][lights]") {
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
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!shading) FAIL(shading.error().toString());
    if (!table) FAIL(table.error().toString());

    // Lambert: Oren-Nayar at roughness 0 is albedo / pi.
    const float albedo = 0.8F;
    auto lambert = (*compiler)->compileXml(
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
        "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\">\n"
        "    <input name=\"color\" type=\"color3\" value=\"0.8, 0.8, 0.8\" />\n"
        "    <input name=\"roughness\" type=\"float\" value=\"0\" />\n"
        "  </oren_nayar_diffuse_bsdf>\n"
        "  <surface name=\"shader\" type=\"surfaceshader\"><input name=\"bsdf\" type=\"BSDF\" nodename=\"d\" /></surface>\n"
        "  <surfacematerial name=\"material\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
        "  </surfacematerial>\n</materialx>\n");
    if (!lambert) FAIL(lambert.error().toString());
    REQUIRE(programs->setModules(std::span<const material::CompiledMaterial>(&*lambert, 1)));
    REQUIRE(shading->setPrograms(*programs));
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *lambert, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu->device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);

    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    world::MeshInstance instance;
    instance.mesh = lambertSquare(*builder, 1.0F);
    instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    instance.material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection));
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());

    struct Case {
        const char*      name;
        light::LightKind kind;
        uint32_t         oracleKind;   // 0 sphere, 1 disk, 2 rect, 3 distant
        float            sizeX;
        float            sizeY;
    };
    // Each light sits between the camera and the plane, facing it down its
    // own -Z, with no occluder: what arrives is what the light emits.
    const std::array<Case, 6> cases{Case{"sphere", light::LightKind::Sphere, 0, 0.4F, 0.0F},
                                    Case{"disk", light::LightKind::Disk, 1, 0.6F, 0.0F},
                                    Case{"rect", light::LightKind::Rect, 2, 1.2F, 0.8F},
                                    Case{"sun", light::LightKind::Distant, 3, 0.0F, 0.0F},
                                    Case{"sun (0.2 rad)", light::LightKind::Distant, 3, 0.2F, 0.0F},
                                    Case{"dome", light::LightKind::Dome, 5, 0.0F, 0.0F}};
    for (const Case& c : cases) {
        light::Light lamp;
        lamp.kind = c.kind;
        lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
        lamp.colour[0] = lamp.colour[1] = lamp.colour[2] = 1.0F;
        lamp.intensity = 1.0F;
        lamp.radius = c.sizeX;
        lamp.width = c.sizeX;
        lamp.height = c.sizeY;
        lamp.angle = c.kind == light::LightKind::Distant ? c.sizeX : 0.0F;
        lamp.normalize = false;
        lamp.shadow = false;   // nothing to shadow against, and no structure bound
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

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
            frame.lights = &*table;
            frame.samples = 4096;
            REQUIRE(shading->shade(batch, visibility, projection, frame, out));
            REQUIRE(batch.submit(true));
        }
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "worst");
        const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(out.colour.rhi());
                cursor["depth"].setBinding(out.depth.rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                technique::setCamera(cursor["camera"], projection, w, h);
                rhi::ShaderCursor p = cursor["plane"];
                p["kind"].setData(c.oracleKind);
                p["vertices"].setData(uint32_t{512});
                p["row0"].setData(toWorld.data(), sizeof(float) * 4);
                p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
                p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
                const float centre[4] = {0.0F, 0.0F, -3.0F, 0.0F};
                const float axisX[4] = {1.0F, 0.0F, 0.0F, c.sizeX};
                // A distant light is a direction: the oracle reads it from axisY.
                const std::array<float, 4> axisY = c.kind == light::LightKind::Distant
                                                       ? std::array<float, 4>{0.0F, 0.0F, 1.0F, c.sizeY}
                                                       : std::array<float, 4>{0.0F, 1.0F, 0.0F, c.sizeY};
                const float normal[4] = {0.0F, 0.0F, 1.0F, albedo};
                const float radiance[4] = {1.0F, 1.0F, 1.0F, 0.02F};
                p["centre"].setData(centre, sizeof(centre));
                p["axisX"].setData(axisX, sizeof(axisX));
                p["axisY"].setData(axisY.data(), sizeof(float) * 4);
                p["normal"].setData(normal, sizeof(normal));
                p["radiance"].setData(radiance, sizeof(radiance));
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c2[2] = {0, 0};
        float relative = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c2), c2));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(relative), &relative));
        std::printf("  %-6s light: %u pixels, %u beyond 2%%, worst %.4f\n", c.name, c2[0], c2[1], double(relative));
        CHECK(c2[0] > 8000);
        CHECK(c2[1] == 0);
    }

    // A point light behind a square occluder: the umbra is exactly what the
    // occluder projects, and the light outside it is untouched.
    {
        auto accel = world::RayTracingScene::create(*gpu->library);
        if (!accel) FAIL(accel.error().toString());
        std::array<world::MeshInstance, 2> both;
        both[0] = instance;
        both[1].mesh = lambertSquare(*builder, 0.25F);
        both[1].objectToWorld = aofx::xform::translation({0.0, 0.0, -4.0});
        both[1].material = 1;
        REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), both.size()), projection));
        REQUIRE(accel->build(*scene));
        light::Light lamp;
        lamp.kind = light::LightKind::Sphere;
        lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
        lamp.colour[0] = lamp.colour[1] = lamp.colour[2] = 1.0F;
        lamp.intensity = 1.0F;
        lamp.radius = 0.0F;   // a point: a hard shadow with an exact edge
        lamp.shadow = true;
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
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
            frame.lights = &*table;
            frame.shadows = accel->topLevel();
            frame.samples = 1;
            REQUIRE(shading->shade(batch, visibility, projection, frame, out));
            REQUIRE(batch.submit(true));
        }
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "worst");
        const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(out.colour.rhi());
                cursor["depth"].setBinding(out.depth.rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                technique::setCamera(cursor["camera"], projection, w, h);
                rhi::ShaderCursor p = cursor["plane"];
                p["kind"].setData(uint32_t{4});   // a point light
                p["vertices"].setData(uint32_t{512});
                p["row0"].setData(toWorld.data(), sizeof(float) * 4);
                p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
                p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
                const float centre[4] = {0.0F, 0.0F, -3.0F, 0.0F};
                const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.0F};
                const float axisY[4] = {0.0F, 1.0F, 0.0F, 0.0F};
                const float normal[4] = {0.0F, 0.0F, 1.0F, albedo};
                const float radiance[4] = {1.0F, 1.0F, 1.0F, 0.02F};
                const float occluder[4] = {0.0F, 0.0F, -4.0F, 0.25F};
                const float axes[4] = {4.5F, 0.0F, 0.0F, 0.0F};   // ignore the occluder's own pixels
                p["centre"].setData(centre, sizeof(centre));
                p["axisX"].setData(axisX, sizeof(axisX));
                p["axisY"].setData(axisY, sizeof(axisY));
                p["normal"].setData(normal, sizeof(normal));
                p["radiance"].setData(radiance, sizeof(radiance));
                p["occluder"].setData(occluder, sizeof(occluder));
                p["occluderAxes"].setData(axes, sizeof(axes));
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c2[2] = {0, 0};
        float relative = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c2), c2));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(relative), &relative));
        std::printf("  shadow: %u plane pixels, %u away from the closed form or its umbra, worst %.4f\n", c2[0],
                    c2[1], double(relative));
        CHECK(c2[0] > 5000);
        CHECK(c2[1] == 0);
    }
}

TEST_CASE("each light's samples follow the density it reports", "[technique][lights][chi2]") {
    LRT_REQUIRE_GPU(gpu);
    auto table = light::LightTable::create(*gpu->device);
    if (!table) FAIL(table.error().toString());
    const auto kernel = [&](const char* entry) {
        auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_check", entry);
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    gpu::ComputeKernel draw = kernel("lightDraw");
    gpu::ComputeKernel count = kernel("lightCount");
    gpu::ComputeKernel expect = kernel("lightExpect");
    gpu::ComputeKernel statistic = kernel("lightStatistic");
    gpu::ComputeKernel consistent = kernel("lightConsistent");

    // Fine enough that a light of small support is resolved by its bins and
    // not smeared across one: the coarse grid a lobe needs is not enough here.
    const uint32_t thetaBins = 128;
    const uint32_t phiBins = 256;
    const uint32_t bins = thetaBins * phiBins;
    const uint32_t samples = 1u << 20;
    gpu::BufferDesc binDesc;
    binDesc.bytes = uint64_t{samples} * 4;
    binDesc.elementBytes = 4;
    binDesc.label = "light.bins";
    auto binBuffer = gpu::Buffer::create(*gpu->device, binDesc);
    REQUIRE(binBuffer);
    const std::vector<float> zeros(size_t{bins} + 1, 0.0F);
    auto observed = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "light.observed");
    auto expected = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "light.expected");
    const std::vector<float> resultZeros(8, 0.0F);
    auto result = gpu::Buffer::fromSpan<float>(*gpu->device, resultZeros, "light.result");
    gpu::Buffer mismatches = test::uintBuffer(*gpu->device, 1, "light.mismatches");
    gpu::Buffer worstPdf = test::uintBuffer(*gpu->device, 1, "light.worstPdf");
    REQUIRE(observed);
    REQUIRE(expected);
    REQUIRE(result);

    struct Case {
        const char*      name;
        light::LightKind kind;
        float            sizeX;
        float            sizeY;
    };
    const std::array<Case, 5> cases{Case{"sphere", light::LightKind::Sphere, 0.4F, 0.0F},
                                    Case{"disk", light::LightKind::Disk, 0.6F, 0.0F},
                                    Case{"rect", light::LightKind::Rect, 1.2F, 0.8F},
                                    Case{"sun (0.2 rad)", light::LightKind::Distant, 0.2F, 0.0F},
                                    Case{"dome", light::LightKind::Dome, 0.0F, 0.0F}};
    for (const Case& c : cases) {
        light::Light lamp;
        lamp.kind = c.kind;
        lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
        lamp.radius = c.sizeX;
        lamp.width = c.sizeX;
        lamp.height = c.sizeY;
        lamp.angle = c.kind == light::LightKind::Distant ? c.sizeX : 0.0F;
        lamp.shadow = false;
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
        const auto bind = [&](rhi::ShaderCursor cursor) {
            // Only `lights` here: LightTable::bind would also set lightCount,
            // which this program does not declare.
            cursor["lights"].setBinding(table->records().rhi());
            cursor["bins"].setBinding(binBuffer->rhi());
            cursor["observed"].setBinding(observed->rhi());
            cursor["expected"].setBinding(expected->rhi());
            cursor["result"].setBinding(result->rhi());
            cursor["mismatches"].setBinding(mismatches.rhi());
            cursor["worst"].setBinding(worstPdf.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["thetaBins"].setData(thetaBins);
            p["phiBins"].setData(phiBins);
            p["samples"].setData(samples);
            const float point[4] = {0.0F, 0.0F, -5.0F, 0.0F};
            const float normal[4] = {0.0F, 0.0F, 1.0F, 0.0F};
            p["point"].setData(point, sizeof(point));
            p["normal"].setData(normal, sizeof(normal));
        };
        {
            gpu::CommandBatch batch(*gpu->device);
            consistent.dispatch(batch, {1, 1, 1}, bind);
            draw.dispatch(batch, {samples, 1, 1}, bind);
            count.dispatch(batch, {1, 1, 1}, bind);
            expect.dispatch(batch, {1, 1, 1}, bind);
            statistic.dispatch(batch, {1, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        float r[8] = {};
        REQUIRE(result->read(*gpu->device, 0, sizeof(r), r));
        const float integral = r[0];
        const float chi2 = r[4];
        const float dof = r[5];
        const float drawn = r[6];
        // Pearson's statistic has mean dof and variance 2 dof.
        const float z = (chi2 - dof) / std::sqrt(2.0F * dof);
        uint32_t differ = 0;
        float worstRelative = 0.0F;
        REQUIRE(mismatches.read(*gpu->device, 0, sizeof(differ), &differ));
        REQUIRE(worstPdf.read(*gpu->device, 0, sizeof(worstRelative), &worstRelative));
        std::printf("  %-13s: chi2 %.1f on %.0f dof (z %.2f), pdf integral %.4f against %.4f drawn; %u samples "
                    "disagree with lightPdf (worst %.2e)\n",
                    c.name, double(chi2), double(dof), double(z), double(integral), double(drawn), differ,
                    double(worstRelative));
        CHECK(differ == 0);
        CHECK(std::abs(z) < 4.0F);
        // The pdf integrates to the fraction of samples the light drew.
        CHECK(std::abs(integral - drawn) < 0.02F);
    }
}
