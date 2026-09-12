// Copyright (c) 2026 lucabRTrender contributors.
//
// Lights against what they analytically put on a Lambert plane: a sphere, a
// disk and a rectangle, each sampled by the shading kernel and compared with
// the closed-form irradiance (lambert_irradiance.slang), which shares no code
// with the renderer.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hio/types.h>

#include "lrt/geom/Mesh.h"
#include "lrt/material/TextureStore.h"
#include "lrt/light/LightTable.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/material/TextureStore.h"
#include "lrt/technique/MaterialShading.h"
#include "lrt/technique/PathTracer.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/RayTracingScene.h"

using namespace lrt;

namespace {

/// Where this file's generated images go, as the other suites keep theirs.
std::filesystem::path scratch(const std::string& name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lrt-tests" / "technique";
    std::filesystem::create_directories(dir);
    return dir / name;
}

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
            // Both ways through the same closed form: with one light the
            // choice is certain, so choosing must change nothing at all.
            frame.chooseLights = std::getenv("LRT_CHOOSE_LIGHTS") != nullptr;
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
        // The occluder is in category 1; a shadow link naming category 0
        // leaves it casting nothing, and the plane reads as if it were not
        // there at all.
        both[1].categories = uint64_t{1} << 1;
        REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), both.size()), projection));
        REQUIRE(accel->build(*scene));
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
    gpu::ComputeKernel roundTrip = kernel("lightRoundTrip");

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
    gpu::Buffer mapping = test::uintBuffer(*gpu->device, 1, "light.mapping");
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
            cursor["roundTrip"].setBinding(mapping.rhi());
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
            roundTrip.dispatch(batch, {1, 1, 1}, bind);
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
        uint32_t mapped = 0;
        REQUIRE(mapping.read(*gpu->device, 0, sizeof(mapped), &mapped));
        if (c.kind == light::LightKind::Dome) {
            // A direction is where the image says it is: 63 x 64 uv through a
            // direction and back. Only a dome maps one at all.
            std::printf("  %-13s: %u of 4032 uv do not survive a turn through a direction\n", c.name, mapped);
            CHECK(mapped == 0);
        }
        CHECK(differ == 0);
        CHECK(std::abs(z) < 4.0F);
        // The pdf integrates to the fraction of samples the light drew.
        CHECK(std::abs(integral - drawn) < 0.02F);
    }
}

TEST_CASE("a dome with a sun in it samples what its density describes",
          "[technique][lights][chi2]") {
    LRT_REQUIRE_GPU(gpu);
    // A sky that varies: dark everywhere but one bright patch. A flat image
    // is better served by the cosine and takes that path instead, so only an
    // image like this one exercises the warp at all.
    const std::filesystem::path png = scratch("dome_sun.png");
    // Square unless asked otherwise: a lat-long is 2:1, and whether that
    // asymmetry is what the descent trips on is exactly what this asks.
    const uint32_t w = std::getenv("LRT_DOME_WIDE") != nullptr ? 64u : 32u;
    const uint32_t h = 32;
    {
        std::filesystem::remove(png);
        std::vector<uint32_t> texels(size_t{w} * h, 0xFF101010u);   // ABGR, a dim sky
        for (uint32_t y = 10; y < 14; ++y) {
            for (uint32_t x = w / 2; x < w / 2 + 6 && x < w; ++x) {
                texels[y * w + x] = 0xFFFFFFFFu;   // the sun
            }
        }
        pxr::HioImageSharedPtr made = pxr::HioImage::OpenForWriting(png.string());
        REQUIRE(made);
        pxr::HioImage::StorageSpec spec;
        spec.width = static_cast<int>(w);
        spec.height = static_cast<int>(h);
        spec.depth = 1;
        spec.format = pxr::HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(made->Write(spec));
    }
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Dome;
    lamp.texture = png.string();
    lamp.textureId = (*textures)->request(png.string(), material::ColourSpace::Auto);
    lamp.sampler = (*textures)->sampler(material::Wrap::Repeat, material::Wrap::Clamp);
    lamp.shadow = false;
    REQUIRE((*textures)->commit());
    auto table = light::LightTable::create(*gpu->device);
    if (!table) FAIL(table.error().toString());
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

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
    gpu::ComputeKernel pyramidCheck = kernel("lightPyramid");

    const uint32_t thetaBins = 32;
    const uint32_t phiBins = 64;
    const uint32_t bins = thetaBins * phiBins;
    const uint32_t samples = 1u << 20;
    gpu::BufferDesc binDesc;
    binDesc.bytes = uint64_t{samples} * 4;
    binDesc.elementBytes = 4;
    binDesc.label = "dome.bins";
    auto binBuffer = gpu::Buffer::create(*gpu->device, binDesc);
    REQUIRE(binBuffer);
    const std::vector<float> zeros(size_t{bins} + 1, 0.0F);
    auto observed = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "dome.observed");
    auto expected = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "dome.expected");
    auto result = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(8, 0.0F), "dome.result");
    gpu::Buffer mismatches = test::uintBuffer(*gpu->device, 1, "dome.mismatches");
    gpu::Buffer worstPdf = test::uintBuffer(*gpu->device, 1, "dome.worstPdf");
    gpu::Buffer mapping = test::uintBuffer(*gpu->device, 1, "dome.mapping");
    auto pyramid = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(1, 0.0F), "dome.pyramid");
    REQUIRE(pyramid);
    REQUIRE(observed);
    REQUIRE(expected);
    REQUIRE(result);
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["lights"].setBinding(table->records().rhi());
        (*textures)->bind(cursor["gTextures"]);
        cursor["bins"].setBinding(binBuffer->rhi());
        cursor["observed"].setBinding(observed->rhi());
        cursor["expected"].setBinding(expected->rhi());
        cursor["result"].setBinding(result->rhi());
        cursor["mismatches"].setBinding(mismatches.rhi());
        cursor["worst"].setBinding(worstPdf.rhi());
        cursor["roundTrip"].setBinding(mapping.rhi());
        cursor["pyramid"].setBinding(pyramid->rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["thetaBins"].setData(thetaBins);
        p["phiBins"].setData(phiBins);
        p["samples"].setData(samples);
        const float point[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        const float normal[4] = {0.0F, 0.0F, 1.0F, 0.0F};
        p["point"].setData(point, sizeof(point));
        p["normal"].setData(normal, sizeof(normal));
    };
    {
        gpu::CommandBatch batch(*gpu->device);
        consistent.dispatch(batch, {1, 1, 1}, bind);
        pyramidCheck.dispatch(batch, {1, 1, 1}, bind);
        draw.dispatch(batch, {samples, 1, 1}, bind);
        count.dispatch(batch, {1, 1, 1}, bind);
        expect.dispatch(batch, {1, 1, 1}, bind);
        statistic.dispatch(batch, {1, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    float r[8] = {};
    uint32_t differ = 0;
    float worstRelative = 0.0F;
    REQUIRE(result->read(*gpu->device, 0, sizeof(r), r));
    REQUIRE(mismatches.read(*gpu->device, 0, sizeof(differ), &differ));
    REQUIRE(worstPdf.read(*gpu->device, 0, sizeof(worstRelative), &worstRelative));
    const float chi2 = r[4];
    const float dof = r[5];
    const float drawn = r[6];
    float gap = 0.0F;
    REQUIRE(pyramid->read(*gpu->device, 0, sizeof(gap), &gap));
    std::printf("  the chain telescopes to within %.4f (a parent against its four children)\n", double(gap));
    const float z = (chi2 - dof) / std::sqrt(2.0F * dof);
    // While the image warp is parked (lights.slang), this is the cosine path
    // with an image on it: the statistic still has to hold.
    std::printf("  dome with a sun: chi2 %.1f on %.0f dof (z %.2f), pdf integral %.4f against %.4f drawn; %u "
                "samples disagree (worst %.2e)\n",
                double(chi2), double(dof), double(z), double(r[0]), double(drawn), differ, double(worstRelative));
    CHECK(differ == 0);
    CHECK(std::abs(z) < 4.0F);
}

TEST_CASE("a light reaches the categories it is linked to, and no others", "[technique][lights][linking]") {
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

    const uint32_t w = 160;
    const uint32_t h = 120;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    // Two squares side by side, in different categories: the left one in the
    // category the light is linked to, the right one in another.
    std::array<world::MeshInstance, 2> both;
    for (size_t k = 0; k < 2; ++k) {
        both[k].mesh = lambertSquare(*builder, 0.9F);
        both[k].objectToWorld = aofx::xform::translation({k == 0 ? -1.0 : 1.0, 0.0, -5.0});
        both[k].material = 1;
        both[k].categories = uint64_t{1} << k;
    }
    REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), both.size()), projection));
    auto counters = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_check", "lightLinkCounters");
    if (!counters) FAIL(counters.error().toString());

    struct Case {
        const char* name;
        uint32_t    category;
        bool        rightLit;
    };
    const std::array<Case, 2> cases{Case{"linked to the left square's category", 0, false},
                                    Case{"no collection at all", light::kLightUnlinked, true}};
    for (const Case& c : cases) {
        light::Light lamp;
        lamp.kind = light::LightKind::Sphere;
        lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
        lamp.radius = 0.4F;
        lamp.shadow = false;
        lamp.lightCategory = c.category;
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
            frame.samples = 16;
            frame.chooseLights = std::getenv("LRT_CHOOSE_LIGHTS") != nullptr;
            REQUIRE(shading->shade(batch, visibility, projection, frame, out));
            REQUIRE(batch.submit(true));
        }
        gpu::Buffer halves = test::uintBuffer(*gpu->device, 4, "halves");
        {
            gpu::CommandBatch batch(*gpu->device);
            counters->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["image"].setBinding(out.colour.rhi());
                cursor["imageDepth"].setBinding(out.depth.rhi());
                cursor["halves"].setBinding(halves.rhi());
                cursor["params"]["thetaBins"].setData(h);   // rows
                cursor["params"]["phiBins"].setData(w);     // columns
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t n[4] = {0, 0, 0, 0};
        REQUIRE(halves.read(*gpu->device, 0, sizeof(n), n));
        std::printf("  %-38s: left %u drawn %u lit, right %u drawn %u lit\n", c.name, n[0], n[1], n[2], n[3]);
        CHECK(n[0] > 1000);          // the left square is there
        CHECK(n[1] == n[0]);         // and the light reaches all of it
        CHECK(n[2] > 1000);          // the right square is there too
        CHECK(n[3] == (c.rightLit ? n[2] : 0u));
    }
}

TEST_CASE("an occluder outside a light's shadow link casts nothing", "[technique][lights][linking]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu->library);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    auto scene = world::GpuScene::create(*gpu->library);
    auto accel = world::RayTracingScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto shading = technique::MaterialShading::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!shading) FAIL(shading.error().toString());
    if (!table) FAIL(table.error().toString());
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
    // The plane, and an occluder between it and a point light. The occluder is
    // in category 1.
    std::array<world::MeshInstance, 2> both;
    both[0].mesh = lambertSquare(*builder, 1.0F);
    both[0].objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    both[0].material = 1;
    both[1].mesh = lambertSquare(*builder, 0.25F);
    both[1].objectToWorld = aofx::xform::translation({0.0, 0.0, -4.0});
    both[1].material = 1;
    both[1].categories = uint64_t{1} << 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), both.size()), projection));
    REQUIRE(accel->build(*scene));
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());

    struct Case {
        const char* name;
        uint32_t    shadowCategory;
        bool        shadowed;
    };
    // Linked to the category the occluder carries, it shadows; linked to
    // another, it does not and the plane is lit as if nothing were there.
    const std::array<Case, 2> cases{Case{"shadow link names the occluder's category", 1, true},
                                    Case{"shadow link names another category", 0, false}};
    for (const Case& c : cases) {
        light::Light lamp;
        lamp.kind = light::LightKind::Sphere;
        lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
        lamp.radius = 0.0F;
        lamp.shadow = true;
        lamp.shadowCategory = c.shadowCategory;
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
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
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
                const float normal[4] = {0.0F, 0.0F, 1.0F, 0.8F};
                const float radiance[4] = {1.0F, 1.0F, 1.0F, 0.02F};
                // The umbra is expected only where the occluder casts one.
                const float occluder[4] = {0.0F, 0.0F, -4.0F, c.shadowed ? 0.25F : 0.0F};
                const float axes[4] = {4.5F, 0.0F, 0.0F, 0.0F};
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
        float probe[5] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c2), c2));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(probe), probe));
        std::printf("  %-42s: %u plane pixels, %u away from the closed form, worst %.4f\n", c.name, c2[0], c2[1],
                    double(probe[0]));
        CHECK(c2[0] > 5000);
        CHECK(c2[1] == 0);
    }
}

TEST_CASE("many lights are lit as one light of their total power", "[technique][lights][many]") {
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

    // Three sphere lights in the same place, of intensity 1, 2 and 3: one
    // light of six times the power, analytically, and a distribution over
    // them that is not uniform -- which is what a choice by power has to get
    // right and a single light can never show.
    std::array<light::Light, 3> lamps;
    for (size_t k = 0; k < lamps.size(); ++k) {
        lamps[k].kind = light::LightKind::Sphere;
        lamps[k].lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
        lamps[k].radius = 0.4F;
        lamps[k].intensity = static_cast<float>(k + 1);
        lamps[k].shadow = false;
    }
    REQUIRE(table->set(std::span<const light::Light>(lamps.data(), lamps.size())));

    for (const bool choose : {false, true}) {
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
            frame.chooseLights = choose;
            REQUIRE(shading->shade(batch, visibility, projection, frame, out));
            REQUIRE(batch.submit(true));
        }
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
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
                p["kind"].setData(uint32_t{0});   // a sphere
                p["vertices"].setData(uint32_t{512});
                p["row0"].setData(toWorld.data(), sizeof(float) * 4);
                p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
                p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
                const float centre[4] = {0.0F, 0.0F, -3.0F, 0.0F};
                const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.4F};
                const float axisY[4] = {0.0F, 1.0F, 0.0F, 0.0F};
                const float normal[4] = {0.0F, 0.0F, 1.0F, 0.8F};
                const float radiance[4] = {6.0F, 6.0F, 6.0F, 0.03F};   // 1 + 2 + 3
                const float none[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                p["centre"].setData(centre, sizeof(centre));
                p["axisX"].setData(axisX, sizeof(axisX));
                p["axisY"].setData(axisY, sizeof(axisY));
                p["normal"].setData(normal, sizeof(normal));
                p["radiance"].setData(radiance, sizeof(radiance));
                p["occluder"].setData(none, sizeof(none));
                p["occluderAxes"].setData(none, sizeof(none));
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[2] = {0, 0};
        float probe[5] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(probe), probe));
        std::printf("  three lights, %s: %u pixels, %u beyond 3%%, worst %.4f\n",
                    choose ? "one chosen a sample" : "every light a sample", c[0], c[1], double(probe[0]));
        CHECK(c[0] > 8000);
        CHECK(c[1] == 0);
    }
}

TEST_CASE("a path traced frame of one bounce agrees with the raster's direct light",
          "[technique][path]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
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
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!shading) FAIL(shading.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
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
    REQUIRE(tracer->setPrograms(*programs));
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
    // One plane and one sphere light: nothing for a bounce to find, so the
    // bounce must contribute nothing and the two frames must agree.
    world::MeshInstance instance;
    instance.mesh = lambertSquare(*builder, 1.0F);
    instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    instance.material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection));
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.4F;
    lamp.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    technique::VisibilityTargets visibility;
    render::RenderTargets direct;
    render::RenderTargets traced;
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    frame.samples = 4096;
    // No structure is bound: this case traces against nothing, which is all
    // right only because the scene holds nothing for a bounce to find, and is
    // exactly why it cannot tell a bounce that returns empty-handed from one
    // that was never traced. The case below binds a structure and puts a wall
    // where the bounce can reach it; that is the one that tests the bounce.
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        REQUIRE(shading->shade(batch, visibility, projection, frame, direct));
        REQUIRE(batch.submit(true));
    }
    // The same light, gathered by paths: many samples, accumulated, so what is
    // left between them is the estimator and not the noise.
    technique::PathSettings paths;
    paths.samples = 256;
    paths.bounces = 1;
    paths.accumulate = true;
    for (uint32_t pass = 0; pass < 16; ++pass) {
        paths.seed = pass * 7919u;
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, traced));
        REQUIRE(batch.submit(true));
    }
    CHECK(tracer->accumulated() == 16u * 256u);
    auto diff = render::compareImages(*gpu->library, direct.colour, traced.colour, w, h);
    REQUIRE(diff);
    std::printf("  one bounce against the raster's direct light: p99 %u, max %u, %llu pixels beyond 2 (%u paths)\n",
                diff->p99, diff->max, static_cast<unsigned long long>(diff->over2), tracer->accumulated());
    // A scene with nothing for a bounce to find: the two must agree to within
    // the noise both estimators still carry.
    CHECK(diff->p99 <= 2);
}

// The case above proves the bounce takes nothing away where there is nothing
// to find -- which is also what an unbound acceleration structure would look
// like. This one puts a surface where the bounce can reach it, and holds the
// same frame at nought bounces against itself at one: same seeds, same direct
// light, so the only thing left between them is the bounce. The control is the
// scene without the wall, where the two must come out identical.
TEST_CASE("the bounce carries light from a second surface", "[technique][path]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("no ray queries on this device: the kernel is generated without a bounce");
    }
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu->library);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    auto scene = world::GpuScene::create(*gpu->library);
    auto accel = world::RayTracingScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
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
    REQUIRE(tracer->setPrograms(*programs));
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
    std::array<world::MeshInstance, 2> both;
    both[0].mesh = lambertSquare(*builder, 1.0F);
    both[0].objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    both[0].material = 1;
    // A wall along the plane's right edge, turned to face it: these matrices
    // take column vectors and apply the rightmost factor first, and
    // rotationY(-90) sends local +z to world -x and local +x to world +z, so
    // the square stands at x 1 from z -5 to z -3, its face towards the plane.
    both[1].mesh = lambertSquare(*builder, 1.0F);
    both[1].objectToWorld = aofx::xform::translation({1.0, 0.0, -4.0}) * aofx::xform::rotationY(-90.0);
    both[1].material = 1;
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.4F;
    lamp.shadow = false;   // the direct term must not differ between the two
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    // Nought bounces against one, over the same seeds, for a given scene.
    const auto difference = [&](size_t instances) {
        REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), instances), projection));
        REQUIRE(accel->build(*scene));
        technique::VisibilityTargets visibility;
        technique::MaterialFrame frame;
        frame.programs = &*programs;
        frame.scene = &*scene;
        frame.records = &*records;
        frame.blob = &*blobBuffer;
        frame.textures = &**textures;
        frame.lights = &*table;
        frame.shadows = accel->topLevel();
        frame.samples = 1;
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
            REQUIRE(batch.submit(true));
        }
        render::RenderTargets out[2];
        for (uint32_t bounces = 0; bounces < 2; ++bounces) {
            tracer->restart();
            technique::PathSettings paths;
            paths.samples = 64;
            paths.bounces = bounces;
            paths.accumulate = true;
            for (uint32_t pass = 0; pass < 8; ++pass) {
                paths.seed = pass * 7919u;
                gpu::CommandBatch batch(*gpu->device);
                REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out[bounces]));
                REQUIRE(batch.submit(true));
            }
        }
        auto diff = render::compareImages(*gpu->library, out[0].colour, out[1].colour, w, h);
        REQUIRE(diff);
        return *diff;
    };

    const auto withWall = difference(2);
    std::printf("  a wall beside the plane, 0 bounces against 1: p99 %u, max %u, %llu pixels beyond 2\n",
                withWall.p99, withWall.max, static_cast<unsigned long long>(withWall.over2));
    // The bounce reaches the wall and brings its light back.
    CHECK(withWall.max > 0);
    const auto alone = difference(1);
    std::printf("  the plane alone, 0 bounces against 1:         p99 %u, max %u, %llu pixels beyond 2\n",
                alone.p99, alone.max, static_cast<unsigned long long>(alone.over2));
    // Nothing to find, and the same seeds: the two frames are the same frame.
    CHECK(alone.max == 0);
}

// The plan's last check for the integrator: the error against a converged
// reference falls as 1/sqrt(N). It is worth more than holding the path tracer
// against the raster, because two estimators wrong in the same way would agree
// with each other and still both be wrong -- a slope is a prediction that can
// fail.
//
// relMSE is a squared error, so sqrt(relMSE) is what falls as 1/sqrt(N):
// quadrupling the paths must halve it. The check is the ratio between points
// and not the value of any one of them; an absolute threshold here would be an
// invented number (this engine asks relMSE < 1e-6 of two routes that should
// agree exactly, and a noisy integrator lives orders above that).
//
// The reference is 8192 paths at 96 x 72, not the plan's 64k spp: 64k at the
// resolution the other cases use measures ~80 s for one scene, most of the
// suite on its own, and depth of reference is not what verifies the law.
// Measured cost is in docs/decisions.md.
TEST_CASE("the path traced error against a converged reference falls as one over root N",
          "[technique][path]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("no ray queries on this device: there would be no bounce to converge");
    }
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu->library);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    auto scene = world::GpuScene::create(*gpu->library);
    auto accel = world::RayTracingScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
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
    REQUIRE(tracer->setPrograms(*programs));
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *lambert, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu->device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);

    const uint32_t w = 96;
    const uint32_t h = 72;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    // The wall is what gives the estimator something to be noisy about: a
    // bounce that finds a surface, not just a light sample.
    std::array<world::MeshInstance, 2> both;
    both[0].mesh = lambertSquare(*builder, 1.0F);
    both[0].objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    both[0].material = 1;
    both[1].mesh = lambertSquare(*builder, 1.0F);
    both[1].objectToWorld = aofx::xform::translation({1.0, 0.0, -4.0}) * aofx::xform::rotationY(-90.0);
    both[1].material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), both.size()), projection));
    REQUIRE(accel->build(*scene));
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.4F;
    lamp.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    technique::VisibilityTargets visibility;
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    frame.shadows = accel->topLevel();
    frame.samples = 1;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        REQUIRE(batch.submit(true));
    }

    // Absolute seeds: random() mixes (sample + path.seed) with sample local to
    // the pass, so a seed that advances by the samples already taken makes a
    // gather cover exactly [base, base + total) whatever its split (the case
    // "the same paths gathered in one pass and in many" holds that to 4e-15).
    // Every draw and the reference get a base of their own, far apart.
    const auto gather = [&](uint32_t total, uint32_t perPass, uint32_t base, render::RenderTargets& out) {
        tracer->restart();
        technique::PathSettings paths;
        paths.samples = perPass;
        paths.bounces = 1;
        paths.accumulate = true;
        for (uint32_t taken = 0; taken < total; taken += perPass) {
            paths.seed = base;   // the sample index is absolute; the seed picks the stream
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out));
            REQUIRE(batch.submit(true));
        }
        REQUIRE(tracer->accumulated() == total);
    };

    // No reference. Two independent estimates at the same N differ by a
    // quantity whose expected square is exactly twice the estimator's
    // variance, so pairs measure 1/sqrt(N) with no floor at all -- where an
    // error against a finite reference carries the reference's own noise and,
    // worse, relMSE's noisy denominator (a-b)^2/(b^2+1e-2): measured against
    // an 8192-path reference the ladder fitted relMSE = C/N + F with F at 46%
    // of the 1024-path point while the reference's variance was 15% of F, so
    // the floor was the metric, not the reference, and no depth would have
    // removed it. Pairs sidestep both. The denominator's Jensen bias at 3%
    // relative noise is of order 0.1% and is left where it is.
    // The statistic is a plain mean squared difference, from the block-error
    // kernel at block 1 -- not relMSE, whose 1/(b^2 + 1e-2) weight lets dark
    // pixels dominate and gives it a tail heavy enough that four pairs fitted
    // exponents of -0.43 twice with adjacent ratios scattered 1.2x to 2.3x
    // around 2: reading that as a shortfall was reading noise. An unweighted
    // square has the tail of the signal, not of the weight.
    auto madeBlock = gpu::ComputeKernel::create(*gpu->library, "lrt/test/block_error", "blockError");
    if (!madeBlock) FAIL(madeBlock.error().toString());
    gpu::ComputeKernel blockError = std::move(*madeBlock);
    gpu::BufferDesc momentsDesc;
    momentsDesc.bytes = 3 * sizeof(float);
    momentsDesc.elementBytes = sizeof(float);
    momentsDesc.label = "ladder.moments";
    auto moments = gpu::Buffer::create(*gpu->device, momentsDesc);
    REQUIRE(moments);
    const auto meanSquare = [&](const render::RenderTargets& a, const render::RenderTargets& b) {
        gpu::CommandBatch batch(*gpu->device);
        blockError.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(a.colour.rhi());
            cursor["b"].setBinding(b.colour.rhi());
            cursor["moments"].setBinding(moments->rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
            cursor["params"]["block"].setData(uint32_t{1});
        });
        REQUIRE(batch.submit(true));
        float m[3] = {0.0F, 0.0F, 0.0F};
        REQUIRE(moments->read(*gpu->device, 0, sizeof(m), m));
        return static_cast<double>(m[1]) / static_cast<double>(m[2]);
    };

    // Calibrated, not chosen: five runs of a shallow ladder (32..512, eight
    // pairs) fitted -0.428, -0.435, -0.440, -0.448 and -0.448 with adjacent
    // ratios scattered from 1.47 to 2.51 around the law's 2; a deep probe
    // (256..4096, four pairs) fitted -0.488 and showed no floor -- 2048 paths
    // gave 5.5e-7 where a floor would have held it at 1.25e-6. So the law
    // holds and the shallow scatter is the statistic's. The window below is
    // that scatter with room, and it still rejects no convergence (0), a floor
    // (-0.3) and a linear law (-1).
    const std::array<uint32_t, 5> counts{64, 128, 256, 512, 1024};
    constexpr uint32_t kPairs = 6;
    std::array<double, 5> rms{};
    for (size_t k = 0; k < counts.size(); ++k) {
        std::array<double, kPairs> variances{};
        for (uint32_t pair = 0; pair < kPairs; ++pair) {
            render::RenderTargets first;
            render::RenderTargets second;
            const uint32_t base = (static_cast<uint32_t>(k) * 32u + pair * 2u + 1u) * 1000000u;
            gather(counts[k], counts[k], base, first);
            gather(counts[k], counts[k], base + 500000u, second);
            variances[pair] = meanSquare(first, second) / 2.0;
        }
        // The median over pairs, because the signal has a heavy tail of its
        // own: a bounce that lands on the wall beside the light is rare and
        // bright, and one such pair read 3.5x its seven neighbours. The mean
        // of eight is at that pair's mercy; the median is not, and it is
        // still the variance a typical pair sees.
        std::sort(variances.begin(), variances.end());
        const double median = 0.5 * (variances[kPairs / 2 - 1] + variances[kPairs / 2]);
        double mean = 0.0;
        for (double v : variances) {
            mean += v / kPairs;
        }
        rms[k] = std::sqrt(median);
        std::printf("  %5u paths: variance median %.4e, mean %.4e over %u pairs (%.4e to %.4e, spread %.2fx)\n",
                    counts[k], median, mean, kPairs, variances.front(), variances.back(),
                    variances.back() / variances.front());
    }
    // Least squares on log(rms) against log(N): the slope is the exponent.
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t k = 0; k < counts.size(); ++k) {
        const double x = std::log(static_cast<double>(counts[k]));
        const double y = std::log(rms[k]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double n = static_cast<double>(counts.size());
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    std::printf("  fitted exponent over five points: %.3f (the law: -0.500)\n", slope);
    CHECK(slope < -0.42);
    CHECK(slope > -0.58);
}

// A white furnace: the path tracer and unweighted next event estimation must
// gather the same light.
//
// This case was written as a prediction of a defect, before the defect was
// fixed. gatherLight used to weigh every non-delta light sample by
// misWeight(lightDensity, stackPdf) -- but the strategy that weight is shared
// with never covered an analytic light. BSDF sampling collects emission from
// *geometry*, a LightTable light has none, and a bounce ray that escapes
// breaks without gathering the dome it passed through, so the estimator was
// scaled down and nothing paid the remainder back.
//
// For an imageless dome the two densities are the same function: sampleLight
// returns max(dot(n,wi),0)/pi (lights.slang, kLightDome) and Oren-Nayar's
// pdfLocal returns wi.z * kInvPi (lobes.slang), and misWeight(a,a) is 0.5. The
// prediction was that a white furnace reads half of what MaterialShading reads
// with `sum += f * ls.radiance / (ls.pdf * choice.probability)`. Measured
// before the fix: p99 relative 0.5453. The weight is gone, and what is left
// here is agreement.
TEST_CASE("a white furnace gathers the same light path traced as by next event estimation",
          "[technique][path][mis]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
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
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!shading) FAIL(shading.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
    // Albedo 1: a white furnace. Roughness 0 so the lobe is the Lambert whose
    // pdf is exactly the dome's.
    auto white = (*compiler)->compileXml(
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
        "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\">\n"
        "    <input name=\"color\" type=\"color3\" value=\"1, 1, 1\" />\n"
        "    <input name=\"roughness\" type=\"float\" value=\"0\" />\n"
        "  </oren_nayar_diffuse_bsdf>\n"
        "  <surface name=\"shader\" type=\"surfaceshader\"><input name=\"bsdf\" type=\"BSDF\" nodename=\"d\" /></surface>\n"
        "  <surfacematerial name=\"material\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
        "  </surfacematerial>\n</materialx>\n");
    if (!white) FAIL(white.error().toString());
    REQUIRE(programs->setModules(std::span<const material::CompiledMaterial>(&*white, 1)));
    REQUIRE(shading->setPrograms(*programs));
    REQUIRE(tracer->setPrograms(*programs));
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *white, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu->device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);

    const uint32_t w = 96;
    const uint32_t h = 72;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    world::MeshInstance instance;
    instance.mesh = lambertSquare(*builder, 1.0F);
    instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    instance.material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection));
    // An imageless dome: no texture, so domeHasImage is false and sampleLight
    // gives the cosine density.
    light::Light sky;
    sky.kind = light::LightKind::Dome;
    sky.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&sky, 1)));

    technique::VisibilityTargets visibility;
    render::RenderTargets direct;
    render::RenderTargets traced;
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    frame.samples = 4096;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        REQUIRE(shading->shade(batch, visibility, projection, frame, direct));
        REQUIRE(batch.submit(true));
    }
    // No bounce at all: the only thing under test is the direct term's weight.
    technique::PathSettings paths;
    paths.samples = 512;
    paths.bounces = 0;
    paths.accumulate = true;
    for (uint32_t pass = 0; pass < 8; ++pass) {
        paths.seed = pass * 7919u;
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, traced));
        REQUIRE(batch.submit(true));
    }
    auto diff = render::compareHdr(*gpu->library, traced.colour, direct.colour, w, h);
    REQUIRE(diff);
    std::printf("  white furnace, path traced against unweighted NEE: relMSE %.4e, p99 relative %.4f, max %.4f\n",
                diff->relMse, diff->p99Relative, diff->maxRelative);
    // Agreement, now that nothing is weighted away. Measured with the weight
    // gone: relMSE 1.3e-11, p99 relative 0.0000, max 0.0003 -- so the bound is
    // what 4096 light samples against 4096 paths actually leave, not a number
    // chosen to pass. Before the fix this read 0.5453.
    CHECK(diff->p99Relative < 0.01);
    CHECK(diff->maxRelative < 0.01);
}

// D1: the same samples, gathered three ways, must give the same frame.
//
// random() mixes (sample + path.seed), and sample runs 0..samples-1 within a
// pass, so a seed that advances per *pass* makes two splits of the same total
// draw different sample sets -- which is why 1x256 and 4x64 are not comparable
// as the ladder above gathers them. Advance the seed by the samples taken
// instead and all three splits cover exactly [base, base + total), so what is
// left between them is accumulation order alone.
//
// This separates the two live explanations for the 1/sqrt(N) anomaly. If the
// three disagree materially, the fault is that a sample's index is relative to
// its pass. If they agree, the sampler's per-pixel structure is what is left.
TEST_CASE("the same paths gathered in one pass and in many give the same frame",
          "[technique][path]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("no ray queries on this device: the kernel is generated without a bounce");
    }
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu->library);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    auto scene = world::GpuScene::create(*gpu->library);
    auto accel = world::RayTracingScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
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
    REQUIRE(tracer->setPrograms(*programs));
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *lambert, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu->device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);

    const uint32_t w = 64;
    const uint32_t h = 48;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    std::array<world::MeshInstance, 2> both;
    both[0].mesh = lambertSquare(*builder, 1.0F);
    both[0].objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    both[0].material = 1;
    both[1].mesh = lambertSquare(*builder, 1.0F);
    both[1].objectToWorld = aofx::xform::translation({1.0, 0.0, -4.0}) * aofx::xform::rotationY(-90.0);
    both[1].material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), both.size()), projection));
    REQUIRE(accel->build(*scene));
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.4F;
    lamp.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    technique::VisibilityTargets visibility;
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    frame.shadows = accel->topLevel();
    frame.samples = 1;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        REQUIRE(batch.submit(true));
    }

    // The seed advances by the samples already taken, so every split covers
    // exactly [base, base + total).
    const uint32_t kBase = 4242u;
    const uint32_t kTotal = 256u;
    const auto gatherAbsolute = [&](uint32_t perPass, render::RenderTargets& out) {
        tracer->restart();
        technique::PathSettings paths;
        paths.samples = perPass;
        paths.bounces = 1;
        paths.accumulate = true;
        for (uint32_t taken = 0; taken < kTotal; taken += perPass) {
            paths.seed = kBase;   // the sample index is absolute; the seed picks the stream
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out));
            REQUIRE(batch.submit(true));
        }
        REQUIRE(tracer->accumulated() == kTotal);
    };

    render::RenderTargets onePass;
    render::RenderTargets fourPasses;
    render::RenderTargets sixteenPasses;
    gatherAbsolute(256, onePass);
    gatherAbsolute(64, fourPasses);
    gatherAbsolute(16, sixteenPasses);

    auto four = render::compareHdr(*gpu->library, fourPasses.colour, onePass.colour, w, h);
    auto sixteen = render::compareHdr(*gpu->library, sixteenPasses.colour, onePass.colour, w, h);
    REQUIRE(four);
    REQUIRE(sixteen);
    std::printf("  1x256 against 4x64:  relMSE %.4e, max relative %.4e\n", four->relMse, four->maxRelative);
    std::printf("  1x256 against 16x16: relMSE %.4e, max relative %.4e\n", sixteen->relMse, sixteen->maxRelative);
    // Float addition is not associative, so the three differ in accumulation
    // order and nothing else.
    CHECK(four->relMse < 1e-9);
    CHECK(sixteen->relMse < 1e-9);
}

// D5: does each pixel get its own sample sequence?
//
// The 1/sqrt(N) ladder's 256-path point spreads by a factor 3.4 between draws
// while its neighbours spread by 14% and 5%. The mechanism on the table is that
// random() mixes pixel, sample, bounce and dimension with XOR before its
// avalanche, so a pixel's pre-avalanche values are the sample's contribution
// XORed with a per-pixel constant: pixels can share sample sets, which leaves
// the mean alone and multiplies the variance of the mean over the frame.
//
// What this instrument can and cannot see, stated before it is read: the
// fingerprint folds a pixel's draws in order, so it catches pixels walking the
// same *sequence*. It does not catch pixels walking the same *set* in another
// order -- and that is the shape the XOR would produce. So a zero here narrows
// the question, it does not clear the sampler. The within-pixel counter is the
// armed control and must be 0.
TEST_CASE("each pixel's first path samples are its own", "[technique][path][rng]") {
    LRT_REQUIRE_GPU(gpu);
    auto sort = gpu::RadixSort::create(*gpu->library);
    if (!sort) FAIL(sort.error().toString());
    // test::kernel derives the entry from the module name; this probe has three
    // entries, so they are named outright.
    auto madeFingerprint = gpu::ComputeKernel::create(*gpu->library, "lrt/test/rng_probe", "rngFingerprint");
    auto madeDuplicates = gpu::ComputeKernel::create(*gpu->library, "lrt/test/rng_probe", "rngDuplicates");
    auto madeWithin = gpu::ComputeKernel::create(*gpu->library, "lrt/test/rng_probe", "rngWithinPixel");
    auto madeSetPrint = gpu::ComputeKernel::create(*gpu->library, "lrt/test/rng_probe", "rngSetPrint");
    if (!madeSetPrint) FAIL(madeSetPrint.error().toString());
    if (!madeFingerprint) FAIL(madeFingerprint.error().toString());
    if (!madeDuplicates) FAIL(madeDuplicates.error().toString());
    if (!madeWithin) FAIL(madeWithin.error().toString());
    gpu::ComputeKernel fingerprint = std::move(*madeFingerprint);
    gpu::ComputeKernel duplicates = std::move(*madeDuplicates);
    gpu::ComputeKernel withinPixel = std::move(*madeWithin);
    gpu::ComputeKernel setPrint = std::move(*madeSetPrint);

    const uint32_t w = 64;
    const uint32_t h = 48;
    const uint32_t pixels = w * h;
    const uint32_t draws = 16;
    gpu::SortBuffers buffers;
    buffers.keysLo = test::uintBuffer(*gpu->device, pixels, "rng.keysLo");
    buffers.keysHi = test::uintBuffer(*gpu->device, pixels, "rng.keysHi");
    buffers.values = test::uintBuffer(*gpu->device, pixels, "rng.values");
    buffers.scratchKeysLo = test::uintBuffer(*gpu->device, pixels, "rng.scratchKeysLo");
    buffers.scratchKeysHi = test::uintBuffer(*gpu->device, pixels, "rng.scratchKeysHi");
    buffers.scratchValues = test::uintBuffer(*gpu->device, pixels, "rng.scratchValues");
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "rng.counts");

    // The seeds the ladder's 256-path point actually used.
    for (const uint32_t seed : {15633u * 7919u, 16610u * 7919u, 4242u}) {
        const auto bindProbe = [&](rhi::ShaderCursor cursor) {
            cursor["keysLo"].setBinding(buffers.keysLo.rhi());
            cursor["keysHi"].setBinding(buffers.keysHi.rhi());
            cursor["values"].setBinding(buffers.values.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["probe"]["width"].setData(w);
            cursor["probe"]["height"].setData(h);
            cursor["probe"]["seed"].setData(seed);
            cursor["probe"]["draws"].setData(draws);
        };
        {
            gpu::CommandBatch batch(*gpu->device);
            fingerprint.dispatch(batch, {pixels, 1, 1}, bindProbe);
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(sort->sort(batch, buffers, pixels, 64));
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(*gpu->device);
            duplicates.dispatch(batch, {1, 1, 1}, bindProbe);
            withinPixel.dispatch(batch, {1, 1, 1}, bindProbe);
            REQUIRE(batch.submit(true));
        }
        uint32_t c[2] = {0, 0};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        std::printf("  seed %10u: %u of %u pixels share a sequence, %u repeats within a pixel\n", seed, c[0],
                    pixels, c[1]);
        // The control: the sample index is bijective through the avalanche.
        CHECK(c[1] == 0);

        // And again with an order-independent fingerprint, which is the one
        // that can see what the ordered one cannot: two pixels drawing the same
        // SET of values in a different order. That is the shape a GF(2) mix of
        // pixel and sample would produce.
        {
            gpu::CommandBatch batch(*gpu->device);
            setPrint.dispatch(batch, {pixels, 1, 1}, bindProbe);
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(sort->sort(batch, buffers, pixels, 64));
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(*gpu->device);
            duplicates.dispatch(batch, {1, 1, 1}, bindProbe);
            REQUIRE(batch.submit(true));
        }
        uint32_t sets[2] = {0, 0};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(sets), sets));
        std::printf("  seed %10u: %u of %u pixels share a sample SET\n", seed, sets[0], pixels);
    }
}

// D4: are neighbouring pixels' errors independent?
//
// The sampler is cleared by the fingerprints above: no two pixels share a
// sequence or a set. This asks the same question from the other end -- the
// image. If per-pixel errors are independent, the variance of a block's mean
// error falls as 1/pixelsInBlock: a 2x2 block quarters it, a 4x4 sixteenths
// it. Correlated errors do not average away like that. The mean is the same
// either way, which is why the ladder's means look sane while its spread does
// not.
TEST_CASE("neighbouring pixels' path traced errors average away as independent errors do",
          "[technique][path][rng]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("no ray queries on this device");
    }
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu->library);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    auto scene = world::GpuScene::create(*gpu->library);
    auto accel = world::RayTracingScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->device);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
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
    REQUIRE(tracer->setPrograms(*programs));
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *lambert, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu->device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);

    const uint32_t w = 64;
    const uint32_t h = 48;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    std::array<world::MeshInstance, 2> both;
    both[0].mesh = lambertSquare(*builder, 1.0F);
    both[0].objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    both[0].material = 1;
    both[1].mesh = lambertSquare(*builder, 1.0F);
    both[1].objectToWorld = aofx::xform::translation({1.0, 0.0, -4.0}) * aofx::xform::rotationY(-90.0);
    both[1].material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(both.data(), both.size()), projection));
    REQUIRE(accel->build(*scene));
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.4F;
    lamp.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    technique::VisibilityTargets visibility;
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    frame.shadows = accel->topLevel();
    frame.samples = 1;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        REQUIRE(batch.submit(true));
    }
    // Absolute seeds: the estimate's samples and the reference's never overlap.
    const auto gather = [&](uint32_t total, uint32_t perPass, uint32_t base, render::RenderTargets& out) {
        tracer->restart();
        technique::PathSettings paths;
        paths.samples = perPass;
        paths.bounces = 1;
        paths.accumulate = true;
        for (uint32_t taken = 0; taken < total; taken += perPass) {
            paths.seed = base;   // the sample index is absolute; the seed picks the stream
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out));
            REQUIRE(batch.submit(true));
        }
    };
    render::RenderTargets reference;
    render::RenderTargets estimate;
    gather(4096, 512, 1000000u, reference);
    gather(256, 256, 0u, estimate);

    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/block_error", "blockError");
    if (!made) FAIL(made.error().toString());
    gpu::ComputeKernel blockError = std::move(*made);
    gpu::BufferDesc desc;
    desc.bytes = 3 * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "block.moments";
    auto moments = gpu::Buffer::create(*gpu->device, desc);
    REQUIRE(moments);
    double variance[3] = {0.0, 0.0, 0.0};
    const uint32_t blocks[3] = {1, 2, 4};
    for (size_t k = 0; k < 3; ++k) {
        gpu::CommandBatch batch(*gpu->device);
        blockError.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(estimate.colour.rhi());
            cursor["b"].setBinding(reference.colour.rhi());
            cursor["moments"].setBinding(moments->rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
            cursor["params"]["block"].setData(blocks[k]);
        });
        REQUIRE(batch.submit(true));
        float m[3] = {0.0F, 0.0F, 0.0F};
        REQUIRE(moments->read(*gpu->device, 0, sizeof(m), m));
        const double mean = m[0] / m[2];
        variance[k] = m[1] / m[2] - mean * mean;
        std::printf("  block %u: %5.0f blocks, mean error %.3e, variance of block mean %.4e\n", blocks[k], m[2],
                    mean, variance[k]);
    }
    const double fall2 = variance[0] / variance[1];
    const double fall4 = variance[0] / variance[2];
    std::printf("  variance fell by %.2fx at 2x2 (independent: 4) and %.2fx at 4x4 (independent: 16)\n", fall2,
                fall4);
    // Independent errors average away as 1/pixels; anything markedly less says
    // neighbours share their noise. Blocks that straddle the wall's edge carry
    // a real mean-error step, which keeps the fall a little under ideal.
    CHECK(fall2 > 2.5);
    CHECK(fall4 > 8.0);
}

