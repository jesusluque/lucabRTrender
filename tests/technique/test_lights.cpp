// Copyright (c) 2026 lucabRTrender contributors.
//
// Lights against what they analytically put on a Lambert plane: a sphere, a
// disk and a rectangle, each sampled by the shading kernel and compared with
// the closed-form irradiance (lambert_irradiance.slang), which shares no code
// with the renderer.
#include "../gpu/GpuTest.h"

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
