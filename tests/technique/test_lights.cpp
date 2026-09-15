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
#include <tuple>
#include <vector>

#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hio/types.h>

#include "lrt/geom/Mesh.h"
#include "lrt/material/TextureStore.h"
#include "lrt/io/Ies.h"
#include "lrt/io/Vdb.h"
#include "lrt/light/LightTable.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/material/TextureStore.h"
#include "lrt/technique/MaterialShading.h"
#include "lrt/technique/Denoiser.h"
#include "lrt/technique/PathTracer.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/BvhScene.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/Instancing.h"
#include "lrt/world/RayTracingScene.h"
#include "lrt/world/VolumeSet.h"

using namespace lrt;

namespace {

/// Where this file's generated images go, as the other suites keep theirs.
std::filesystem::path scratch(const std::string& name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lrt-tests" / "technique";
    std::filesystem::create_directories(dir);
    return dir / name;
}

std::shared_ptr<const geom::GpuMesh> lambertSquare(geom::MeshBuilder& builder, float half, float shiftX = 0.0F,
                                                   uint64_t topology = 0) {
    static std::vector<float> points;
    points = {-half + shiftX, -half, 0, half + shiftX, -half, 0, half + shiftX, half, 0, -half + shiftX, half, 0};
    static const std::vector<int32_t> counts{4};
    static const std::vector<int32_t> indices{0, 1, 2, 3};
    geom::MeshInput in;
    in.source = "square";
    in.topology = topology;
    in.points = {std::as_bytes(std::span<const float>(points)), false};
    in.faceVertexCounts = counts;
    in.faceVertexIndices = indices;
    in.smoothNormals = false;
    auto mesh = builder.build(in);
    if (!mesh) FAIL(mesh.error().toString());
    return std::make_shared<const geom::GpuMesh>(std::move(*mesh));
}

/// A UV sphere of radius `radius` whose faces wind so their normals point
/// inward: a closed furnace seen from inside, with no edge for a bounce's
/// origin offset to push it through.
std::shared_ptr<const geom::GpuMesh> insideSphere(geom::MeshBuilder& builder, float radius, uint32_t rings,
                                                  uint32_t segments) {
    static std::vector<float> points;
    static std::vector<int32_t> counts;
    static std::vector<int32_t> indices;
    points.clear();
    counts.clear();
    indices.clear();
    // Poles as rings of coincident points keep every face a quad.
    for (uint32_t r = 0; r <= rings; ++r) {
        const double theta = 3.14159265358979 * double(r) / double(rings);
        for (uint32_t c = 0; c <= segments; ++c) {
            const double phi = 2.0 * 3.14159265358979 * double(c) / double(segments);
            const double rho = static_cast<double>(radius);
            points.push_back(float(rho * std::sin(theta) * std::cos(phi)));
            points.push_back(float(rho * std::cos(theta)));
            points.push_back(float(rho * std::sin(theta) * std::sin(phi)));
        }
    }
    const auto at = [&](uint32_t r, uint32_t c) { return int32_t(r * (segments + 1) + c); };
    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t c = 0; c < segments; ++c) {
            counts.push_back(4);
            // Clockwise seen from outside, so the normal faces the centre.
            indices.push_back(at(r, c));
            indices.push_back(at(r, c + 1));
            indices.push_back(at(r + 1, c + 1));
            indices.push_back(at(r + 1, c));
        }
    }
    geom::MeshInput in;
    in.source = "insideSphere";
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
    auto table = light::LightTable::create(*gpu->library);
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
        lamp.length = c.sizeY;
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
    auto table = light::LightTable::create(*gpu->library);
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
    gpu::ComputeKernel consistentReduce = kernel("lightConsistentReduce");
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
    gpu::Buffer sampleFlags = test::uintBuffer(*gpu->device, samples, "light.sampleFlags");
    auto sampleGaps = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(samples, 0.0F), "light.sampleGaps");
    REQUIRE(sampleGaps);
    const std::vector<float> zeros(size_t{bins} + 1, 0.0F);
    auto observed = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "light.observed");
    auto expected = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "light.expected");
    const std::vector<float> resultZeros(8, 0.0F);
    auto result = gpu::Buffer::fromSpan<float>(*gpu->device, resultZeros, "light.result");
    gpu::Buffer mismatches = test::uintBuffer(*gpu->device, 2, "light.mismatches");
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
    // The cylinder's length is not twice its radius, so a wrong axis could
    // not pass as the right one.
    const std::array<Case, 6> cases{Case{"sphere", light::LightKind::Sphere, 0.4F, 0.0F},
                                    Case{"disk", light::LightKind::Disk, 0.6F, 0.0F},
                                    Case{"rect", light::LightKind::Rect, 1.2F, 0.8F},
                                    Case{"sun (0.2 rad)", light::LightKind::Distant, 0.2F, 0.0F},
                                    Case{"dome", light::LightKind::Dome, 0.0F, 0.0F},
                                    Case{"cylinder", light::LightKind::Cylinder, 0.3F, 1.4F}};
    for (const Case& c : cases) {
        light::Light lamp;
        lamp.kind = c.kind;
        lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
        lamp.radius = c.sizeX;
        lamp.width = c.sizeX;
        lamp.height = c.sizeY;
        lamp.length = c.sizeY;
        lamp.angle = c.kind == light::LightKind::Distant ? c.sizeX : 0.0F;
        lamp.shadow = false;
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
        const auto bind = [&](rhi::ShaderCursor cursor) {
            // Only `lights` here: LightTable::bind would also set lightCount,
            // which this program does not declare.
            cursor["lights"].setBinding(table->records().rhi());
            cursor["bins"].setBinding(binBuffer->rhi());
            cursor["sampleFlags"].setBinding(sampleFlags.rhi());
            cursor["sampleGaps"].setBinding(sampleGaps->rhi());
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
            consistent.dispatch(batch, {samples, 1, 1}, bind);
            consistentReduce.dispatch(batch, {1, 1, 1}, bind);
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
        uint32_t hitsDiffer = 0;
        float worstRelative = 0.0F;
        REQUIRE(mismatches.read(*gpu->device, 0, sizeof(differ), &differ));
        REQUIRE(mismatches.read(*gpu->device, 4, sizeof(hitsDiffer), &hitsDiffer));
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
        // Every sampled direction is found again by lightHit, where and as
        // bright as the sample said: what the material's strategy adds is
        // the same light.
        std::printf("  %-13s: %u samples lightHit does not find as sampled\n", c.name, hitsDiffer);
        CHECK(hitsDiffer == 0);
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
    auto table = light::LightTable::create(*gpu->library);
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
    gpu::ComputeKernel consistentReduce = kernel("lightConsistentReduce");
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
    gpu::Buffer sampleFlags = test::uintBuffer(*gpu->device, samples, "light.sampleFlags");
    auto sampleGaps = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(samples, 0.0F), "light.sampleGaps");
    REQUIRE(sampleGaps);
    const std::vector<float> zeros(size_t{bins} + 1, 0.0F);
    auto observed = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "dome.observed");
    auto expected = gpu::Buffer::fromSpan<float>(*gpu->device, zeros, "dome.expected");
    auto result = gpu::Buffer::fromSpan<float>(*gpu->device, std::vector<float>(8, 0.0F), "dome.result");
    gpu::Buffer mismatches = test::uintBuffer(*gpu->device, 2, "dome.mismatches");
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
        cursor["sampleFlags"].setBinding(sampleFlags.rhi());
        cursor["sampleGaps"].setBinding(sampleGaps->rhi());
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
        consistent.dispatch(batch, {samples, 1, 1}, bind);
        consistentReduce.dispatch(batch, {1, 1, 1}, bind);
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
    uint32_t hitsDiffer = 0;
    REQUIRE(mismatches.read(*gpu->device, 4, sizeof(hitsDiffer), &hitsDiffer));
    std::printf("  dome with a sun: %u samples lightHit does not find as sampled\n", hitsDiffer);
    CHECK(hitsDiffer == 0);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
    auto table = light::LightTable::create(*gpu->library);
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
        const double mean = static_cast<double>(m[0]) / static_cast<double>(m[2]);
        variance[k] = static_cast<double>(m[1]) / static_cast<double>(m[2]) - mean * mean;
        std::printf("  block %u: %5.0f blocks, mean error %.3e, variance of block mean %.4e\n", blocks[k],
                    static_cast<double>(m[2]), mean, variance[k]);
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

// The closed furnace: a shell that everywhere emits E and reflects rho, seen
// from inside. After N bounces the radiance is E (1 + rho + ... + rho^N). With
// cosine sampling of a Lambert lobe each bounce's weight over pdf is rho with
// no variance, so this is exact, not statistical -- and it is the only check
// there is on PathSettings::bounces above one.
TEST_CASE("a closed emissive shell reads the geometric series of its bounces",
          "[technique][path][furnace]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("no ray queries on this device: no bounce to check");
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
    auto table = light::LightTable::create(*gpu->library);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
    const float kEmission = 0.2F;
    const float kAlbedo = 0.5F;
    auto glowing = (*compiler)->compileXml(
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
        "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\">\n"
        "    <input name=\"color\" type=\"color3\" value=\"0.5, 0.5, 0.5\" />\n"
        "    <input name=\"roughness\" type=\"float\" value=\"0\" />\n"
        "  </oren_nayar_diffuse_bsdf>\n"
        "  <uniform_edf name=\"edf\" type=\"EDF\">\n"
        "    <input name=\"color\" type=\"color3\" value=\"0.2, 0.2, 0.2\" />\n"
        "  </uniform_edf>\n"
        "  <surface name=\"shader\" type=\"surfaceshader\">\n"
        "    <input name=\"bsdf\" type=\"BSDF\" nodename=\"d\" />\n"
        "    <input name=\"edf\" type=\"EDF\" nodename=\"edf\" />\n"
        "  </surface>\n"
        "  <surfacematerial name=\"material\" type=\"material\">\n"
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
        "  </surfacematerial>\n</materialx>\n");
    if (!glowing) FAIL(glowing.error().toString());
    REQUIRE(programs->setModules(std::span<const material::CompiledMaterial>(&*glowing, 1)));
    REQUIRE(tracer->setPrograms(*programs));
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *glowing, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
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
    // A sphere seen from inside, not a box: a box has edges, and a bounce
    // leaving a face near one is offset by the tracer's p + (n + wi) * 1e-3
    // scale before it is traced -- which, beside an edge, puts the origin
    // outside the box, where the neighbouring face is a back face and culled,
    // so the ray escapes and that sample loses the rest of its series.
    // Measured with a box of half 2: 25 of 3072 pixels at two bounces, each
    // one sample in sixteen short, worst rho^2/(1 + rho + rho^2)/16 = 0.89%
    // exactly; overlapping the faces past the corners changed nothing, since
    // the origin was already outside. From inside a sphere n + wi always
    // points in.
    world::MeshInstance shell;
    shell.mesh = insideSphere(*builder, 2.0F, 24, 48);
    shell.material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(&shell, 1), projection));
    REQUIRE(accel->build(*scene));
    // No lights: everything the frame gathers is emission carried by bounces.
    REQUIRE(table->set(std::span<const light::Light>()));

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
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/furnace_check", "furnaceCheck");
    if (!made) FAIL(made.error().toString());
    gpu::ComputeKernel check = std::move(*made);
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "furnace.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "furnace.worst");

    for (const uint32_t bounces : {0u, 1u, 2u, 3u, 6u}) {
        render::RenderTargets out;
        tracer->restart();
        technique::PathSettings paths;
        paths.samples = 16;
        paths.bounces = bounces;
        paths.seed = 7u;
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out));
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(*gpu->device);
            check.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(out.colour.rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                cursor["furnace"]["emission"].setData(kEmission);
                cursor["furnace"]["albedo"].setData(kAlbedo);
                cursor["furnace"]["bounces"].setData(bounces);
                cursor["furnace"]["tolerance"].setData(1.0e-4F);
                cursor["furnace"]["pixels"].setData(w * h);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[2] = {0, 0};
        float largest = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(largest), &largest));
        std::printf("  %u bounces: %u covered, %u beyond 1e-4, worst relative %.2e\n", bounces, c[0], c[1],
                    static_cast<double>(largest));
        CHECK(c[0] == w * h);
        CHECK(c[1] == 0);
    }
}

// The path tracer's first-hit albedo and shading normal, which the denoiser
// takes as its guides. Counters: a Lambert of roughness 0 returns everything,
// so its directional albedo is the authored colour; the normal is unit and
// faces the eye.
TEST_CASE("the path tracer's first hit reports its albedo and shading normal", "[technique][path][aov]") {
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
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->library);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
    auto lambert = (*compiler)->compileXml(
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
        "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\">\n"
        "    <input name=\"color\" type=\"color3\" value=\"0.8, 0.6, 0.4\" />\n"
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
    world::MeshInstance instance;
    instance.mesh = lambertSquare(*builder, 1.0F);
    instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    instance.material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection));
    light::Light sky;
    sky.kind = light::LightKind::Dome;
    sky.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&sky, 1)));

    technique::VisibilityTargets visibility;
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    frame.samples = 1;
    render::RenderTargets out;
    technique::PathAux aux;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        technique::PathSettings paths;
        paths.samples = 1;
        paths.bounces = 0;
        REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out, &aux));
        REQUIRE(batch.submit(true));
    }
    REQUIRE(aux.valid());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/aux_check", "auxCheck");
    if (!made) FAIL(made.error().toString());
    gpu::ComputeKernel check = std::move(*made);
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "aux.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 2, "aux.worst");
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["planes"].setBinding(aux.planes.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            cursor["aux"]["albedoR"].setData(0.8F);
            cursor["aux"]["albedoG"].setData(0.6F);
            cursor["aux"]["albedoB"].setData(0.4F);
            cursor["aux"]["tolerance"].setData(1.0e-3F);
            cursor["aux"]["pixels"].setData(w * h);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[4] = {0, 0, 0, 0};
    float e[2] = {0.0F, 0.0F};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(e), e));
    std::printf("  %u covered: %u albedo off (worst %.2e), %u not unit (worst %.2e), %u not facing the eye\n", c[0],
                c[1], static_cast<double>(e[0]), c[2], static_cast<double>(e[1]), c[3]);
    CHECK(c[0] > 1000);
    CHECK(c[1] == 0);
    CHECK(c[2] == 0);
    CHECK(c[3] == 0);
}

// OIDN reduces the error: the plan's fifth check. A noisy frame of sixteen
// paths, denoised on the device with the engine's own buffers shared -- no
// copy to the host -- against a 4096-path reference. Guided by the first
// hit's albedo and normal, and unguided. The reduction is asserted; which of
// the two guides better is printed, since on a flat Lambert scene there is no
// reason it must be the guided one.
TEST_CASE("the denoiser lowers a path traced frame's error against a deep reference",
          "[technique][path][denoise]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("no ray queries on this device");
    }
    if (!technique::denoiserBuilt()) {
        SKIP("built without OIDN");
    }
    auto denoiser = technique::Denoiser::create(*gpu->library);
    if (!denoiser) {
        SKIP(std::string("no denoiser here: ") + denoiser.error().toString());
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
    auto table = light::LightTable::create(*gpu->library);
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
    const auto gather = [&](uint32_t total, uint32_t perPass, uint32_t seed, render::RenderTargets& out,
                            technique::PathAux* aux) {
        tracer->restart();
        technique::PathSettings paths;
        paths.samples = perPass;
        paths.bounces = 1;
        paths.accumulate = true;
        paths.seed = seed;
        for (uint32_t taken = 0; taken < total; taken += perPass) {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out, aux));
            REQUIRE(batch.submit(true));
        }
    };
    render::RenderTargets reference;
    render::RenderTargets noisy;
    technique::PathAux aux;
    gather(4096, 512, 1000000u, reference, nullptr);
    gather(16, 16, 1u, noisy, &aux);

    gpu::BufferDesc desc;
    desc.bytes = uint64_t{w} * h * 16;
    desc.elementBytes = 16;
    desc.label = "denoised";
    auto guided = gpu::Buffer::create(*gpu->device, desc);
    auto unguided = gpu::Buffer::create(*gpu->device, desc);
    REQUIRE(guided);
    REQUIRE(unguided);
    if (auto r = denoiser->denoise(noisy.colour, &aux.planes, 0, aux.normalOffsetBytes(), *guided, w, h); !r) {
        FAIL(r.error().toString());
    }
    if (auto r = denoiser->denoise(noisy.colour, nullptr, 0, 0, *unguided, w, h); !r) {
        FAIL(r.error().toString());
    }

    auto before = render::compareHdr(*gpu->library, noisy.colour, reference.colour, w, h);
    auto withGuides = render::compareHdr(*gpu->library, *guided, reference.colour, w, h);
    auto without = render::compareHdr(*gpu->library, *unguided, reference.colour, w, h);
    REQUIRE(before);
    REQUIRE(withGuides);
    REQUIRE(without);
    std::printf("  %s\n", denoiser->description().c_str());
    std::printf("  16 paths against 4096: relMSE %.4e noisy, %.4e denoised with albedo and normal, %.4e without\n",
                before->relMse, withGuides->relMse, without->relMse);
    std::printf("  max relative: %.4f noisy, %.4f with guides, %.4f without (1.0000 would be an output of zeros)\n",
                before->maxRelative, withGuides->maxRelative, without->maxRelative);
    CHECK(withGuides->relMse < 0.5 * before->relMse);
    CHECK(without->relMse < 0.5 * before->relMse);
}

// Adaptive sampling: a pixel stops when its own relative standard error falls
// below a target. What matters is not that it stops but that its estimate is
// truthful -- so the converged pixels are held against a deep reference at
// three of their own standard errors, and their second moments must never
// fall below their mean squared.
TEST_CASE("adaptive sampling stops a pixel where its error estimate says, and the estimate is truthful",
          "[technique][path][adaptive]") {
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
    auto table = light::LightTable::create(*gpu->library);
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
    // The reference, uniform and deep.
    render::RenderTargets reference;
    {
        tracer->restart();
        technique::PathSettings paths;
        paths.samples = 512;
        paths.bounces = 1;
        paths.accumulate = true;
        paths.seed = 100000u;
        for (uint32_t taken = 0; taken < 4096; taken += 512) {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, reference));
            REQUIRE(batch.submit(true));
        }
    }
    // The reference's own moments, kept before the adaptive runs overwrite
    // them: the true per-sample spread, against which a pixel's own estimate
    // can be judged.
    gpu::Buffer referenceSum = test::uintBuffer(*gpu->device, uint64_t{w} * h * 4, "reference.sum");
    gpu::Buffer referenceSquares = test::uintBuffer(*gpu->device, uint64_t{w} * h, "reference.squares");
    {
        auto copier = gpu::ComputeKernel::create(*gpu->library, "lrt/technique/buffer_copy", "copyWords");
        if (!copier) FAIL(copier.error().toString());
        gpu::CommandBatch batch(*gpu->device);
        copier->dispatch(batch, {w * h * 4, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["src"].setBinding(tracer->sum().rhi());
            cursor["dst"].setBinding(referenceSum.rhi());
            cursor["copy"]["words"].setData(w * h * 4);
        });
        copier->dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["src"].setBinding(tracer->moments().rhi());   // the squares are its first plane
            cursor["dst"].setBinding(referenceSquares.rhi());
            cursor["copy"]["words"].setData(w * h);
        });
        REQUIRE(batch.submit(true));
    }
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/adaptive_check", "adaptiveCheck");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 7, "adaptive.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 2, "adaptive.worst");
    for (const uint32_t minSamples : {16u, 64u}) {
            render::RenderTargets adaptiveOut;
        tracer->restart();
        technique::PathSettings paths;
        paths.samples = 4;
        paths.bounces = 1;
        paths.accumulate = true;
        paths.seed = 1u + minSamples;
        paths.adaptive = true;
        paths.errorTarget = 0.05F;
        paths.minSamples = minSamples;
        technique::PathProgress progress;
        uint32_t passes = 0;
        for (; passes < 512; ++passes) {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, adaptiveOut));
            REQUIRE(batch.submit(true));
            auto p = tracer->progress(visibility);
            REQUIRE(p);
            progress = *p;
            if (progress.covered > 0 && progress.converged == progress.covered) {
                break;
            }
        }
        std::printf("  min %u, target 5%%: %u of %u covered pixels stopped after %u passes (%u paths a pixel at most)\n",
                    minSamples, progress.converged, progress.covered, passes + 1, tracer->accumulated());
        CHECK(progress.covered > 0);
        CHECK(progress.converged == progress.covered);
        {
            gpu::CommandBatch batch(*gpu->device);
            made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(adaptiveOut.colour.rhi());
                cursor["reference"].setBinding(reference.colour.rhi());
                cursor["sum"].setBinding(tracer->sum().rhi());
                cursor["moments"].setBinding(tracer->moments().rhi());
                cursor["referenceSum"].setBinding(referenceSum.rhi());
                cursor["referenceSquares"].setBinding(referenceSquares.rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                cursor["adaptive"]["pixels"].setData(w * h);
                cursor["adaptive"]["referencePaths"].setData(4096.0F);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[7] = {0, 0, 0, 0, 0, 0, 0};
        float largest[2] = {0.0F, 0.0F};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(largest), largest));
        std::printf("    %u converged: %u beyond 3 of their OWN sigma (%u below the reference, worst %.1f); "
                    "%u beyond 3 TRUE sigma (worst %.1f); own sigma optimistic by 3x in %u; %u negative variances; "
                    "fewest paths %u\n",
                    c[0], c[1], c[4], static_cast<double>(largest[0]), c[5], static_cast<double>(largest[1]), c[6],
                    c[2], c[3]);
        // The means, against the true spread: three sigma leaves 0.27% by
        // chance, and these read 0.24% and 0.36%. The pixel's own estimate is
        // optimistic by threefold in 0.2-0.3% -- the paths that never saw the
        // wall -- and beyond three of its own sigma in 2.3%, down from 12%
        // before the neighbourhood floor. That last number is the method's
        // known weakness, bounded here at what it measures with room, and
        // written up rather than hidden.
        CHECK(c[5] * 100 <= c[0]);
        CHECK(c[6] * 100 <= c[0]);
        CHECK(c[1] * 20 <= c[0]);
        CHECK(c[2] == 0);
        CHECK(c[3] >= minSamples);
    }
}

// A Lambert plane under a cylinder light, against the closed form: the
// lateral surface as 512 one-sided strips, each a rectangle under Lambert's
// edge formula. The cylinder lies along world x, off the plane's normal, with
// a length that is not twice its radius -- so an axis swapped with y would
// read as a different shape, not the same one.
TEST_CASE("a Lambert plane under a cylinder light is lit as the closed form says",
          "[technique][lights][cylinder]") {
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
    auto table = light::LightTable::create(*gpu->library);
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

    const uint32_t w = 81;
    const uint32_t h = 61;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    world::MeshInstance instance;
    instance.mesh = lambertSquare(*builder, 1.0F);
    instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    instance.material = 1;
    REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection));
    light::Light lamp;
    lamp.kind = light::LightKind::Cylinder;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.3F;
    lamp.length = 1.4F;
    lamp.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    technique::VisibilityTargets visibility;
    render::RenderTargets out;
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    // A one-sided curved emitter rejects half its samples and varies over the
    // rest, so it needs far more than the flat lights' 4096. Measured against
    // the closed form at 161 x 121: 3103 of 8281 pixels beyond 2% at 4096
    // (noise, sigma ~2%), 172 at 32768 (sigma ~0.9%), 0 at 131072 (sigma
    // ~0.4%, worst 1.97%, 86 s). At 65536 the sigma is ~0.6%, so 3% is five
    // of them, and a quarter of the pixels keeps the case to a few seconds.
    frame.samples = 65536;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        REQUIRE(shading->shade(batch, visibility, projection, frame, out));
        REQUIRE(batch.submit(true));
    }
    test::dumpPpm(*gpu, "cylinder_light", out.colour, w, h);
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    // Two closed forms that share nothing: Lambert's edge formula over 512
    // strips, and dense quadrature of the form-factor integral. Both are held
    // against the render; the strips are the check, the integral the check
    // on the check.
    for (const uint32_t kind : {6u, 7u}) {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(out.colour.rhi());
            cursor["depth"].setBinding(out.depth.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor p = cursor["plane"];
            p["kind"].setData(kind);
            p["vertices"].setData(uint32_t{512});
            p["row0"].setData(toWorld.data(), sizeof(float) * 4);
            p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            const float centre[4] = {0.0F, 0.0F, -3.0F, 0.0F};
            const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.3F};   // the axis, and the radius
            const float axisY[4] = {0.0F, 1.0F, 0.0F, 1.4F};   // a radial axis, and the length
            const float normal[4] = {0.0F, 0.0F, 1.0F, 0.8F};
            const float radiance[4] = {1.0F, 1.0F, 1.0F, 0.03F};
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
        uint32_t c[2] = {0, 0};
        float probe[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(probe), probe));
        float rendered[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        REQUIRE(out.colour.read(*gpu->device, (uint64_t{h / 2} * w + w / 2 + 20) * 16, sizeof(rendered), rendered));
        std::printf("  cylinder light, %s: %u pixels, %u beyond 3%%, worst %.4f; probe pixel expects %.5f, rendered %.5f\n",
                    kind == 6 ? "strips  " : "integral", c[0], c[1], double(probe[0]), double(probe[1]),
                    double(rendered[0]));
        CHECK(c[0] > 2000);
        CHECK(c[1] == 0);
    }
}

// The cylinder sampler on its own, at one point: the irradiance factor it
// estimates against the same integral by quadrature. Nothing of shading in
// between, so a sampler's bias can be told from a shader's.
TEST_CASE("the cylinder sampler alone estimates the irradiance its integral gives",
          "[technique][lights][cylinder]") {
    LRT_REQUIRE_GPU(gpu);
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_check", "lightIrradianceAt");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer irradiance = test::uintBuffer(*gpu->device, 2, "irradiance");
    light::Light lamp;
    lamp.kind = light::LightKind::Cylinder;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.3F;
    lamp.length = 1.4F;
    lamp.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
    // Three points on the plane below: under the centre, off to the side
    // along the axis, and off to the side across it.
    const float points[3][3] = {{0.0F, 0.0F, -5.0F}, {0.9F, 0.0F, -5.0F}, {0.0F, 0.9F, -5.0F}};
    for (const auto& pt : points) {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["lights"].setBinding(table->records().rhi());
            cursor["irradiance"].setBinding(irradiance.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["thetaBins"].setData(uint32_t{1});
            p["phiBins"].setData(uint32_t{1});
            p["samples"].setData(uint32_t{1u << 20});
            const float point[4] = {pt[0], pt[1], pt[2], 0.0F};
            const float normal[4] = {0.0F, 0.0F, 1.0F, 0.0F};
            p["point"].setData(point, sizeof(point));
            p["normal"].setData(normal, sizeof(normal));
        });
        REQUIRE(batch.submit(true));
        float e[2] = {0.0F, 0.0F};
        REQUIRE(irradiance.read(*gpu->device, 0, sizeof(e), e));
        const double relative = std::abs(e[0] - e[1]) / std::max(static_cast<double>(e[1]), 1e-9);
        std::printf("  at (%.1f, %.1f, %.1f): sampler %.6f, quadrature %.6f, %.3f%% apart\n", double(pt[0]),
                    double(pt[1]), double(pt[2]), double(e[0]), double(e[1]), relative * 100.0);
        CHECK(relative < 0.005);
    }
}

// The light table's cumulative shares are accumulated on the device now,
// not on the host: checked against the power written a second time from the
// record alone, in order, for five lights of mixed kinds and exposures.
TEST_CASE("a light table's cumulative shares are each light's power, accumulated on the device",
          "[technique][lights][prefix]") {
    LRT_REQUIRE_GPU(gpu);
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_check", "lightPrefixCheck");
    if (!made) FAIL(made.error().toString());
    std::array<light::Light, 5> lamps;
    lamps[0].kind = light::LightKind::Sphere;   lamps[0].radius = 0.4F;  lamps[0].intensity = 2.0F;
    lamps[1].kind = light::LightKind::Rect;     lamps[1].width = 1.2F;   lamps[1].height = 0.8F; lamps[1].exposure = 1.0F;
    lamps[2].kind = light::LightKind::Distant;  lamps[2].intensity = 3.0F;
    lamps[3].kind = light::LightKind::Cylinder; lamps[3].radius = 0.3F;  lamps[3].length = 1.4F;
    lamps[4].kind = light::LightKind::Disk;     lamps[4].radius = 0.6F;  lamps[4].normalize = true;
    REQUIRE(table->set(std::span<const light::Light>(lamps.data(), lamps.size())));
    gpu::Buffer misses = test::uintBuffer(*gpu->device, 1, "prefix.misses");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["lights"].setBinding(table->records().rhi());
            cursor["prefixMisses"].setBinding(misses.rhi());
            cursor["prefixCount"].setData(table->count());
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t n = 0;
    REQUIRE(misses.read(*gpu->device, 0, sizeof(n), &n));
    std::printf("  five lights: %u cumulative shares miss their power\n", n);
    CHECK(n == 0);
}

// IES profiles: a synthetic LM-63 file of 1000 cos^4(theta) at five-degree
// nodes, read as authored, put on a sphere light, and sampled by the shader.
// Every node must come back as its own candela; between nodes the shader
// follows the closed form to what piecewise-linear interpolation allows.
TEST_CASE("an IES profile is read as authored and sampled at its nodes exactly", "[technique][lights][ies]") {
    LRT_REQUIRE_GPU(gpu);
    // The file, as a luminaire manufacturer would write it: keyword lines,
    // TILT=NONE, the ten and three numbers, angles, then candelas.
    std::string text = "IESNA:LM-63-2002\n[TEST] cos^4\n[MANUFAC] lucabRTrender tests\nTILT=NONE\n1 1000 1 37 1 1 2 0 0 0\n1 1 0\n";
    for (int k = 0; k < 37; ++k) {
        text += std::to_string(k * 5) + (k == 36 ? "\n" : " ");
    }
    text += "0\n";
    for (int k = 0; k < 37; ++k) {
        const double theta = k * 5.0 * 3.14159265358979 / 180.0;
        const double c = std::max(std::cos(theta), 0.0);
        text += std::to_string(1000.0 * c * c * c * c) + (k == 36 ? "\n" : " ");
    }
    auto profile = io::parseIes(text);
    if (!profile) FAIL(profile.error().toString());
    CHECK(profile->vertical.size() == 37);
    CHECK(profile->horizontal.size() == 1);
    CHECK(profile->candela.size() == 37);
    CHECK(profile->photometricType == 1);
    std::printf("  parsed: %zu vertical, %zu horizontal, %zu candelas, multiplier %.1f\n", profile->vertical.size(),
                profile->horizontal.size(), profile->candela.size(), double(profile->multiplier));

    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    lamp.radius = 0.05F;
    lamp.ies = std::make_shared<const io::IesProfile>(std::move(*profile));
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
    auto nodes = gpu::ComputeKernel::create(*gpu->library, "lrt/test/ies_check", "iesNodes");
    auto cosine = gpu::ComputeKernel::create(*gpu->library, "lrt/test/ies_check", "iesCos");
    if (!nodes) FAIL(nodes.error().toString());
    if (!cosine) FAIL(cosine.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 6, "ies.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 2, "ies.worst");
    const auto bind = [&](rhi::ShaderCursor cursor) {
        table->bind(cursor);
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst.rhi());
        cursor["check"]["nodes"].setData(uint32_t{37});
        cursor["check"]["samples"].setData(uint32_t{4096});
        cursor["check"]["exponent"].setData(4.0F);
        // Piecewise-linear over 5 degrees: error <= h^2/8 max|f''|, and for
        // 1000 cos^4 the second derivative is at most 4000 -- 0.0873^2 / 8 *
        // 4000 = 3.8 -- plus the float of the angle lists.
        cursor["check"]["bound"].setData(4.0F);
    };
    {
        gpu::CommandBatch batch(*gpu->device);
        nodes->dispatch(batch, {1, 1, 1}, bind);
        cosine->dispatch(batch, {1, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    uint32_t c[6] = {};
    float e[2] = {};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(e), e));
    std::printf("  nodes: %u of 37 off (worst %.2e relative); against 1000 cos^4: %u of 4096 beyond 4 (worst %.3f)\n",
                c[0], double(e[0]), c[1], double(e[1]));
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
}


// Light instancing: one sphere light under an instancer of four elements,
// against the four lights authored where the instancer puts them. The
// table copies the prototype and a kernel places the copies from the rows
// world::Instancing composed; a check kernel compares every record with its
// authored twin. The instancer and the prototype both rotate, so a product
// in the wrong order shows.
TEST_CASE("an instanced light's records are the lights authored where the instancer puts them",
          "[technique][lights][instancing]") {
    LRT_REQUIRE_GPU(gpu);
    auto instancing = world::Instancing::create(*gpu->library);
    if (!instancing) FAIL(instancing.error().toString());
    auto instanced = light::LightTable::create(*gpu->library);
    auto authored = light::LightTable::create(*gpu->library);
    if (!instanced) FAIL(instanced.error().toString());
    if (!authored) FAIL(authored.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_check", "lightInstanceCheck");
    if (!made) FAIL(made.error().toString());

    const std::vector<float> translations{-2, 0, 0, 0, 1, -1, 2, 0.5F, 0, 0.5F, -1, 1};
    const std::vector<int32_t> indices{2, 0, 3, 1};   // the prototype takes the elements out of order
    world::InstancerLevel level;
    level.indices = indices;
    level.translations = {std::as_bytes(std::span<const float>(translations)), false};
    level.instancerTransform = aofx::xform::rotationY(30.0) * aofx::xform::translation({0.0, 0.25, 0.0});
    const std::array<world::InstancerLevel, 1> levels{level};
    auto chain = instancing->compose(levels);
    if (!chain) FAIL(chain.error().toString());
    REQUIRE(chain->count == 4);

    light::Light proto;
    proto.kind = light::LightKind::Rect;   // oriented: its rows carry the rotation
    proto.width = 0.8F;
    proto.height = 0.5F;
    proto.intensity = 3.0F;
    proto.lightToWorld = aofx::xform::translation({0.1, 0.0, -0.3}) * aofx::xform::rotationX(-20.0);
    light::Light copies = proto;
    copies.instanceRows = &chain->rows;
    copies.instanceCount = chain->count;
    // A plain light before and after, so the copies sit inside the table.
    light::Light before;
    before.kind = light::LightKind::Sphere;
    before.radius = 0.3F;
    light::Light after;
    after.kind = light::LightKind::Distant;
    after.intensity = 2.0F;
    const std::array<light::Light, 3> withInstances{before, copies, after};
    REQUIRE(instanced->set(std::span<const light::Light>(withInstances.data(), withInstances.size())));
    CHECK(instanced->count() == 6);

    std::vector<light::Light> oneByOne{before};
    for (const int32_t e : indices) {
        light::Light l = proto;
        l.lightToWorld = level.instancerTransform *
                         aofx::xform::translation({translations[size_t(e) * 3], translations[size_t(e) * 3 + 1],
                                                   translations[size_t(e) * 3 + 2]}) *
                         proto.lightToWorld;
        oneByOne.push_back(l);
    }
    oneByOne.push_back(after);
    REQUIRE(authored->set(std::span<const light::Light>(oneByOne.data(), oneByOne.size())));

    gpu::Buffer misses = test::uintBuffer(*gpu->device, 2, "instances.misses");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["lights"].setBinding(instanced->records().rhi());
            cursor["authored"].setBinding(authored->records().rhi());
            cursor["instanceMisses"].setBinding(misses.rhi());
            cursor["prefixCount"].setData(instanced->count());
            cursor["authoredCount"].setData(authored->count());
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t m[2] = {0, 0};
    REQUIRE(misses.read(*gpu->device, 0, sizeof(m), m));
    std::printf("  instanced rect light: %u of %u records differ from the authored lights\n", m[0], m[1]);
    CHECK(m[1] == 6);
    CHECK(m[0] == 0);
}

// The camera's lens. A uniformly lit Lambert plane parallel to the image,
// ending in a straight vertical edge, drawn by the path tracer's own primary
// rays: in focus the frame must be the pinhole's (every lens ray through a
// pixel meets its pixel ray there), and out of focus the edge must follow the
// circular-segment profile of a uniform disk whose radius the thin lens
// gives -- lens radius * |z - focus| / (z * focus) * focal, in pixels.
namespace {

struct LensFrame {
    gpu::Buffer colour;
    uint32_t    width = 0;
    uint32_t    height = 0;
};

/// The plane moves `motionShiftX` along x over the shutter when that is not
/// 0: as a transform (rigid) or, `deform`, as its points under the same
/// transform; drawn in `buckets` shutter slices.
struct PlaneMotion {
    double   shiftX = 0.0;
    bool     deform = false;
    uint32_t buckets = 1;
};

/// What else a frame of the plane can hold: volumes, the sun's shadows,
/// another light in the sun's place, bounces.
struct LensExtras {
    const world::VolumeSet* volumes = nullptr;
    bool                    sunShadows = false;
    const light::Light*     light = nullptr;   ///< in place of the sun
    uint32_t                bounces = 0;
};

LensFrame renderLensPlane(test::Gpu& gpu, const render::Projection& projection, uint32_t w, uint32_t h,
                          double planeZ, double half, double rightEdgeX, uint32_t samples,
                          PlaneMotion motion = {}, const LensExtras& extras = {}) {
    const world::VolumeSet* volumes = extras.volumes;
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu.device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu.library);
    auto builder = geom::MeshBuilder::create(*gpu.library);
    auto scene = world::GpuScene::create(*gpu.library);
    auto accel = world::RayTracingScene::create(*gpu.library);
    auto raster = technique::VisibilityRaster::create(*gpu.library);
    auto programs = technique::MaterialPrograms::create(*gpu.library);
    auto tracer = technique::PathTracer::create(*gpu.library);
    auto table = light::LightTable::create(*gpu.library);
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
    if (auto set = tracer->setPrograms(*programs); !set) FAIL(set.error().toString());
    const std::vector<float> blob = material::MaterialCompiler::parameters(
        *lambert, **textures, [](const std::string&) { return 0xFFFFFFFFu; });
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}, {1, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu.device, rows, "materials.records");
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu.device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);
    world::MeshInstance instance;
    instance.mesh = lambertSquare(*builder, static_cast<float>(half), 0.0F, 11);
    instance.objectToWorld = aofx::xform::translation({rightEdgeX - half, 0.0, planeZ});
    instance.material = 1;
    if (motion.shiftX != 0.0 || motion.buckets > 1) {
        world::MeshMotion m;
        m.objectToWorldStart = instance.objectToWorld;
        if (motion.deform) {
            m.objectToWorldEnd = instance.objectToWorld;
            m.meshEnd = lambertSquare(*builder, static_cast<float>(half), static_cast<float>(motion.shiftX), 11);
        } else {
            m.objectToWorldEnd = aofx::xform::translation({rightEdgeX - half + motion.shiftX, 0.0, planeZ});
        }
        instance.motion = m;
    }
    REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection, {}, motion.buckets));
    REQUIRE(accel->build(*scene));
    // A sun down -z: the plane faces +z, so its irradiance is uniform.
    light::Light sun;
    sun.kind = light::LightKind::Distant;
    sun.intensity = 3.0F;
    sun.shadow = extras.sunShadows;
    REQUIRE(table->set(std::span<const light::Light>(extras.light != nullptr ? extras.light : &sun, 1)));
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
    if (volumes != nullptr) {
        frame.volumes = &volumes->words();
        frame.volumeCount = volumes->count();
    }
    render::RenderTargets out;
    {
        gpu::CommandBatch batch(*gpu.device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, visibility));
        technique::PathSettings paths;
        paths.samples = samples;
        paths.bounces = extras.bounces;
        REQUIRE(tracer->trace(batch, visibility, projection, frame, paths, out, nullptr));
        REQUIRE(batch.submit(true));
    }
    LensFrame result;
    result.colour = std::move(out.colour);
    result.width = w;
    result.height = h;
    return result;
}

}   // namespace

TEST_CASE("a thin lens leaves the plane in focus as the pinhole draws it, and blurs an edge out of focus by "
          "the circle of confusion",
          "[technique][path][camera][dof]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection pinhole = render::projectionFor(camera, w, h);
    render::Projection lens = pinhole;
    lens.lensRadius = 0.2;
    lens.focusDistance = 4.0;

    // In focus: the plane at the focus distance, its edge left of centre.
    {
        const LensFrame a = renderLensPlane(*gpu, pinhole, w, h, -4.0, 4.0, -0.5, 16);
        const LensFrame b = renderLensPlane(*gpu, lens, w, h, -4.0, 4.0, -0.5, 16);
        auto diff = render::compareHdr(*gpu->library, a.colour, b.colour, w, h);
        REQUIRE(diff);
        std::printf("  plane in focus, lens against pinhole: relMSE %.2e, max relative %.2e\n", diff->relMse,
                    diff->maxRelative);
        CHECK(diff->relMse < 1e-8);
    }
    // Out of focus: the plane twice as far, its edge at x = 0, blurred by a
    // disk of lensRadius * |z - f| / (z f) * focal pixels.
    {
        const double z = 8.0;
        const double radius = lens.lensRadius * std::abs(z - lens.focusDistance) / (z * lens.focusDistance) *
                              pinhole.focalX;
        const LensFrame blurred = renderLensPlane(*gpu, lens, w, h, -z, 8.0, 0.0, 4096);
        test::dumpPpm(*gpu, "dof_edge", blurred.colour, w, h);
        auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/dof_check", "dofEdge");
        if (!made) FAIL(made.error().toString());
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "dof.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "dof.worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["image"].setBinding(blurred.colour.rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                rhi::ShaderCursor d = cursor["dof"];
                d["width"].setData(w);
                d["height"].setData(h);
                d["edge"].setData(static_cast<float>(pinhole.centreX));
                d["radius"].setData(static_cast<float>(radius));
                d["plateau"].setData(static_cast<float>(pinhole.centreX - 4.0 * radius - 4.0));
                // 4096 lens samples: the coverage's standard error is at most
                // 0.5 / 64 = 0.8% of the plateau; five of those.
                d["tolerance"].setData(0.04F);
                d["band"].setData(3.0F);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[2] = {0, 0};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(e), &e));
        std::printf("  edge out of focus, circle of confusion %.2f px: %u pixels within 3 radii, %u beyond 4%% of "
                    "the segment profile, worst %.4f\n",
                    radius, c[0], c[1], static_cast<double>(e));
        CHECK(radius > 4.0);
        CHECK(c[0] > 1000);
        CHECK(c[1] == 0);
    }
}

// Radial distortion moves a straight edge: in each row the lit pixels are
// those whose distorted ray meets the plane on the lit side, and the count
// must match to the pixel.
TEST_CASE("radial lens distortion puts the edge of a plane where the distorted ray meets it",
          "[technique][path][camera][distortion]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    render::Projection projection = render::projectionFor(camera, w, h);
    projection.distortionK1 = 0.5;
    projection.distortionK2 = -0.2;
    const double z = 4.0;
    const double edgeX = 0.6;   // world x of the edge, off centre so the radial term shows
    const LensFrame frame = renderLensPlane(*gpu, projection, w, h, -z, 8.0, edgeX, 1);
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/dof_check", "distortionEdge");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "distortion.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "distortion.worst");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["image"].setBinding(frame.colour.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            rhi::ShaderCursor l = cursor["lens"];
            l["width"].setData(w);
            l["height"].setData(h);
            l["focalX"].setData(static_cast<float>(projection.focalX));
            l["focalY"].setData(static_cast<float>(projection.focalY));
            l["centreX"].setData(static_cast<float>(projection.centreX));
            l["centreY"].setData(static_cast<float>(projection.centreY));
            l["k1"].setData(static_cast<float>(projection.distortionK1));
            l["k2"].setData(static_cast<float>(projection.distortionK2));
            l["edgeX"].setData(static_cast<float>(edgeX / z));
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[4] = {0, 0, 0, 0};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    std::printf("  distorted edge: %u rows, %u off by more than a pixel, %u off at all, %u moved by the distortion\n",
                c[0], c[1], c[2], c[3]);
    CHECK(c[0] == h);
    CHECK(c[3] > h / 2);
    CHECK(c[1] == 0);
}

// Motion blur, against the staircase its buckets make. The plane's edge
// slides along x over the shutter -- as a transform, and as its points under
// a still transform -- and every pixel the edge crosses must read the
// fraction of buckets at whose centre the edge is past it. Then two buckets
// (linear between the shutter samples, exact for a translation), and no
// motion at all under eight buckets, which must be the still frame.
TEST_CASE("motion blur places a sliding edge at each shutter bucket's centre, rigid and deforming alike",
          "[technique][path][motion]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    const double z = 4.0;
    const double edgeX = -0.6;
    const double shift = 1.0;   // world units over the shutter: 57 pixels
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/motion_check", "motionEdge");
    if (!made) FAIL(made.error().toString());
    const auto check = [&](const LensFrame& frame, uint32_t buckets, float tolerance, const char* what) {
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "motion.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "motion.worst");
        const float edge0 = static_cast<float>(projection.centreX + projection.focalX * edgeX / z);
        const float edge1 = static_cast<float>(projection.centreX + projection.focalX * (edgeX + shift) / z);
        {
            gpu::CommandBatch batch(*gpu->device);
            made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["image"].setBinding(frame.colour.rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                rhi::ShaderCursor m = cursor["motion"];
                m["width"].setData(w);
                m["height"].setData(h);
                m["edge0"].setData(edge0);
                m["edge1"].setData(edge1);
                m["buckets"].setData(buckets);
                m["plateau"].setData(static_cast<float>(edge0 - 6.0));
                m["tolerance"].setData(tolerance);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[2] = {0, 0};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(e), &e));
        std::printf("  %s, %u buckets: edge from column %.1f to %.1f, %u pixels checked, %u beyond %.0f%% of the "
                    "staircase, worst %.4f\n",
                    what, buckets, edge0, edge1, c[0], c[1], tolerance * 100.0F, static_cast<double>(e));
        CHECK(c[0] > 2000);
        CHECK(c[1] == 0);
    };
    // 1024 samples: a coverage's standard error is at most 0.5 / 32; five of those.
    {
        const LensFrame rigid = renderLensPlane(*gpu, projection, w, h, -z, 8.0, edgeX, 1024, {shift, false, 8});
        test::dumpPpm(*gpu, "motion_rigid", rigid.colour, w, h);
        check(rigid, 8, 0.08F, "rigid translation");
    }
    {
        const LensFrame deforming = renderLensPlane(*gpu, projection, w, h, -z, 8.0, edgeX, 1024, {shift, true, 8});
        check(deforming, 8, 0.08F, "deforming points");
    }
    {
        const LensFrame two = renderLensPlane(*gpu, projection, w, h, -z, 8.0, edgeX, 1024, {shift, false, 2});
        check(two, 2, 0.08F, "rigid translation");
    }
    {
        const LensFrame still = renderLensPlane(*gpu, projection, w, h, -z, 8.0, edgeX, 16);
        const LensFrame bucketed = renderLensPlane(*gpu, projection, w, h, -z, 8.0, edgeX, 16, {0.0, false, 8});
        auto diff = render::compareHdr(*gpu->library, still.colour, bucketed.colour, w, h);
        REQUIRE(diff);
        std::printf("  no motion under 8 buckets against the still frame: relMSE %.2e, max relative %.2e\n",
                    diff->relMse, diff->maxRelative);
        CHECK(diff->relMse < 1e-8);
    }
}

// Invisible faces: a grid with its odd faces invisible, drawn by the three
// visibility routes through the cutout passes (which is where a hidden face
// is skipped, whatever its material). Every pixel still drawn must show the
// (instance, triangle) the full grid showed there -- hiding must not
// renumber -- and none may belong to an odd face; and about half the
// grid's pixels must be gone.
TEST_CASE("invisible faces are skipped by every route without renumbering the faces that stay",
          "[technique][visibility][hidden]") {
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
    auto bvh = world::BvhScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto trace = technique::VisibilityTrace::create(*gpu->library);
    auto walk = technique::VisibilityBvh::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/hidden_check", "hiddenCheck");
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!bvh) FAIL(bvh.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!trace) FAIL(trace.error().toString());
    if (!walk) FAIL(walk.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!check) FAIL(check.error().toString());
    REQUIRE((*textures)->commit());
    const std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}};
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*gpu->device, rows, "materials.records");
    const std::vector<float> blob{0.0F};
    auto blobBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, blob, "materials.blob");
    REQUIRE(records);
    REQUIRE(blobBuffer);

    const uint32_t w = 241;
    const uint32_t h = 181;
    const render::Projection projection =
        render::projectionFor(render::Camera::lookingAt({3.0, 4.0, 11.0}, {0.0, 0.0, 0.0}), w, h);
    const int n = 16;
    std::vector<float> points;
    std::vector<int32_t> counts;
    std::vector<int32_t> indices;
    std::vector<int32_t> oddFaces;
    for (int y = 0; y <= n; ++y) {
        for (int x = 0; x <= n; ++x) {
            const float fx = static_cast<float>(x) / n * 6.0F - 3.0F;
            const float fy = static_cast<float>(y) / n * 6.0F - 3.0F;
            points.insert(points.end(), {fx, 0.4F * std::sin(fx * 1.7F) * std::cos(fy * 1.3F), fy});
        }
    }
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int p = y * (n + 1) + x;
            counts.push_back(4);
            indices.insert(indices.end(), {p, p + n + 1, p + n + 2, p + 1});
            if ((y * n + x) % 2 == 1) {
                oddFaces.push_back(y * n + x);
            }
        }
    }
    const auto grid = [&](bool hideOdd) {
        geom::MeshInput in;
        in.source = "grid";
        in.points = {std::as_bytes(std::span<const float>(points)), false};
        in.faceVertexCounts = counts;
        in.faceVertexIndices = indices;
        if (hideOdd) {
            in.invisibleFaces = oddFaces;
        }
        auto built = builder->build(in);
        if (!built) FAIL(built.error().toString());
        return std::make_shared<const geom::GpuMesh>(std::move(*built));
    };
    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*scene;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    // The three routes' ids for a mesh, each through the cutout pass.
    struct Ids {
        technique::VisibilityTargets raster, rays, walked;
    };
    const auto routes = [&](const std::shared_ptr<const geom::GpuMesh>& mesh) {
        world::MeshInstance instance;
        instance.mesh = mesh;
        instance.doubleSided = true;
        REQUIRE(scene->update(std::span<const world::MeshInstance>(&instance, 1), projection));
        REQUIRE(accel->build(*scene));
        REQUIRE(bvh->build(*scene));
        Ids ids;
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *scene, projection, w, h, ids.raster, &frame));
        REQUIRE(trace->render(batch, *accel, projection, w, h, ids.rays, &frame));
        REQUIRE(walk->render(batch, *scene, *bvh, projection, w, h, ids.walked, &frame));
        REQUIRE(batch.submit(true));
        return ids;
    };
    const Ids full = routes(grid(false));
    const Ids hidden = routes(grid(true));
    CHECK(scene->anyHidden());
    for (const auto& [name, a, b] : {std::tuple{"raster", &full.raster, &hidden.raster},
                                     std::tuple{"rays", &full.rays, &hidden.rays},
                                     std::tuple{"compute BVH", &full.walked, &hidden.walked}}) {
        gpu::Buffer out = test::uintBuffer(*gpu->device, 4, "hidden.counts");
        auto viewA = a->ids.view(0);
        auto viewB = b->ids.view(0);
        REQUIRE(viewA);
        REQUIRE(viewB);
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["full"].setBinding((*viewA).get());
            cursor["hidden"].setBinding((*viewB).get());
            cursor["triangleFaces"].setBinding(scene->triangleFaces().rhi());
            cursor["counts"].setBinding(out.rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
            cursor["params"]["firstTriangle"].setData(scene->firstTriangle(0));
        });
        REQUIRE(batch.submit(true));
        uint32_t c[4] = {0, 0, 0, 0};
        REQUIRE(out.read(*gpu->device, 0, sizeof(c), c));
        std::printf("  %s: %u pixels drawn with the odd faces hidden (%u with every face); %u differ from the full "
                    "grid's ids where it showed an even face, %u lie on an odd face\n",
                    name, c[0], c[3], c[1], c[2]);
        CHECK(c[3] > 4000);
        CHECK(c[0] > c[3] / 3);
        CHECK(c[0] < c[3] * 2 / 3);
        CHECK(c[1] == 0);
        CHECK(c[2] == 0);
    }
}

// The light BVH: forty lights of every bounded kind scattered about, a dome
// and a sun in the unbounded list. Its nodes hold their children (boxes and
// cones, 0 violations); at a hundred random points the choice's
// probabilities over the lights sum to one; a million draws at one point
// report the probability lightPdfChoice recomputes (0 mismatches) and fill
// a histogram that follows those probabilities (chi-square); and the dome's
// share is exactly its share of the power. Then the many-lights closed form
// again, chosen through the tree: the lesson of M5 is that a per-sample
// check cannot see a biased distribution, so the histogram is the point.
TEST_CASE("the light BVH chooses lights by their importance at the point, and says the probability it did",
          "[technique][lights][bvh]") {
    LRT_REQUIRE_GPU(gpu);
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    std::vector<light::Light> lamps;
    for (int k = 0; k < 40; ++k) {
        light::Light l;
        const int kind = k % 4;
        l.kind = kind == 0 ? light::LightKind::Sphere : kind == 1 ? light::LightKind::Disk
                                                     : kind == 2 ? light::LightKind::Rect : light::LightKind::Cylinder;
        l.radius = 0.1F + 0.05F * (k % 5);
        l.width = 0.4F;
        l.height = 0.3F;
        l.length = 0.6F;
        l.intensity = 1.0F + 0.5F * (k % 7);
        l.shadow = false;
        l.lightToWorld = aofx::xform::translation({-3.0 + 0.16 * k, 0.5 + 0.9 * std::sin(k * 0.8), -1.5 + 0.07 * k}) *
                         aofx::xform::rotationX(30.0 * k) * aofx::xform::rotationY(17.0 * k);
        lamps.push_back(l);
    }
    light::Light dome;
    dome.kind = light::LightKind::Dome;
    dome.intensity = 0.7F;
    dome.shadow = false;
    lamps.push_back(dome);
    light::Light sun;
    sun.kind = light::LightKind::Distant;
    sun.intensity = 2.0F;
    sun.shadow = false;
    lamps.push_back(sun);
    REQUIRE(table->set(std::span<const light::Light>(lamps.data(), lamps.size())));
    REQUIRE(table->hasBvh());
    CHECK(table->treeNodes() == 79);
    CHECK(table->unboundedCount() == 2);
    const auto kernel = [&](const char* entry) {
        auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_bvh_check", entry);
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    const uint32_t lightCount = table->count();
    constexpr uint32_t kDraws = 1u << 20;
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "bvh.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 4, "bvh.worst");
    gpu::Buffer histogram = test::uintBuffer(*gpu->device, lightCount, "bvh.histogram");
    gpu::Buffer mismatches = test::uintBuffer(*gpu->device, 2, "bvh.mismatches");
    gpu::Buffer expected = test::uintBuffer(*gpu->device, lightCount, "bvh.expected");
    const float position[3] = {0.4F, 0.2F, -0.7F};
    const float normal[3] = {0.0F, 1.0F, 0.0F};
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["lights"].setBinding(table->records().rhi());
        cursor["nodeValues"].setBinding(table->nodeValues().rhi());
        cursor["nodeBase"].setData(table->nodeBase());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst.rhi());
        cursor["histogram"].setBinding(histogram.rhi());
        cursor["mismatches"].setBinding(mismatches.rhi());
        cursor["expected"].setBinding(expected.rhi());
        rhi::ShaderCursor c = cursor["check"];
        c["treeNodes"].setData(table->treeNodes());
        c["unboundedCount"].setData(table->unboundedCount());
        c["lightCount"].setData(lightCount);
        c["draws"].setData(kDraws);
        c["points"].setData(uint32_t{100});
        c["seed"].setData(uint32_t{97});
        c["position"].setData(position, sizeof(position));
        c["normal"].setData(normal, sizeof(normal));
    };
    gpu::ComputeKernel containment = kernel("lightBvhContainment");
    gpu::ComputeKernel pdfSum = kernel("lightBvhPdfSum");
    gpu::ComputeKernel draws = kernel("lightBvhDraws");
    gpu::ComputeKernel expectedKernel = kernel("lightBvhExpected");
    {
        gpu::CommandBatch batch(*gpu->device);
        containment.dispatch(batch, {1, 1, 1}, bind);
        REQUIRE(batch.submit(true));
        uint32_t c[3] = {0, 0, 0};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        std::printf("  %u internal nodes: %u whose box misses a child's, %u whose cone does\n", c[2], c[0], c[1]);
        CHECK(c[2] == 39);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
    }
    {
        gpu::CommandBatch batch(*gpu->device);
        pdfSum.dispatch(batch, {1, 1, 1}, bind);
        REQUIRE(batch.submit(true));
        uint32_t c[2] = {0, 0};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(e), &e));
        std::printf("  the probabilities over the lights sum to one at %u random points: %u off by more than 1e-4 "
                    "(worst %.2e)\n",
                    c[1], c[0], static_cast<double>(e));
        CHECK(c[0] == 0);
    }
    {
        gpu::CommandBatch batch(*gpu->device);
        draws.dispatch(batch, {kDraws, 1, 1}, bind);
        expectedKernel.dispatch(batch, {lightCount, 1, 1}, bind);
        REQUIRE(batch.submit(true));
        uint32_t m[2] = {0, 0};
        REQUIRE(mismatches.read(*gpu->device, 0, sizeof(m), m));
        std::vector<uint32_t> observed(lightCount);
        std::vector<float> probabilities(lightCount);
        REQUIRE(histogram.read(*gpu->device, 0, observed.size() * 4, observed.data()));
        REQUIRE(expected.read(*gpu->device, 0, probabilities.size() * 4, probabilities.data()));
        // Chi-square over the lights with a probability worth five draws.
        double statistic = 0.0;
        double dof = -1.0;
        double sum = 0.0;
        for (uint32_t k = 0; k < lightCount; ++k) {
            const double e = static_cast<double>(probabilities[k]) * kDraws;
            sum += probabilities[k];
            if (e < 5.0) continue;
            const double d = static_cast<double>(observed[k]) - e;
            statistic += d * d / e;
            dof += 1.0;
        }
        const double kz = 2.0 / (9.0 * dof);
        const double z = (std::cbrt(statistic / dof) - (1.0 - kz)) / std::sqrt(kz);
        const float domeShare = probabilities[40];
        std::printf("  %u draws: %u report a probability other than lightPdfChoice's; histogram chi2 %.1f on %.0f "
                    "dof (z %.2f), probabilities summing to %.5f; the dome's share %.4f, the sun's %.4f\n",
                    m[1], m[0], statistic, dof, z, sum, static_cast<double>(domeShare),
                    static_cast<double>(probabilities[41]));
        CHECK(m[1] == kDraws);
        CHECK(m[0] == 0);
        CHECK(std::abs(z) < 4.0);
        CHECK(std::abs(sum - 1.0) < 1e-4);
        CHECK(domeShare > 0.0F);
    }
}


// Volumes in the path tracer (M9): an absorbing box of constant density,
// laid out from a .vdb, between the camera and the evenly lit plane. The
// frame with it over the frame without is, pixel by pixel, the fraction of
// paths delta tracking let through, and must sit on Beer-Lambert along that
// pixel's ray within the binomial deviation the check derives from the
// number of paths (volume_render_check.slang); outside the box the two
// frames must agree.
TEST_CASE("the path tracer through an absorbing volume is Beer-Lambert's, pixel by pixel",
          "[technique][path][volume]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lrt-tests" / "technique";
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "absorbing_box.vdb";
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.5F};
    REQUIRE(io::writeVdbBoxes(path, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    auto grid = io::readVdbGrid(path, "density");
    if (!grid) FAIL(grid.error().toString());
    auto set = world::VolumeSet::create(*gpu->library);
    if (!set) FAIL(set.error().toString());
    // The box, 3.2 on a side, over the right half of the view: x in [0, 3.2),
    // y in [-1.6, 1.6), z in [-3.8, -0.6); nothing scatters back.
    world::VolumeInput input;
    input.grid = &*grid;
    input.objectToWorld = aofx::xform::translation({0.0, -1.6, -3.8});
    input.densityScale = 1.0F;
    input.albedo = {0.0F, 0.0F, 0.0F};
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
        REQUIRE(batch.submit(true));
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    constexpr uint32_t kPaths = 1024;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/volume_render_check", "volumeRenderCheck");
    if (!made) FAIL(made.error().toString());
    for (const bool shadows : {false, true}) {
        const LensFrame without =
            renderLensPlane(*gpu, projection, w, h, -4.0, 4.0, 4.0, kPaths, {}, {nullptr, shadows});
        const LensFrame with = renderLensPlane(*gpu, projection, w, h, -4.0, 4.0, 4.0, kPaths, {}, {&*set, shadows});
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 5, "volume.render.counts");
        gpu::Buffer sumsBuffer = test::uintBuffer(*gpu->device, 4, "volume.render.sums");
        {
            gpu::CommandBatch batch(*gpu->device);
            made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
                technique::setCamera(c["camera"], projection, w, h);
                const std::array<float, 12> rows = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
                c["toWorld"]["row0"].setData(rows.data(), sizeof(float) * 4);
                c["toWorld"]["row1"].setData(rows.data() + 4, sizeof(float) * 4);
                c["toWorld"]["row2"].setData(rows.data() + 8, sizeof(float) * 4);
                const float boxMin[3] = {0.0F, -1.6F, -3.8F};
                const float boxMax[3] = {3.2F, 1.6F, -0.6F};
                const float towardLight[3] = {0.0F, 0.0F, 1.0F};   // the sun shines down -z
                c["check"]["boxMin"].setData(boxMin, sizeof(boxMin));
                c["check"]["boxMax"].setData(boxMax, sizeof(boxMax));
                c["check"]["towardLight"].setData(towardLight, sizeof(towardLight));
                c["check"]["planeZ"].setData(-4.0F);
                c["check"]["sigma"].setData(0.5F);
                c["check"]["paths"].setData(static_cast<float>(kPaths));
                c["check"]["deviations"].setData(5.0F);
                c["check"]["shadows"].setData(uint32_t{shadows ? 1u : 0u});
                c["with"].setBinding(with.colour.rhi());
                c["without"].setBinding(without.colour.rhi());
                c["counts"].setBinding(counts.rhi());
                c["zBits"].setBinding(sumsBuffer.rhi());
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t n[5] = {};
        float sums[4] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
        REQUIRE(sumsBuffer.read(*gpu->device, 0, sizeof(sums), sums));
        const double meanSquare = n[0] > 0 ? sums[0] / n[0] : 0.0;
        std::printf("  sun shadows %s: %u pixels whose light crosses the box: %u beyond 5 binomial deviations of "
                    "Beer-Lambert, mean z^2 %.3f (1 for binomial noise alone), mean ratio %.4f against %.4f; %u "
                    "untouched, %u of them changed\n",
                    shadows ? "on" : "off", n[0], n[1], meanSquare, n[0] ? sums[1] / n[0] : 0.0F,
                    n[0] ? sums[2] / n[0] : 0.0F, n[2], n[3]);
        {
            std::vector<float> rgba(size_t{w} * h * 4);
            REQUIRE(with.colour.read(*gpu->device, 0, rgba.size() * 4, rgba.data()));
            test::dumpPpm(shadows ? "volume_absorbing_box_shadows" : "volume_absorbing_box", rgba.data(), w, h);
        }
        CHECK(n[0] > 3000);
        CHECK(n[1] == 0);
        CHECK(meanSquare > 0.7);
        CHECK(meanSquare < 1.3);
        CHECK(n[2] > 1000);
        CHECK(n[3] == 0);
    }
}


// The volume furnace: a medium that scatters everything and absorbs
// nothing (albedo 1), under a dome of radiance 1 and nothing else in view.
// Every path that meets the medium random-walks until it leaves, and each
// light sample along the way estimates what the walk's next escape would
// see -- so a path's radiance is the dome's, whatever the density and
// whatever the phase, once enough bounces are allowed. The medium is thin
// (optical depth at most 0.56 across the box's diagonal), so the walk's
// chance of 32 further collisions is below 1e-8. Judged over the covered
// pixels: the mean of each pixel's radiance over its opacity, within five
// standard errors of 1 measured from those same pixels' spread, for an
// isotropic phase and a forward one (g 0.7).
TEST_CASE("an albedo-one medium under a uniform dome reads the dome's radiance, for any phase",
          "[technique][path][volume][furnace]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lrt-tests" / "technique";
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "furnace_box.vdb";
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.5F};
    REQUIRE(io::writeVdbBoxes(path, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    auto grid = io::readVdbGrid(path, "density");
    if (!grid) FAIL(grid.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/volume_render_check", "volumeFurnaceCheck");
    if (!made) FAIL(made.error().toString());
    const uint32_t w = 81;
    const uint32_t h = 61;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    light::Light dome;
    dome.kind = light::LightKind::Dome;
    dome.intensity = 1.0F;
    dome.shadow = true;
    for (const float g : {0.0F, 0.7F}) {
        auto set = world::VolumeSet::create(*gpu->library);
        if (!set) FAIL(set.error().toString());
        world::VolumeInput input;
        input.grid = &*grid;
        input.objectToWorld = aofx::xform::translation({-1.6, -1.6, -5.0});   // centred on the view, 1.8 to 5 away
        input.densityScale = 0.2F;   // sigma 0.1
        input.albedo = {1.0F, 1.0F, 1.0F};
        input.g = g;
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
            REQUIRE(batch.submit(true));
        }
        // The plane the helper draws is a speck behind the camera.
        LensExtras extras;
        extras.volumes = &*set;
        extras.light = &dome;
        extras.bounces = 32;
        const LensFrame frame = renderLensPlane(*gpu, projection, w, h, 50.0, 0.001, 0.001, 256, {}, extras);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "furnace.counts");
        gpu::Buffer sums = test::uintBuffer(*gpu->device, 2, "furnace.sums");
        {
            gpu::CommandBatch batch(*gpu->device);
            made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
                c["furnace"]["pixels"].setData(w * h);
                c["furnace"]["minAlpha"].setData(0.25F);
                c["with"].setBinding(frame.colour.rhi());
                c["counts"].setBinding(counts.rhi());
                c["zBits"].setBinding(sums.rhi());
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t n[2] = {};
        float s[2] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
        REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
        const double count = n[0];
        const double mean = count > 0 ? s[0] / count : 0.0;
        const double spread = count > 1 ? std::sqrt(std::max(s[1] / count - mean * mean, 0.0)) : 0.0;
        const double standardError = spread / std::sqrt(std::max(count, 1.0));
        std::printf("  g %.1f: %u pixels with opacity over 0.25 (of %u the medium touched at all): mean radiance %.5f, spread "
                    "%.4f a pixel, %.2f standard errors from the dome's 1\n",
                    static_cast<double>(g), n[0], n[1], mean, spread, (mean - 1.0) / std::max(standardError, 1e-9));
        {
            std::vector<float> rgba(size_t{w} * h * 4);
            REQUIRE(frame.colour.read(*gpu->device, 0, rgba.size() * 4, rgba.data()));
            test::dumpPpm(g == 0.0F ? "volume_furnace_isotropic" : "volume_furnace_forward", rgba.data(), w, h);
        }
        CHECK(n[0] > 1000);
        CHECK(std::abs(mean - 1.0) < 5.0 * standardError);
    }
}

// A dome's and a sun's share of the lights' power, against an area light's
// (light_prefix.slang): a dome of radiance L over a scene of radius R counts
// L pi R^2 and a sun of irradiance E counts E R^2, beside a rectangle's L A.
// Before, a dome counted L whatever the scene's size, and a kitchen in
// centimetres chose its dome once in four hundred thousand samples beside a
// window light. Checked by a kernel that writes the shares again from the
// records, at two scene radii.
TEST_CASE("a dome's and a sun's share of the lights' power grows with the scene they light",
          "[technique][lights][power]") {
    LRT_REQUIRE_GPU(gpu);
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_power_check", "lightPowerCheck");
    if (!made) FAIL(made.error().toString());
    light::Light dome;
    dome.kind = light::LightKind::Dome;
    dome.intensity = 0.4F;
    light::Light sun;
    sun.kind = light::LightKind::Distant;
    sun.intensity = 2.0F;
    light::Light window;
    window.kind = light::LightKind::Rect;
    window.width = 120.0F;
    window.height = 160.0F;
    window.intensity = 6.0F;
    const std::vector<light::Light> lights{dome, sun, window};
    for (const float radius : {1.0F, 400.0F}) {
        REQUIRE(table->set(lights, radius));
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "power.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "power.worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
                c["lights"].setBinding(table->records().rhi());
                c["check"]["count"].setData(uint32_t{3});
                c["check"]["sceneRadius"].setData(radius);
                c["check"]["tolerance"].setData(1e-4F);
                c["counts"].setBinding(counts.rhi());
                c["worst"].setBinding(worst.rhi());
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t n[2] = {};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
        REQUIRE(worst.read(*gpu->device, 0, 4, &e));
        std::printf("  scene radius %.0f: %u of %u shares off the written-again powers (worst %.2e)\n",
                    static_cast<double>(radius), n[0], n[1], static_cast<double>(e));
        CHECK(n[1] == 3);
        CHECK(n[0] == 0);
    }
}
