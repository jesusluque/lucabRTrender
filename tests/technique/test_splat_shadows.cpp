// Copyright (c) 2026 lucabRTrender contributors.
//
// Splat clouds as something meshes are shadowed by.
//
// The path tracer traces triangles: until now a relit cloud lit a floor it
// never darkened, and the two techniques disagreed about a scene with both in
// it. It now dims its shadow rays by what the cloud lets through -- the same
// query the splat tracer has always had (rt_shadow.slang), over tables packed
// into one buffer because the kernel is at Metal's limit of 31 buffers.
//
// Two things to hold: that the packed tables answer what the separate ones
// do, and that a frame is darkened by exactly what a ray through the cloud
// lets through.
#include "../gpu/GpuTest.h"

#include <cstdio>

#include "../render/SplatFixtures.h"
#include "lrt/light/LightTable.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"
#include "lrt/render/GaussianRayTracer.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/scene/GpuClouds.h"
#include "lrt/technique/MaterialPrograms.h"
#include "lrt/technique/PathTracer.h"
#include "lrt/technique/SplatShadows.h"
#include "lrt/technique/Visibility.h"
#include "lrt/geom/Mesh.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/RayTracingScene.h"

using namespace lrt;

namespace {

/// Particles stacked along -z, every one isotropic, so a ray down -z through
/// the axis peaks at each centre with power 0 and takes its opacity whole.
/// The same cloud the render tests hold the unpacked query to.
test::CloudBuilder stack(float opacity, uint32_t count) {
    test::CloudBuilder built;
    for (uint32_t k = 0; k < count; ++k) {
        built.add(0.0F, 0.0F, -2.0F * float(k), opacity, 0.2F, 0.2F, 0.2F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
    }
    return built;
}

/// A square of half `half` at the origin, facing +z.
std::shared_ptr<const geom::GpuMesh> lambertSquareMesh(geom::MeshBuilder& builder, float half) {
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

TEST_CASE("the packed shadow tables answer what the separate ones answer", "[technique][gpu][splatshadow]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !caps.rayQuery) {
        SKIP("the packed query is inline: no ray queries here");
    }
    auto loader = scene::CloudLoader::create(*gpu->library);
    // The shadow query wants the proxies and their structures, which is the
    // Hardware route's business; the compute BVH builds no top level at all.
    render::RayTracerSettings settings;
    settings.route = render::RayTracingRoute::Hardware;
    auto tracer = render::GaussianRayTracer::create(*gpu->library, settings);
    auto shadows = technique::SplatShadows::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!shadows) FAIL(shadows.error().toString());

    // Four stacked particles on the axis to hold the closed forms, and twenty
    // thousand around them so the packing is a real one: the tables are then
    // 100000 entries, and a packing kernel dispatched as though its argument
    // were groups would leave all but the first 14592 of them zero -- which
    // is what happened, and what a cloud of four could not show, since one
    // group is 256 threads and four particles need nine entries.
    const float opacity = 0.5F;
    const uint32_t stacked = 4;
    test::CloudBuilder built = stack(opacity, stacked);
    const test::CloudBuilder around = test::randomCloud(20000, 7);
    built.raw.records.insert(built.raw.records.end(), around.raw.records.begin(), around.raw.records.end());
    built.raw.count += around.raw.count;
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 6.0}, {0.0, 0.0, -3.0});
    auto stats = tracer->prepare(render::projectionFor(camera, 32, 32), instances);
    if (!stats) FAIL(stats.error().toString());
    const render::ShadowScene scene = tracer->shadowScene();
    REQUIRE(scene.tlas != nullptr);

    // The same four rays the unpacked query is held to: through every
    // particle, through the nearest alone, past the cloud, and away from it.
    const std::vector<float> rays{
        0.0F, 0.0F, 5.0F, 1.0e-3F,   0.0F, 0.0F, -1.0F, 100.0F,
        0.0F, 0.0F, 5.0F, 1.0e-3F,   0.0F, 0.0F, -1.0F, 6.0F,
        3.0F, 0.0F, 5.0F, 1.0e-3F,   0.0F, 0.0F, -1.0F, 100.0F,
        0.0F, 0.0F, 5.0F, 1.0e-3F,   0.0F, 0.0F, 1.0F, 100.0F,
    };
    const uint32_t count = 4;
    auto rayBuffer = gpu::Buffer::fromSpan<float>(*gpu->device, rays, "shadow.rays");
    REQUIRE(rayBuffer);
    gpu::Buffer packedAnswers = test::uintBuffer(*gpu->device, count, "shadow.packed.answers");
    gpu::Buffer plainAnswers = test::uintBuffer(*gpu->device, count, "shadow.plain.answers");
    gpu::Buffer counters = test::uintBuffer(*gpu->device, 4, "shadow.counters");

    auto packedKernel = gpu::ComputeKernel::create(*gpu->library, "lrt/test/splat_shadow_check", "packedShadowRays");
    auto plainKernel = gpu::ComputeKernel::create(*gpu->library, "lrt/rt/rt_shadow_kernel", "rtShadowRays");
    if (!packedKernel) FAIL(packedKernel.error().toString());
    if (!plainKernel) FAIL(plainKernel.error().toString());

    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(shadows->prepare(batch, scene));
        REQUIRE(batch.submit(true));
    }
    REQUIRE(shadows->valid());
    const technique::PackedShadowLayout& where = shadows->layout();
    {
        gpu::CommandBatch batch(*gpu->device);
        plainKernel->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["frames"].setBinding(scene.frames->rhi());
            cursor["colours"].setBinding(scene.colours->rhi());
            cursor["instanceData"].setBinding(scene.instanceData->rhi());
            cursor["instanceIndices"].setBinding(scene.instanceIndices->rhi());
            cursor["rays"].setBinding(rayBuffer->rhi());
            cursor["shadow"].setBinding(plainAnswers.rhi());
            cursor["counters"].setBinding(counters.rhi());
            cursor["scene"].setBinding(scene.tlas);
            cursor["shadowParams"]["rays"].setData(count);
            cursor["shadowParams"]["cut"].setData(0.0F);
            cursor["shadowParams"]["nearZ"].setData(0.0F);
            cursor["shadowParams"]["farZ"].setData(3.0e38F);
            cursor["shadowParams"]["cull"].setData(0u);
        });
        packedKernel->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["packed"].setBinding(shadows->packed().rhi());
            cursor["rays"].setBinding(rayBuffer->rhi());
            cursor["answers"].setBinding(packedAnswers.rhi());
            cursor["cloud"].setBinding(shadows->topLevel());
            cursor["check"]["where"]["frames"].setData(where.frames);
            cursor["check"]["where"]["colours"].setData(where.colours);
            cursor["check"]["where"]["instances"].setData(where.instances);
            cursor["check"]["where"]["indices"].setData(where.indices);
            cursor["check"]["where"]["instanceCount"].setData(where.instanceCount);
            cursor["check"]["cut"].setData(0.0F);
            cursor["check"]["nearZ"].setData(0.0F);
            cursor["check"]["farZ"].setData(3.0e38F);
            cursor["check"]["rays"].setData(count);
        });
        REQUIRE(batch.submit(true));
    }
    auto differ = render::countDifferent(*gpu->library, packedAnswers, plainAnswers, count);
    REQUIRE(differ);
    // Equal answers prove nothing if both are 1: the first ray goes through
    // four particles of opacity 0.5 and must come back with much less.
    float first = 1.0F;
    REQUIRE(packedAnswers.read(*gpu->device, 0, sizeof(first), &first));
    std::printf("  the packed query reads %.4f through the four (unpacked: the render tests' closed form)\n",
                static_cast<double>(first));
    CHECK(first < 0.2F);
    std::printf("  %llu of %u rays differ between the packed tables and the separate ones\n",
                static_cast<unsigned long long>(*differ), count);
    CHECK(*differ == 0);
}

// A cloud between a light and a floor, path traced. What the cloud lets
// through is a closed form where a shadow ray goes through an isotropic
// particle's centre -- the ray peaks there with power 0, so it takes the
// particle's opacity whole and `1 - opacity` is left -- and one where it
// passes nowhere near. The frame with the cloud shadowing, over the same
// frame without it, must be those two numbers; and no pixel anywhere may come
// out brighter than it was, since a cloud only takes light away.
TEST_CASE("a cloud between a light and a surface darkens it by what it lets through",
          "[technique][gpu][splatshadow][path]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.accelerationStructure || !caps.rayQuery) {
        SKIP("needs rasterisation and inline ray queries");
    }
    std::vector<std::filesystem::path> shaderPaths;
    for (const std::string& path : gpu->device->shaderSearchPaths()) {
        shaderPaths.emplace_back(path);
    }
    auto compiler = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaderPaths);
    auto textures = material::TextureStore::create(*gpu->library);
    auto builder = geom::MeshBuilder::create(*gpu->library);
    auto meshes = world::GpuScene::create(*gpu->library);
    auto accel = world::RayTracingScene::create(*gpu->library);
    auto raster = technique::VisibilityRaster::create(*gpu->library);
    auto programs = technique::MaterialPrograms::create(*gpu->library);
    auto tracer = technique::PathTracer::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->library);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto shadows = technique::SplatShadows::create(*gpu->library);
    render::RayTracerSettings settings;
    settings.route = render::RayTracingRoute::Hardware;
    auto splatTracer = render::GaussianRayTracer::create(*gpu->library, settings);
    if (!compiler) FAIL(compiler.error().toString());
    if (!textures) FAIL(textures.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!meshes) FAIL(meshes.error().toString());
    if (!accel) FAIL(accel.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!programs) FAIL(programs.error().toString());
    if (!tracer) FAIL(tracer.error().toString());
    if (!table) FAIL(table.error().toString());
    if (!loader) FAIL(loader.error().toString());
    if (!shadows) FAIL(shadows.error().toString());
    if (!splatTracer) FAIL(splatTracer.error().toString());

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

    // A square filling the view at z = -3, a point light back at the camera,
    // and one particle halfway between them on the axis. The middle pixel's
    // shadow ray runs straight up the axis through the particle's centre; a
    // corner's passes a whole unit away from it, which is many of the
    // particle's 0.2 sigmas.
    const uint32_t w = 65;
    const uint32_t h = 49;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    world::MeshInstance floorInstance;
    floorInstance.mesh = lambertSquareMesh(*builder, 4.0F);
    floorInstance.objectToWorld = aofx::xform::translation({0.0, 0.0, -3.0});
    floorInstance.material = 1;
    REQUIRE(meshes->update(std::span<const world::MeshInstance>(&floorInstance, 1), projection));
    REQUIRE(accel->build(*meshes));

    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, 0.0});
    lamp.colour[0] = lamp.colour[1] = lamp.colour[2] = 1.0F;
    lamp.intensity = 40.0F;
    // A point: every shadow ray goes to the same place, so the one under the
    // particle goes through its centre exactly. At radius 0.02 the samples
    // spread over the sphere and passed a fifth of a sigma off the axis,
    // which read 0.5056 where the closed form is 0.5.
    lamp.radius = 0.0F;
    lamp.normalize = false;
    lamp.shadow = true;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    const float opacity = 0.5F;
    test::CloudBuilder built;
    // Narrow: a corner's shadow ray passes about half a unit from the axis,
    // which is ten of these sigmas and nothing at all of the particle. At
    // 0.2 it was three sigmas and the corner still lost 1%.
    built.add(0.0F, 0.0F, -1.5F, opacity, 0.05F, 0.05F, 0.05F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    auto prepared = splatTracer->prepare(projection, instances);
    if (!prepared) FAIL(prepared.error().toString());

    technique::VisibilityTargets visibility;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(raster->render(batch, *meshes, projection, w, h, visibility));
        REQUIRE(shadows->prepare(batch, splatTracer->shadowScene()));
        REQUIRE(batch.submit(true));
    }
    REQUIRE(shadows->valid());

    technique::MaterialFrame frame;
    frame.programs = &*programs;
    frame.scene = &*meshes;
    frame.records = &*records;
    frame.blob = &*blobBuffer;
    frame.textures = &**textures;
    frame.lights = &*table;
    frame.shadows = accel->topLevel();
    frame.samples = 1;
    technique::PathSettings paths;
    paths.samples = 4;
    paths.bounces = 0;   // direct light alone: what the cloud dims and nothing else
    paths.seed = 3u;

    const auto draw = [&](bool withCloud) {
        render::RenderTargets out;
        frame.splatShadows = withCloud ? &*shadows : nullptr;
        tracer->restart();
        gpu::CommandBatch batch(*gpu->device);
        auto traced = tracer->trace(batch, visibility, projection, frame, paths, out);
        if (!traced) FAIL(traced.error().toString());
        if (!batch.submit(true)) FAIL("submit");
        return out;
    };
    const render::RenderTargets shadowed = draw(true);
    const render::RenderTargets plain = draw(false);

    auto ratio = gpu::ComputeKernel::create(*gpu->library, "lrt/test/splat_shadow_check", "splatShadowRatio");
    if (!ratio) FAIL(ratio.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "shadow.ratio.counts");
    gpu::Buffer ratios = test::uintBuffer(*gpu->device, 2, "shadow.ratio.values");
    const uint32_t under = (h / 2) * w + w / 2;
    const uint32_t beside = 2 * w + 2;
    {
        gpu::CommandBatch batch(*gpu->device);
        ratio->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["shadowed"].setBinding(shadowed.colour.rhi());
            cursor["unshadowed"].setBinding(plain.colour.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["ratios"].setBinding(ratios.rhi());
            cursor["ratio"]["pixels"].setData(w * h);
            cursor["ratio"]["under"].setData(under);
            cursor["ratio"]["beside"].setData(beside);
            cursor["ratio"]["wantUnder"].setData(1.0F - opacity);
            cursor["ratio"]["wantBeside"].setData(1.0F);
            cursor["ratio"]["tolerance"].setData(2.0e-3F);
            cursor["ratio"]["floorLevel"].setData(1.0e-4F);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[3] = {0, 0, 0};
    float values[2] = {0.0F, 0.0F};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(ratios.read(*gpu->device, 0, sizeof(values), values));
    std::printf("  %u lit pixels, %u brighter with the cloud; under the particle %.4f (want %.4f), beside it %.4f\n",
                c[0], c[1], static_cast<double>(values[0]), static_cast<double>(1.0F - opacity),
                static_cast<double>(values[1]));
    CHECK(c[0] > w * h / 2);
    CHECK(c[1] == 0);
    CHECK(c[2] == 0);
}
