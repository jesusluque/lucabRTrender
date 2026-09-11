// Copyright (c) 2026 lucabRTrender contributors.
//
// USD as the scene: a cloud written to a stage and rendered through the Hydra
// delegate draws what the same cloud draws directly; points prims too; and
// the delegate loads as a plugin the way a USD application finds it.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/usd/usd/primDefinition.h>
#include <pxr/usd/usd/schemaRegistry.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/vec3h.h>
#include <cstdio>

#include "lrt/io/Readers.h"
#include "lrt/lod/Lod.h"
#include "lrt/lod/Lrtc.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/GaussianRayTracer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"
#include "lrt/usd/Export.h"
#include "lrt/usd/StageRenderer.h"

using namespace lrt;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

// Before any case builds USD's schema registry: registered later, a plugin's
// schemas are not seen by a registry that already exists.
[[maybe_unused]] const bool kPluginsRegistered = [] {
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(LRT_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "lrt-tests" / "usd";
    fs::create_directories(dir);
    return dir / name;
}

io::RawSplats cloud(uint32_t count) {
    io::RawSplats raw;
    raw.source = "usd synthetic";
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 23;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = 3; e.restColourOuter = 1;
    uint64_t state = 99;
    const auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0F;
    };
    for (uint32_t i = 0; i < count; ++i) {
        for (int k = 0; k < 23; ++k) {
            raw.records.push_back(0.0F);
        }
        float* r = raw.records.data() + raw.records.size() - 23;
        r[0] = next() * 4 - 2; r[1] = next() * 3 - 1.5F; r[2] = next() * 4 - 2;
        r[3] = next() * 6 - 3;
        r[4] = std::log(0.02F + next() * 0.2F); r[5] = std::log(0.02F + next() * 0.2F);
        r[6] = std::log(0.01F + next() * 0.1F);
        r[7] = next() - 0.5F; r[8] = next() - 0.5F; r[9] = next() - 0.5F; r[10] = next() - 0.5F;
        for (int k = 11; k < 23; ++k) {
            r[k] = next() * 2 - 1;
        }
        raw.count += 1;
    }
    return raw;
}

}   // namespace

TEST_CASE("a cloud written to USD and drawn through Hydra matches the cloud drawn directly",
          "[usd][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path path = scratch("cloud.usdc");
    fs::remove(path);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, {.addCamera = false}));

    // A shot, in a layer of its own over the exported one.
    const fs::path shot = scratch("shot.usda");
    {
        std::ofstream out(shot);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def \"Cloud\" ( references = @./cloud.usdc@</World/Splats> )\n    {\n"
               "        double3 xformOp:rotateXYZ = (0, 35, 0)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) {
        FAIL(renderer.error().toString());
    }
    auto image = (*renderer)->render("/World/Shot", 0.0, 240, 180);
    if (!image) {
        FAIL(image.error().toString());
    }

    // The same cloud, the same camera, straight to the rasteriser.
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(raw);
    REQUIRE(splats);
    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.4, 0.2, 7.0});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::RenderTargets direct;
    const std::vector<render::SplatInstance> instances{{&*splats, aofx::xform::rotationY(35.0)}};
    REQUIRE(rasterizer->render(camera, instances, settings, direct));

    auto hydra = gpu::Buffer::fromSpan<float>(*gpu->device, image->rgba, "hydra");
    REQUIRE(hydra);
    gpu::BufferDesc desc;
    desc.bytes = hydra->bytes();
    desc.elementBytes = 16;
    // Reinterpret the float buffer as float4 for the comparison kernel.
    auto hydraRgba = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(hydraRgba);
    auto diff = render::compareImages(*gpu->library, *hydraRgba, direct.colour, 240, 180);
    REQUIRE(diff);
    CHECK(diff->p99 <= 1);
    CHECK(diff->max <= 2);

    // And ray traced: the delegate's lrt:technique against the ray tracer.
    auto traced = (*renderer)->render("/World/Shot", 0.0, 240, 180, "rt");
    if (!traced) {
        FAIL(traced.error().toString());
    }
    auto tracer = render::GaussianRayTracer::create(*gpu->library);
    REQUIRE(tracer);
    render::RenderTargets directTraced;
    REQUIRE(tracer->render(camera, instances, settings, directTraced));
    auto tracedRgba = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
    REQUIRE(tracedRgba);
    auto tracedDiff = render::compareImages(*gpu->library, *tracedRgba, directTraced.colour, 240, 180);
    REQUIRE(tracedDiff);
    CHECK(tracedDiff->p99 <= 1);
    CHECK(tracedDiff->max <= 2);
}

TEST_CASE("a UsdGeomPoints prim draws through Hydra", "[usd][gpu][points]") {
    LRT_REQUIRE_GPU(gpu);
    const fs::path path = scratch("points.usda");
    fs::remove(path);
    {
        UsdStageRefPtr stage = UsdStage::CreateNew(path.string());
        auto points = UsdGeomPoints::Define(stage, SdfPath("/Points"));
        VtVec3fArray positions;
        VtVec3fArray colours;
        for (int y = -10; y <= 10; ++y) {
            for (int x = -10; x <= 10; ++x) {
                positions.push_back(GfVec3f(x * 0.1F, y * 0.1F, 0.0F));
                colours.push_back(GfVec3f(1.0F, 0.5F, 0.25F));
            }
        }
        points.CreatePointsAttr(VtValue(positions));
        points.CreateWidthsAttr(VtValue(VtFloatArray{0.08F}));
        points.SetWidthsInterpolation(UsdGeomTokens->constant);
        points.CreateDisplayColorAttr(VtValue(colours));
        auto camera = UsdGeomCamera::Define(stage, SdfPath("/Camera"));
        camera.CreateFocalLengthAttr(VtValue(35.0F));
        camera.CreateHorizontalApertureAttr(VtValue(24.576F));
        camera.CreateVerticalApertureAttr(VtValue(24.576F));
        UsdGeomXformCommonAPI(camera.GetPrim()).SetTranslate(GfVec3d(0, 0, 4));
        stage->GetRootLayer()->Save();
    }
    auto renderer = usd::StageRenderer::open(path);
    REQUIRE(renderer);
    auto image = (*renderer)->render("", 0.0, 128, 128);
    if (!image) {
        FAIL(image.error().toString());
    }
    const size_t centre = (64 * 128 + 64) * 4;
    CHECK(image->rgba[centre + 3] > 0.99F);
    CHECK(image->rgba[centre] > 0.9F);
    CHECK(image->rgba[centre + 1] == Catch::Approx(0.5F).margin(0.02F));
    // View z: the points plane sits 4 units from the camera; the corner is empty.
    CHECK(image->depth[64 * 128 + 64] == Catch::Approx(4.0F).margin(0.05F));
    CHECK(image->depth[2 * 128 + 2] == 0.0F);

    // What a host that maps Hydra's render buffers reads: converted on the
    // device, top row first, depth in the projection's [0, 1].
    auto colour = (*renderer)->mappedOutput("color");
    auto depth = (*renderer)->mappedOutput("depth");
    REQUIRE(colour);
    REQUIRE(depth);
    REQUIRE(colour->size() == size_t{128} * 128 * 16);
    REQUIRE(depth->size() == size_t{128} * 128 * 4);
    const auto floatAt = [](const std::vector<uint8_t>& bytes, size_t index) {
        float v = 0.0F;
        std::memcpy(&v, bytes.data() + index * 4, 4);
        return v;
    };
    // Row 64 from the top is row 63 from the bottom: the centre texel's neighbour, still inside the grid.
    CHECK(floatAt(*colour, (64 * 128 + 64) * 4 + 3) == image->rgba[(63 * 128 + 64) * 4 + 3]);
    CHECK(floatAt(*colour, (64 * 128 + 64) * 4 + 1) == image->rgba[(63 * 128 + 64) * 4 + 1]);
    const float centreDepth = floatAt(*depth, 64 * 128 + 64);
    CHECK(centreDepth > 0.0F);
    CHECK(centreDepth < 1.0F);
    CHECK(floatAt(*depth, 2 * 128 + 2) == 1.0F);   // nothing: the far plane
}

TEST_CASE("the hdLrt plugin loads through USD's renderer plugin registry", "[usd][plugin]") {
    const fs::path plugins = LRT_HYDRA_PLUGIN_DIR;
    PlugRegistry::GetInstance().RegisterPlugins(plugins.string());
    HfPluginDescVector descs;
    HdRendererPluginRegistry::GetInstance().GetPluginDescs(&descs);
    bool found = false;
    for (const HfPluginDesc& desc : descs) {
        found = found || desc.id == TfToken("HdLrtRendererPlugin");
    }
    CHECK(found);
    HdPluginRenderDelegateUniqueHandle delegate =
        HdRendererPluginRegistry::GetInstance().CreateRenderDelegate(TfToken("HdLrtRendererPlugin"));
    CHECK(delegate);
}

TEST_CASE("a SplatEdit authored on an ancestor Xform reaches the cloud through Hydra", "[usd][gpu][edit]") {
    LRT_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path path = scratch("edit-cloud.usdc");
    fs::remove(path);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, {.addCamera = false}));

    // The edit on a group above the cloud, as constant primvars: inherited by
    // everything under it, in each cloud's own space.
    const fs::path shot = scratch("edit-shot.usda");
    {
        std::ofstream out(shot);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def Xform \"Group\" (\n        prepend apiSchemas = [\"LrtSplatEditAPI\"]\n    )\n    {\n"
               "        bool primvars:lrt:edit:active = 1 ( interpolation = \"constant\" )\n"
               "        token primvars:lrt:edit:shape = \"sphere\" ( interpolation = \"constant\" )\n"
               "        token primvars:lrt:edit:mode = \"grade\" ( interpolation = \"constant\" )\n"
               "        float3 primvars:lrt:edit:centre = (0.5, 0, 0) ( interpolation = \"constant\" )\n"
               "        float3 primvars:lrt:edit:size = (1.5, 0, 0) ( interpolation = \"constant\" )\n"
               "        color3f primvars:lrt:edit:tint = (1, 0.3, 0.1) ( interpolation = \"constant\" )\n"
               "        float primvars:lrt:edit:saturation = 0.5 ( interpolation = \"constant\" )\n"
               "        float primvars:lrt:edit:opacity = 0.6 ( interpolation = \"constant\" )\n"
               "        bool primvars:lrt:edit:invert = 1 ( interpolation = \"constant\" )\n"
               "        def \"Cloud\" ( references = @./edit-cloud.usdc@</World/Splats> )\n        {\n"
               "            double3 xformOp:rotateXYZ = (0, 35, 0)\n"
               "            uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n        }\n    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) {
        FAIL(renderer.error().toString());
    }
    auto image = (*renderer)->render("/World/Shot", 0.0, 240, 180);
    if (!image) {
        FAIL(image.error().toString());
    }

    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(raw);
    REQUIRE(splats);
    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.4, 0.2, 7.0});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::SplatEdit edit;
    edit.active = true;
    edit.shape = render::SplatEdit::Shape::Sphere;
    edit.mode = render::SplatEdit::Mode::Grade;
    edit.centre = {0.5F, 0.0F, 0.0F};
    edit.size = {1.5F, 0.0F, 0.0F};
    edit.tint = {1.0F, 0.3F, 0.1F};
    edit.saturation = 0.5F;
    edit.opacity = 0.6F;
    edit.invert = true;
    render::RenderTargets direct, plain;
    REQUIRE(rasterizer->render(camera, std::vector<render::SplatInstance>{{&*splats, aofx::xform::rotationY(35.0), edit}},
                               settings, direct));
    REQUIRE(rasterizer->render(camera, std::vector<render::SplatInstance>{{&*splats, aofx::xform::rotationY(35.0)}},
                               settings, plain));

    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(hydra);
    auto diff = render::compareImages(*gpu->library, *hydra, direct.colour, 240, 180);
    REQUIRE(diff);
    CHECK(diff->p99 <= 1);
    CHECK(diff->max <= 2);
    auto unedited = render::compareImages(*gpu->library, *hydra, plain.colour, 240, 180);
    REQUIRE(unedited);
    CHECK(unedited->over2 > unedited->pixels / 20);   // the edit really arrived
}

TEST_CASE("a .lrtc referenced from USD is cut and streamed through Hydra as it is directly", "[usd][gpu][lod]") {
    LRT_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(cloud(20000));
    REQUIRE(splats);
    auto builder = lod::LodBuilder::create(*gpu->library);
    REQUIRE(builder);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 1000;
    auto built = builder->build(*splats, chunked);
    REQUIRE(built);
    REQUIRE(lod::writeLrtc(*gpu->device, *built, scratch("stream.lrtc")));

    const float threshold = 12.0F;
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.3, 0.4, 4.5});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    auto cutter = lod::CutSelector::create(*gpu->library);
    REQUIRE(cutter);
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);

    for (const uint64_t budget : {uint64_t{0}, uint64_t{8000}}) {
        INFO("budget " << budget);
        const fs::path shot = scratch("stream-shot-" + std::to_string(budget) + ".usda");
        {
            std::ofstream out(shot);
            out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
                   "def Xform \"World\"\n{\n"
                   "    def ParticleField3DGaussianSplat \"Cloud\" (\n"
                   "        prepend apiSchemas = [\"LrtStreamedAssetAPI\"]\n    )\n    {\n"
                   "        asset primvars:lrt:asset = @./stream.lrtc@ ( interpolation = \"constant\" )\n"
                   "        float primvars:lrt:lod:threshold = " << threshold << " ( interpolation = \"constant\" )\n"
                   "        int64 primvars:lrt:stream:budget = " << budget << " ( interpolation = \"constant\" )\n"
                   "    }\n"
                   "    def Camera \"Shot\"\n    {\n"
                   "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
                   "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
                   "        double3 xformOp:translate = (0.3, 0.4, 4.5)\n"
                   "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(shot);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/World/Shot", 0.0, settings.width, settings.height);
        if (!image) FAIL(image.error().toString());

        // The same, directly: the file read whole, or the same stream settled.
        std::unique_ptr<lod::StreamingPool> pool;
        std::optional<lod::LodCloud> whole;
        if (budget == 0) {
            auto read = lod::readLrtc(*gpu->device, scratch("stream.lrtc"));
            REQUIRE(read);
            whole.emplace(std::move(*read));
        } else {
            auto opened = lod::StreamingPool::open(*gpu->device, scratch("stream.lrtc"), {budget, 2});
            REQUIRE(opened);
            pool = std::move(*opened);
        }
        const lod::LodCloud& lodCloud = pool ? pool->cloud() : *whole;
        std::vector<lod::CutStats> stats;
        std::vector<render::SplatInstance> drawn;
        for (int round = 0; round < 16; ++round) {
            auto selected = cutter->select(projection, std::vector<lod::LodInstance>{{&lodCloud}}, threshold, &stats);
            REQUIRE(selected);
            drawn = *selected;
            if (!pool) {
                break;
            }
            pool->want(stats.front().needs);
            auto placed = pool->update(true);
            REQUIRE(placed);
            if (*placed == 0) {
                break;
            }
        }
        std::printf("  budget %llu: %u splats + %u merged drawn\n", static_cast<unsigned long long>(budget),
                    stats.front().splats, stats.front().merged);
        CHECK(stats.front().merged > 0);
        CHECK(stats.front().splats > 0);
        render::RenderTargets direct;
        REQUIRE(rasterizer->render(projection, drawn, settings, direct));

        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(hydra);
        auto diff = render::compareImages(*gpu->library, *hydra, direct.colour, settings.width, settings.height);
        REQUIRE(diff);
        std::printf("  budget %llu: Hydra against direct p99 %u, max %u\n", static_cast<unsigned long long>(budget),
                    diff->p99, diff->max);
        CHECK(diff->p99 <= 1);
        CHECK(diff->max <= 2);
    }
}

TEST_CASE("a cloud authored in half floats draws as the same cloud in floats", "[usd][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path floats = scratch("floats.usdc");
    const fs::path halves = scratch("halves.usdc");
    fs::remove(floats);
    fs::remove(halves);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, floats, {.addCamera = false}));
    {
        // The fixture: the same attributes, authored as their half twins.
        UsdStageRefPtr from = UsdStage::Open(floats.string());
        REQUIRE(from);
        const UsdVolParticleField3DGaussianSplat source(from->GetPrimAtPath(SdfPath("/World/Splats")));
        UsdStageRefPtr to = UsdStage::CreateNew(halves.string());
        UsdGeomXform::Define(to, SdfPath("/World"));
        auto target = UsdVolParticleField3DGaussianSplat::Define(to, SdfPath("/World/Splats"));
        const auto halve3 = [](const VtVec3fArray& in) {
            VtVec3hArray out(in.size());
            for (size_t i = 0; i < in.size(); ++i) {
                out[i] = GfVec3h(in[i]);
            }
            return out;
        };
        VtVec3fArray positions, scales, coefficients;
        VtQuatfArray orientations;
        VtFloatArray opacities;
        int degree = 0;
        REQUIRE(source.GetPositionsAttr().Get(&positions));
        REQUIRE(source.GetScalesAttr().Get(&scales));
        REQUIRE(source.GetOrientationsAttr().Get(&orientations));
        REQUIRE(source.GetOpacitiesAttr().Get(&opacities));
        REQUIRE(source.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(&coefficients));
        REQUIRE(source.GetRadianceSphericalHarmonicsDegreeAttr().Get(&degree));
        VtQuathArray orientationsh(orientations.size());
        for (size_t i = 0; i < orientations.size(); ++i) {
            orientationsh[i] = GfQuath(orientations[i]);
        }
        target.CreatePositionshAttr(VtValue(halve3(positions)));
        target.CreateScaleshAttr(VtValue(halve3(scales)));
        target.CreateOrientationshAttr(VtValue(orientationsh));
        target.CreateOpacitieshAttr(VtValue(VtHalfArray(opacities.begin(), opacities.end())));
        target.CreateRadianceSphericalHarmonicsCoefficientshAttr(VtValue(halve3(coefficients)));
        target.CreateRadianceSphericalHarmonicsDegreeAttr(VtValue(degree));
        REQUIRE(to->GetRootLayer()->Save());
    }
    const auto shotOf = [&](const std::string& name, const std::string& cloudFile) {
        const fs::path shot = scratch(name);
        std::ofstream out(shot);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def \"Cloud\" ( references = @./" << cloudFile << "@</World/Splats> )\n    {\n    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
        return shot;
    };
    auto a = usd::StageRenderer::open(shotOf("floats-shot.usda", "floats.usdc"));
    auto b = usd::StageRenderer::open(shotOf("halves-shot.usda", "halves.usdc"));
    if (!a) FAIL(a.error().toString());
    if (!b) FAIL(b.error().toString());
    auto imageA = (*a)->render("/World/Shot", 0.0, 240, 180);
    auto imageB = (*b)->render("/World/Shot", 0.0, 240, 180);
    if (!imageA) FAIL(imageA.error().toString());
    if (!imageB) FAIL(imageB.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = imageA->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto bufferA = gpu::Buffer::create(*gpu->device, desc, imageA->rgba.data());
    auto bufferB = gpu::Buffer::create(*gpu->device, desc, imageB->rgba.data());
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    auto diff = render::compareImages(*gpu->library, *bufferA, *bufferB, 240, 180);
    REQUIRE(diff);
    std::printf("  half against float: p99 %u, max %u\n", diff->p99, diff->max);
    // Half precision moves a unit-scale position by up to 1/1024: sub-pixel here.
    CHECK(diff->p99 <= 1);
    CHECK(diff->over2 * 100 <= diff->pixels);
}

TEST_CASE("the codeless lrt schemas register, with their defaults", "[usd][schema]") {
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(LRT_HYDRA_PLUGIN_DIR).string());
    const UsdSchemaRegistry& registry = UsdSchemaRegistry::GetInstance();
    const UsdPrimDefinition* edit = registry.FindAppliedAPIPrimDefinition(TfToken("LrtSplatEditAPI"));
    REQUIRE(edit != nullptr);
    CHECK(registry.FindAppliedAPIPrimDefinition(TfToken("LrtPointStyleAPI")) != nullptr);

    UsdStageRefPtr stage = UsdStage::CreateInMemory();
    UsdPrim group = stage->DefinePrim(SdfPath("/Group"), TfToken("Xform"));
    CHECK(registry.FindAppliedAPIPrimDefinition(TfToken("LrtStreamedAssetAPI")) != nullptr);
    CHECK(group.ApplyAPI(TfToken("LrtSplatEditAPI")));
    CHECK(group.HasAPI(TfToken("LrtSplatEditAPI")));
    // Unauthored, the schema's defaults answer.
    float saturation = 0.0F;
    CHECK(group.GetAttribute(TfToken("primvars:lrt:edit:saturation")).Get(&saturation));
    CHECK(saturation == 1.0F);
    TfToken mode;
    CHECK(group.GetAttribute(TfToken("primvars:lrt:edit:mode")).Get(&mode));
    CHECK(mode == TfToken("grade"));
    VtValue allowed;
    CHECK(group.GetAttribute(TfToken("primvars:lrt:edit:shape")).GetMetadata(TfToken("allowedTokens"), &allowed));
}
