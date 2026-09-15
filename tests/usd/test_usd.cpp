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
#include <sstream>
#include <memory>
#include <optional>

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hio/image.h>
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

#include "lrt/technique/Visibility.h"
#include "lrt/io/Exr.h"
#include "lrt/io/Vdb.h"
#include "lrt/io/Readers.h"
#include "lrt/lod/Lod.h"
#include "lrt/lod/Lrtc.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/GaussianRayTracer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"
#include "lrt/usd/Export.h"
#include "lrt/technique/Denoiser.h"
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

TEST_CASE("splats and points behind a mesh leave it as it is; in front of it they show", "[usd][gpu][mesh][layers]") {
    LRT_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path clouds = scratch("layers_cloud.usdc");
    fs::remove(clouds);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, clouds, {.addCamera = false}));
    // A wall (single sided, facing the camera) at z, splats within |z| <= 2
    // and a sheet of points at z = 0, seen from z = 9.
    const auto stage = [&](const std::string& name, double wallZ, bool withSplats, bool withPoints) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-9, -9, 0), (9, -9, 0), (9, 9, 0), (-9, 9, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.3, 0.6, 0.9)] ( interpolation = \"constant\" )\n"
               "    double3 xformOp:translate = (0, 0, " << wallZ << ")\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "}\n";
        if (withSplats) {
            out << "def \"Cloud\" ( references = @./layers_cloud.usdc@</World/Splats> )\n{\n}\n";
        }
        if (withPoints) {
            out << "def Points \"Sheet\"\n{\n    point3f[] points = [";
            for (int y = -4; y <= 4; ++y) {
                for (int x = -4; x <= 4; ++x) {
                    out << (x == -4 && y == -4 ? "" : ", ") << "(" << x * 0.3 << ", " << y * 0.3 << ", 0)";
                }
            }
            out << "]\n    float[] widths = [0.12] ( interpolation = \"constant\" )\n"
                   "    color3f[] primvars:displayColor = [(1, 0.5, 0.25)] ( interpolation = \"constant\" )\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float2 clippingRange = (0.1, 1000)\n    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    double3 xformOp:translate = (0.2, 0.1, 9)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 240;
    const uint32_t h = 180;
    const auto render = [&](const fs::path& path, const char* route) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        REQUIRE((*renderer)->setMeshVisibility(route));
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const fs::path hiddenPath = stage("layers_hidden.usda", 3.0, true, true);
    const fs::path frontWallPath = stage("layers_front_wall.usda", 3.0, false, false);
    const fs::path shownPath = stage("layers_shown.usda", -5.0, true, true);
    const fs::path backWallPath = stage("layers_back_wall.usda", -5.0, false, false);
    for (const char* route : {"raster", "rays", "bvh"}) {
        const gpu::Buffer hidden = render(hiddenPath, route);
        const gpu::Buffer frontWall = render(frontWallPath, route);
        const gpu::Buffer shown = render(shownPath, route);
        const gpu::Buffer backWall = render(backWallPath, route);
        auto behind = render::compareImages(*gpu->library, hidden, frontWall, w, h);
        auto before = render::compareImages(*gpu->library, shown, backWall, w, h);
        REQUIRE(behind);
        REQUIRE(before);
        std::printf("  %s: splats and points behind the wall change %llu pixels (max %u); in front, %llu (max %u)\n",
                    route, static_cast<unsigned long long>(behind->over2), behind->max,
                    static_cast<unsigned long long>(before->over2), before->max);
        CHECK(behind->max == 0);
        CHECK(before->over2 > uint64_t{w} * h / 10);
    }
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
    // device, bottom row first, depth in the projection's [0, 1].
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
    CHECK(floatAt(*colour, (63 * 128 + 64) * 4 + 3) == image->rgba[(63 * 128 + 64) * 4 + 3]);
    CHECK(floatAt(*colour, (63 * 128 + 64) * 4 + 1) == image->rgba[(63 * 128 + 64) * 4 + 1]);
    const float centreDepth = floatAt(*depth, 64 * 128 + 64);
    CHECK(centreDepth > 0.0F);
    CHECK(centreDepth < 1.0F);
    CHECK(floatAt(*depth, 2 * 128 + 2) == 1.0F);   // nothing: the far plane
}

TEST_CASE("a UsdGeomMesh draws through Hydra where, how deep and how lit it analytically is", "[usd][gpu][mesh]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("mesh.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.4, 0.2)] ( interpolation = \"constant\" )\n"
               "}\n"
               "def Mesh \"Guide\"\n{\n"
               "    uniform token purpose = \"guide\"\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-3, -3, -2), (3, -3, -2), (3, 3, -2), (-3, 3, -2)]\n"
               "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 161;
    const uint32_t h = 121;
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colour = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colour);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/visibility_check", "planeCheck");
    if (!check) FAIL(check.error().toString());
    gpu::BufferDesc wordsDesc;
    wordsDesc.bytes = 12;
    wordsDesc.elementBytes = 4;
    auto counts = gpu::Buffer::create(*gpu->device, wordsDesc);
    wordsDesc.bytes = 4;
    auto worst = gpu::Buffer::create(*gpu->device, wordsDesc);
    REQUIRE(counts);
    REQUIRE(worst);
    gpu::CommandBatch batch(*gpu->device);
    check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(colour->rhi());
        cursor["depth"].setBinding(depth->rhi());
        cursor["counts"].setBinding(counts->rhi());
        cursor["worst"].setBinding(worst->rhi());
        rhi::ShaderCursor c = cursor["camera"];
        c["width"].setData(w);
        c["height"].setData(h);
        c["focalX"].setData(static_cast<float>(projection.focalX));
        c["focalY"].setData(static_cast<float>(projection.focalY));
        c["centreX"].setData(static_cast<float>(projection.centreX));
        c["centreY"].setData(static_cast<float>(projection.centreY));
        c["nearZ"].setData(0.1F);
        c["farZ"].setData(1000.0F);
        c["orthographic"].setData(uint32_t{0});
        cursor["plane"]["z"].setData(5.0F);
        cursor["plane"]["half"].setData(1.0F);
        cursor["plane"]["colourR"].setData(0.8F);
        cursor["plane"]["colourG"].setData(0.4F);
        cursor["plane"]["colourB"].setData(0.2F);
    });
    REQUIRE(batch.submit(true));
    uint32_t c[3] = {0, 0, 0};
    float depthError = 0.0F;
    REQUIRE(counts->read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(depthError), &depthError));
    std::printf("  Hydra mesh: %u pixels covered, %u coverage and %u colour mismatches, depth off by %.2e\n", c[2],
                c[0], c[1], static_cast<double>(depthError));
    // The guide-purpose mesh in front is not in the geometry collection's render tags.
    CHECK(c[2] > 800);
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
    CHECK(depthError < 1e-3F);
}

namespace {

/// A stage's /Camera image against the analytic headlit square at z = -5
/// (visibility_check.slang's planeCheck): coverage, depth and colour mismatches.
std::array<uint32_t, 3> squareMismatches(test::Gpu& gpu, const usd::StageImage& image, std::array<float, 3> colour,
                                         float z = 5.0F, float half = 1.0F) {
    const uint32_t w = image.width;
    const uint32_t h = image.height;
    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image.rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu.device, colourDesc, image.rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu.device, image.depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu.library, "lrt/test/visibility_check", "planeCheck");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 3, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu.device, 1, "worst");
    gpu::CommandBatch batch(*gpu.device);
    check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(colours->rhi());
        cursor["depth"].setBinding(depth->rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst.rhi());
        rhi::ShaderCursor c = cursor["camera"];
        c["width"].setData(w);
        c["height"].setData(h);
        c["focalX"].setData(static_cast<float>(projection.focalX));
        c["focalY"].setData(static_cast<float>(projection.focalY));
        c["centreX"].setData(static_cast<float>(projection.centreX));
        c["centreY"].setData(static_cast<float>(projection.centreY));
        c["nearZ"].setData(0.1F);
        c["farZ"].setData(1000.0F);
        c["orthographic"].setData(uint32_t{0});
        cursor["plane"]["z"].setData(z);
        cursor["plane"]["half"].setData(half);
        cursor["plane"]["colourR"].setData(colour[0]);
        cursor["plane"]["colourG"].setData(colour[1]);
        cursor["plane"]["colourB"].setData(colour[2]);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 3> c{};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(c), c.data()));
    return c;
}

const char* kSquareStage = R"(#usda 1.0
(
    upAxis = "Y"
)
def Mesh "Square" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "vertex"
    )
    uniform token subdivisionScheme = "none"
    color3f[] primvars:displayColor = [(0.1, 0.9, 0.1)] (
        interpolation = "constant"
    )
    rel material:binding = </Materials/Mat>
}
def Camera "Camera"
{
    float focalLength = 35
    float horizontalAperture = 24.576
    float verticalAperture = 18.432
    float2 clippingRange = (0.1, 1000)
}
)";

}   // namespace

TEST_CASE("materials bound in USD shade a mesh: MaterialX with a texture, and UsdPreviewSurface",
          "[usd][gpu][mesh][materials]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    // A texture of one colour, exactly representable in 8 bits.
    const fs::path png = scratch("orange.png");
    {
        fs::remove(png);
        std::vector<uint32_t> texels(16 * 16, 0xFF3399CCu);   // ABGR: r 0xCC, g 0x99, b 0x33
        HioImageSharedPtr image = HioImage::OpenForWriting(png.string());
        REQUIRE(image);
        HioImage::StorageSpec spec;
        spec.width = 16;
        spec.height = 16;
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(image->Write(spec));
    }
    const std::array<float, 3> textureColour{0xCC / 255.0F, 0x99 / 255.0F, 0x33 / 255.0F};
    SECTION("MaterialX: oren_nayar coloured by an image") {
        const fs::path path = scratch("material_mtlx.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
                   "        def Shader \"Surface\"\n        {\n"
                   "            uniform token info:id = \"ND_surface\"\n"
                   "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
                   "            token outputs:out\n        }\n"
                   "        def Shader \"Diffuse\"\n        {\n"
                   "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
                   "            color3f inputs:color.connect = </Materials/Mat/Texture.outputs:out>\n"
                   "            token outputs:out\n        }\n"
                   "        def Shader \"Texture\"\n        {\n"
                   "            uniform token info:id = \"ND_image_color3\"\n"
                   "            asset inputs:file = @" << png.string() << "@ ( colorSpace = \"lin_rec709\" )\n"
                   "            color3f outputs:out\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const auto c = squareMismatches(*gpu, *image, textureColour);
        // The same material seen by rays instead of the rasteriser: shading
        // reads the visibility buffer, whichever route filled it.
        if (gpu->device->caps().rayQuery && gpu->device->caps().accelerationStructure) {
            REQUIRE((*renderer)->setMeshVisibility("rays"));
            auto traced = (*renderer)->render("/Camera", 0.0, 160, 120);
            REQUIRE((*renderer)->setMeshVisibility("raster"));
            if (!traced) FAIL(traced.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto a = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            auto b = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
            REQUIRE(a);
            REQUIRE(b);
            auto diff = render::compareImages(*gpu->library, *a, *b, 160, 120);
            REQUIRE(diff);
            std::printf("  raster against rays, textured material: p99 %u, max %u, %llu pixels beyond 2\n", diff->p99,
                        diff->max, static_cast<unsigned long long>(diff->over2));
            CHECK(diff->max == 0);
        }
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  MaterialX image material: %u covered, %u coverage and %u colour mismatches; centre %.4f %.4f "
                    "%.4f (texture %.4f %.4f %.4f)\n",
                    c[2], c[0], c[1], double(centre[0]), double(centre[1]), double(centre[2]),
                    double(textureColour[0]), double(textureColour[1]), double(textureColour[2]));
        CHECK(c[2] > 800);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
    }
    SECTION("UsdPreviewSurface with a UsdUVTexture read through UsdPrimvarReader") {
        const fs::path path = scratch("material_preview_texture.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Texture.outputs:rgb>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Texture\"\n        {\n"
                   "            uniform token info:id = \"UsdUVTexture\"\n"
                   "            asset inputs:file = @" << png.string() << "@\n"
                   "            token inputs:sourceColorSpace = \"raw\"\n"
                   "            float2 inputs:st.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            color3f outputs:rgb\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                   "            string inputs:varname = \"st\"\n"
                   "            float2 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  UsdPreviewSurface textured centre: %.4f %.4f %.4f (texture %.4f %.4f %.4f)\n",
                    double(centre[0]), double(centre[1]), double(centre[2]), double(textureColour[0]),
                    double(textureColour[1]), double(textureColour[2]));
        for (int k = 0; k < 3; ++k) {
            CHECK(centre[k] == Catch::Approx(textureColour[size_t(k)]).epsilon(0.01));
        }
    }
    SECTION("a texture authored relative to its layer, read from another directory, wrapping as USD spells it") {
        // The layer in a directory of its own, the image beside it under
        // textures/: "./textures/orange.png" names it only relative to the
        // layer, never to the process's working directory.
        const fs::path directory = scratch("relative_layer");
        fs::create_directories(directory / "textures");
        fs::copy_file(png, directory / "textures" / "orange.png", fs::copy_options::overwrite_existing);
        const fs::path path = directory / "material_relative.usda";
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Texture.outputs:rgb>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Texture\"\n        {\n"
                   "            uniform token info:id = \"UsdUVTexture\"\n"
                   "            asset inputs:file = @./textures/orange.png@\n"
                   "            token inputs:wrapS = \"repeat\"\n"
                   "            token inputs:wrapT = \"repeat\"\n"
                   "            token inputs:sourceColorSpace = \"raw\"\n"
                   "            float2 inputs:st.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            color3f outputs:rgb\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                   "            string inputs:varname = \"st\"\n"
                   "            float2 outputs:result\n        }\n    }\n}\n";
        }
        REQUIRE(fs::current_path() != directory);
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  relative texture centre: %.4f %.4f %.4f (texture %.4f %.4f %.4f)\n", double(centre[0]),
                    double(centre[1]), double(centre[2]), double(textureColour[0]), double(textureColour[1]),
                    double(textureColour[2]));
        for (int k = 0; k < 3; ++k) {
            CHECK(centre[k] == Catch::Approx(textureColour[size_t(k)]).epsilon(0.01));
        }
    }
    SECTION("UsdPreviewSurface's diffuseColor from a colour primvar through a float3 reader") {
        // The reader is a vector to MaterialX and diffuseColor a colour: the
        // material failed its declaration and the mesh fell back to its
        // displayColor, (0.1, 0.9, 0.1). The primvar read here is another
        // colour, so the fallback cannot pass for the material.
        const fs::path path = scratch("material_preview_reader3.usda");
        {
            std::ofstream out(path);
            std::string square = kSquareStage;
            const std::string binding = "    rel material:binding";
            square.insert(square.find(binding),
                          "    color3f[] primvars:tint = [(0.7, 0.3, 0.5)] (\n        interpolation = \"constant\"\n    )\n");
            out << square
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float3\"\n"
                   "            string inputs:varname = \"tint\"\n"
                   "            float3 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  a colour primvar through a float3 reader: centre %.4f %.4f %.4f (primvar 0.7 0.3 0.5)\n",
                    double(centre[0]), double(centre[1]), double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.7F).margin(0.07F));
        CHECK(centre[1] == Catch::Approx(0.3F).margin(0.03F));
        CHECK(centre[2] == Catch::Approx(0.5F).margin(0.05F));
    }
    SECTION("a primvar authored on the material reaches the mesh bound to it") {
        // The blend shape test stage's material carries primvars:displayColor
        // itself and reads it back: the mesh has none of its own. Transferred
        // by hdsi, the square is the material's colour; untransferred, the
        // reader reads its default, black.
        const fs::path path = scratch("material_primvar_transfer.usda");
        {
            std::string square = kSquareStage;
            const std::string colour = "    color3f[] primvars:displayColor = [(0.1, 0.9, 0.1)] (\n"
                                       "        interpolation = \"constant\"\n    )\n";
            square.erase(square.find(colour), colour.size());
            std::ofstream out(path);
            out << square
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        color3f primvars:tint = (0.3, 0.6, 0.9)\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float3\"\n"
                   "            string inputs:varname = \"tint\"\n"
                   "            float3 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  a material's own primvar: centre %.4f %.4f %.4f (primvar 0.3 0.6 0.9)\n", double(centre[0]),
                    double(centre[1]), double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.3F).margin(0.03F));
        CHECK(centre[1] == Catch::Approx(0.6F).margin(0.06F));
        CHECK(centre[2] == Catch::Approx(0.9F).margin(0.09F));
    }
    SECTION("a normal map authored as USD writes it: float4 scale and bias, a colour into a vector normal") {
        // UsdUVTexture's scale and bias are float4 in USD and color4 in
        // MaterialX; UsdPreviewSurface's normal is a vector fed from the
        // texture's rgb, a colour. As authored (McUsd writes every material
        // so) the whole material failed and fell back to displayColor.
        const fs::path flat = scratch("flat_normal.png");
        {
            fs::remove(flat);
            std::vector<uint32_t> texels(16 * 16, 0xFFFF8080u);   // ABGR: (128, 128, 255), the unperturbed normal
            HioImageSharedPtr image = HioImage::OpenForWriting(flat.string());
            REQUIRE(image);
            HioImage::StorageSpec spec;
            spec.width = 16;
            spec.height = 16;
            spec.depth = 1;
            spec.format = HioFormatUNorm8Vec4;
            spec.data = texels.data();
            REQUIRE(image->Write(spec));
        }
        const fs::path path = scratch("material_preview_normalmap.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor = (0.8, 0.4, 0.2)\n"
                   "            float3 inputs:normal.connect = </Materials/Mat/Normal.outputs:rgb>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Normal\"\n        {\n"
                   "            uniform token info:id = \"UsdUVTexture\"\n"
                   "            asset inputs:file = @" << flat.string() << "@\n"
                   "            float4 inputs:bias = (-1, -1, -1, -1)\n"
                   "            float4 inputs:scale = (2, 2, 2, 2)\n"
                   "            token inputs:sourceColorSpace = \"raw\"\n"
                   "            float2 inputs:st.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            float3 outputs:rgb\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                   "            string inputs:varname = \"st\"\n"
                   "            float2 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  USD-typed normal map: centre %.4f %.4f %.4f (diffuse 0.8 0.4 0.2; the fallback is 0.1 0.9 0.1)\n",
                    double(centre[0]), double(centre[1]), double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.8F).margin(0.08F));
        CHECK(centre[1] == Catch::Approx(0.4F).margin(0.04F));
        CHECK(centre[2] == Catch::Approx(0.2F).margin(0.02F));
    }
    SECTION("UsdPreviewSurface, rough, specular workflow without specular") {
        const fs::path path = scratch("material_preview.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor = (0.8, 0.4, 0.2)\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        // Not the displayColor fallback: the material's colour, less what the
        // (black) specular layer's albedo fit keeps of it.
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  UsdPreviewSurface centre: %.4f %.4f %.4f\n", double(centre[0]), double(centre[1]),
                    double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.8F).margin(0.08F));
        CHECK(centre[1] == Catch::Approx(0.4F).margin(0.04F));
        CHECK(centre[2] == Catch::Approx(0.2F).margin(0.02F));
    }
}

namespace {

/// Two squares, one in front of the other, each with its own material: the
/// front one's is a UsdPreviewSurface whose opacity `opacity` is cut at
/// `threshold`, the back one's a MaterialX diffuse of a known colour. With
/// `alpha` the opacity is that texture's alpha channel instead.
std::string cutoutStage(const std::string& opacity, float threshold, const std::array<float, 3>& back,
                        const std::string& extra = {}) {
    std::ostringstream out;
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Front\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-1, -1, -3), (1, -1, -3), (1, 1, -3), (-1, 1, -3)]\n"
           "    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (\n        interpolation = \"vertex\"\n    )\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    rel material:binding = </Materials/Cut>\n}\n"
           "def Mesh \"Back\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    rel material:binding = </Materials/Back>\n}\n"
           "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 1000)\n}\n"
           "def Scope \"Materials\"\n{\n"
           "    def Material \"Cut\"\n    {\n"
           "        token outputs:surface.connect = </Materials/Cut/Preview.outputs:surface>\n"
           "        def Shader \"Preview\"\n        {\n"
           "            uniform token info:id = \"UsdPreviewSurface\"\n"
           "            color3f inputs:diffuseColor = (0.2, 0.4, 0.8)\n"
           "            int inputs:useSpecularWorkflow = 1\n"
           "            color3f inputs:specularColor = (0, 0, 0)\n"
           "            float inputs:roughness = 1\n"
        << "            " << opacity << "\n"
        << "            float inputs:opacityThreshold = " << threshold << "\n"
           "            token outputs:surface\n        }\n"
        << extra
        << "    }\n"
           "    def Material \"Back\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Back/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_surface\"\n"
           "            token inputs:bsdf.connect = </Materials/Back/Diffuse.outputs:out>\n"
           "            token outputs:out\n        }\n"
           "        def Shader \"Diffuse\"\n        {\n"
           "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
        << "            color3f inputs:color = (" << back[0] << ", " << back[1] << ", " << back[2] << ")\n"
        << "            token outputs:out\n        }\n    }\n}\n";
    return out.str();
}

}   // namespace

TEST_CASE("a material's opacityThreshold cuts its samples out of visibility itself, in every route",
          "[usd][gpu][mesh][materials][cutout]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    const bool rays = caps.rayQuery && caps.accelerationStructure;
    std::vector<const char*> routes{"raster", "bvh"};
    if (rays) {
        routes.insert(routes.begin() + 1, "rays");
    }
    const std::array<float, 3> backColour{0.8F, 0.4F, 0.2F};
    const uint32_t w = 160;
    const uint32_t h = 120;

    SECTION("cut away, what is behind shows exactly as if it were alone") {
        const fs::path path = scratch("cutout_all.usda");
        {
            std::ofstream out(path);
            out << cutoutStage("float inputs:opacity = 0.2", 0.5F, backColour);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        for (const char* route : routes) {
            REQUIRE((*renderer)->setMeshVisibility(route));
            auto image = (*renderer)->render("/Camera", 0.0, w, h);
            if (!image) FAIL(image.error().toString());
            // The back square at z = -5, analytically: the front one is gone,
            // and nothing of it is left in the depth either.
            const auto c = squareMismatches(*gpu, *image, backColour);
            std::printf("  cutout by %s: %u covered, %u coverage and %u colour mismatches against the square behind\n",
                        route, c[2], c[0], c[1]);
            CHECK(c[2] > 800);
            CHECK(c[0] == 0);
            CHECK(c[1] == 0);
        }
    }

    SECTION("an opacity above the threshold is not cut") {
        const fs::path path = scratch("cutout_none.usda");
        {
            std::ofstream out(path);
            out << cutoutStage("float inputs:opacity = 0.2", 0.1F, backColour);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        for (const char* route : routes) {
            REQUIRE((*renderer)->setMeshVisibility(route));
            auto image = (*renderer)->render("/Camera", 0.0, w, h);
            if (!image) FAIL(image.error().toString());
            // The front square at z = -3 covers the back one: its coverage and
            // depth, not the colour, which is the preview surface's.
            const auto c = squareMismatches(*gpu, *image, {0.2F, 0.4F, 0.8F}, 3.0F);
            std::printf("  kept by %s: %u covered, %u coverage mismatches against the square in front (%u colour)\n",
                        route, c[2], c[0], c[1]);
            CHECK(c[2] > 2000);
            CHECK(c[0] == 0);
        }
    }

    SECTION("half a square cut by a texture's alpha, the same in every route") {
        // Alpha 0 on the left half of the texture, 1 on the right.
        const fs::path png = scratch("cutout_alpha.png");
        {
            fs::remove(png);
            std::vector<uint32_t> texels(size_t{16} * 16, 0u);
            for (size_t y = 0; y < 16; ++y) {
                for (size_t x = 0; x < 16; ++x) {
                    texels[y * 16 + x] = x < 8 ? 0x00FFFFFFu : 0xFFFFFFFFu;   // ABGR
                }
            }
            HioImageSharedPtr made = HioImage::OpenForWriting(png.string());
            REQUIRE(made);
            HioImage::StorageSpec spec;
            spec.width = 16;
            spec.height = 16;
            spec.depth = 1;
            spec.format = HioFormatUNorm8Vec4;
            spec.data = texels.data();
            REQUIRE(made->Write(spec));
        }
        const fs::path path = scratch("cutout_texture.usda");
        {
            const std::string alpha =
                "        def Shader \"Alpha\"\n        {\n"
                "            uniform token info:id = \"UsdUVTexture\"\n"
                "            asset inputs:file = @" + png.string() + "@\n"
                "            token inputs:sourceColorSpace = \"raw\"\n"
                "            float2 inputs:st.connect = </Materials/Cut/Reader.outputs:result>\n"
                "            float outputs:a\n        }\n"
                "        def Shader \"Reader\"\n        {\n"
                "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                "            string inputs:varname = \"st\"\n"
                "            float2 outputs:result\n        }\n";
            std::ofstream out(path);
            out << cutoutStage("float inputs:opacity.connect = </Materials/Cut/Alpha.outputs:a>", 0.5F, backColour,
                               alpha);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        std::vector<gpu::Buffer> images;
        for (const char* route : routes) {
            REQUIRE((*renderer)->setMeshVisibility(route));
            auto image = (*renderer)->render("/Camera", 0.0, w, h);
            if (!image) FAIL(image.error().toString());
            // Left of the square: the back one shows through the hole. Right:
            // the front one, nearer, is kept.
            const float* left = image->rgba.data() + (size_t{h} / 2 * w + w / 2 - 20) * 4;
            const float* right = image->rgba.data() + (size_t{h} / 2 * w + w / 2 + 20) * 4;
            std::printf("  half cut by %s: left %.3f %.3f %.3f, right %.3f %.3f %.3f\n", route, double(left[0]),
                        double(left[1]), double(left[2]), double(right[0]), double(right[1]), double(right[2]));
            CHECK(left[0] > left[2]);    // the back square's orange
            CHECK(right[2] > right[0]);  // the front square's blue
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(buffer);
            images.push_back(std::move(*buffer));
        }
        for (size_t k = 1; k < images.size(); ++k) {
            auto diff = render::compareImages(*gpu->library, images[0], images[k], w, h);
            REQUIRE(diff);
            std::printf("  raster against %s: p99 %u, max %u, %llu pixels beyond 2\n", routes[k], diff->p99, diff->max,
                        static_cast<unsigned long long>(diff->over2));
            // The cut is decided at the pixel's centre by the same material in
            // every route, as the silhouettes are: a few edge pixels apart.
            CHECK(diff->over2 * 100 <= uint64_t{w} * h);
        }
    }
}

TEST_CASE("a GeomSubset's material shades its faces; the mesh's the rest", "[usd][gpu][mesh][materials]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto diffuse = [](const std::string& name, const std::string& colour) {
        return "    def Material \"" + name + "\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/" + name + "/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/" + name + "/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (" + colour + ")\n"
               "            token outputs:out\n        }\n    }\n";
    };
    const fs::path path = scratch("subsets.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Faces\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4, 4]\n"
               "    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]\n"
               "    point3f[] points = [(-2, -1, -5), (0, -1, -5), (2, -1, -5), (-2, 1, -5), (0, 1, -5), (2, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    uniform token subsetFamily:materialBind:familyType = \"nonOverlapping\"\n"
               "    rel material:binding = </Materials/Green>\n"
               "    def GeomSubset \"Left\" (\n        prepend apiSchemas = [\"MaterialBindingAPI\"]\n    )\n    {\n"
               "        uniform token elementType = \"face\"\n"
               "        uniform token familyName = \"materialBind\"\n"
               "        int[] indices = [0]\n"
               "        rel material:binding = </Materials/Red>\n    }\n}\n"
               "def Scope \"Materials\"\n{\n" << diffuse("Red", "0.9, 0.1, 0.1") << diffuse("Green", "0.1, 0.9, 0.1")
            << "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 20\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    const auto pixel = [&](uint32_t x, uint32_t y) {
        const float* p = image->rgba.data() + (size_t{y} * w + x) * 4;
        return std::array<float, 3>{p[0], p[1], p[2]};
    };
    const auto left = pixel(w / 2 - 20, h / 2);
    const auto right = pixel(w / 2 + 20, h / 2);
    std::printf("  subset face: %.3f %.3f %.3f; the rest: %.3f %.3f %.3f\n", double(left[0]), double(left[1]),
                double(left[2]), double(right[0]), double(right[1]), double(right[2]));
    CHECK(left[0] > 0.8F);
    CHECK(left[1] < 0.15F);
    CHECK(right[1] > 0.8F);
    CHECK(right[0] < 0.15F);
}

TEST_CASE("displayColor reaches the pixels per face and per indexed face-vertex", "[usd][gpu][mesh][primvars]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("primvars.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               // Two faces, one colour each.
               "def Mesh \"PerFace\"\n{\n"
               "    int[] faceVertexCounts = [4, 4]\n"
               "    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]\n"
               "    point3f[] points = [(-2, 0.1, -5), (0, 0.1, -5), (2, 0.1, -5), (-2, 1.5, -5), (0, 1.5, -5), (2, 1.5, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(1, 0, 0), (0, 1, 0)] ( interpolation = \"uniform\" )\n"
               "}\n"
               // One face, red on its left corners and blue on its right, indexed.
               "def Mesh \"PerCorner\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -1.5, -5), (2, -1.5, -5), (2, -0.1, -5), (-2, -0.1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(1, 0, 0), (0, 0, 1)] ( interpolation = \"faceVarying\" )\n"
               "    int[] primvars:displayColor:indices = [0, 1, 1, 0]\n"
               "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 20\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    // Pixels well inside each region (bottom row first): what they read.
    const auto pixel = [&](uint32_t x, uint32_t y) {
        const float* p = image->rgba.data() + (size_t{y} * w + x) * 4;
        return std::array<float, 3>{p[0], p[1], p[2]};
    };
    const auto left = pixel(w / 2 - 20, h / 2 + 15);
    const auto right = pixel(w / 2 + 20, h / 2 + 15);
    const auto cornerLeft = pixel(w / 2 - 45, h / 2 - 15);
    const auto cornerMiddle = pixel(w / 2, h / 2 - 15);
    const auto cornerRight = pixel(w / 2 + 45, h / 2 - 15);
    std::printf("  per face: (%.2f %.2f %.2f) | (%.2f %.2f %.2f); per corner: r/b %.2f/%.2f, %.2f/%.2f, %.2f/%.2f\n",
                double(left[0]), double(left[1]), double(left[2]), double(right[0]), double(right[1]),
                double(right[2]), double(cornerLeft[0]), double(cornerLeft[2]), double(cornerMiddle[0]),
                double(cornerMiddle[2]), double(cornerRight[0]), double(cornerRight[2]));
    CHECK(left[0] > 0.5F);
    CHECK(left[1] == 0.0F);
    CHECK(right[1] > 0.5F);
    CHECK(right[0] == 0.0F);
    CHECK(cornerLeft[0] > cornerLeft[2]);
    CHECK(cornerRight[2] > cornerRight[0]);
    CHECK(std::abs(cornerMiddle[0] - cornerMiddle[2]) < 0.1F);
    CHECK(cornerMiddle[1] == 0.0F);

    // The same frame's ids, normals and a primvar, as Hydra's outputs.
    (*renderer)->requestOutputs({"primId", "elementId", "Neye", "normal", "primvars:displayColor"});
    REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
    auto primId = (*renderer)->mappedOutput("primId");
    auto elementId = (*renderer)->mappedOutput("elementId");
    auto eye = (*renderer)->mappedOutput("Neye");
    auto world = (*renderer)->mappedOutput("normal");
    auto colour = (*renderer)->mappedOutput("primvars:displayColor");
    REQUIRE(primId);
    REQUIRE(elementId);
    REQUIRE(eye);
    REQUIRE(world);
    REQUIRE(colour);
    // Hydra's buffers are bottom row first, as the engine's images.
    const auto at = [&](const std::vector<uint8_t>& bytes, uint32_t x, uint32_t y, uint32_t channels, uint32_t c) {
        int32_t v = 0;
        std::memcpy(&v, bytes.data() + (size_t{y} * w + x) * channels * 4 + c * 4, 4);
        return v;
    };
    const auto atFloat = [&](const std::vector<uint8_t>& bytes, uint32_t x, uint32_t y, uint32_t channels,
                             uint32_t c) {
        float v = 0.0F;
        std::memcpy(&v, bytes.data() + (size_t{y} * w + x) * channels * 4 + c * 4, 4);
        return v;
    };
    const uint32_t lx = w / 2 - 20, rx = w / 2 + 20, fy = h / 2 + 15, cy = h / 2 - 15;
    std::printf("  primId %d %d %d, background %d; elementId %d %d; Neye z %.3f; normal z %.3f; displayColor r %.3f\n",
                at(*primId, lx, fy, 1, 0), at(*primId, rx, fy, 1, 0), at(*primId, w / 2, cy, 1, 0),
                at(*primId, 2, 2, 1, 0), at(*elementId, lx, fy, 1, 0), at(*elementId, rx, fy, 1, 0),
                double(atFloat(*eye, lx, fy, 3, 2)), double(atFloat(*world, lx, fy, 3, 2)),
                double(atFloat(*colour, lx, fy, 3, 0)));
    CHECK(at(*primId, lx, fy, 1, 0) == at(*primId, rx, fy, 1, 0));
    CHECK(at(*primId, lx, fy, 1, 0) != at(*primId, w / 2, cy, 1, 0));
    CHECK(at(*primId, lx, fy, 1, 0) >= 0);
    CHECK(at(*primId, 2, 2, 1, 0) == -1);
    CHECK(at(*elementId, lx, fy, 1, 0) == 0);
    CHECK(at(*elementId, rx, fy, 1, 0) == 1);
    CHECK(atFloat(*eye, lx, fy, 3, 2) == Catch::Approx(1.0F).margin(1e-4F));
    CHECK(atFloat(*world, lx, fy, 3, 2) == Catch::Approx(1.0F).margin(1e-4F));
    CHECK(atFloat(*colour, lx, fy, 3, 0) == Catch::Approx(1.0F).margin(1e-5F));
    CHECK(atFloat(*colour, lx, fy, 3, 1) == Catch::Approx(0.0F).margin(1e-5F));

    // What a viewer asks of the same frame: the prim under a pixel (from the
    // top left) and where the stage is.
    auto overFace = (*renderer)->pick(lx, h - 1 - fy);
    auto overCorner = (*renderer)->pick(w / 2, h - 1 - cy);
    auto overNothing = (*renderer)->pick(2, 2);
    REQUIRE(overFace);
    REQUIRE(overCorner);
    REQUIRE(overNothing);
    REQUIRE(overFace->has_value());
    REQUIRE(overCorner->has_value());
    CHECK((*overFace)->prim == "/PerFace");
    CHECK((*overCorner)->prim == "/PerCorner");
    CHECK(!overNothing->has_value());
    auto box = (*renderer)->bounds();
    REQUIRE(box);
    REQUIRE(box->has_value());
    std::printf("  picked %s and %s; bounds (%.2f %.2f %.2f)-(%.2f %.2f %.2f)\n", (*overFace)->prim.c_str(),
                (*overCorner)->prim.c_str(), double((*box)->min[0]), double((*box)->min[1]), double((*box)->min[2]),
                double((*box)->max[0]), double((*box)->max[1]), double((*box)->max[2]));
    CHECK((*box)->min[0] == Catch::Approx(-2.0F).margin(1e-5F));
    CHECK((*box)->min[1] == Catch::Approx(-1.5F).margin(1e-5F));
    CHECK((*box)->max[1] == Catch::Approx(1.5F).margin(1e-5F));
    CHECK((*box)->max[2] == Catch::Approx(-5.0F).margin(1e-5F));
}

TEST_CASE("a PointInstancer draws as its instances authored one by one", "[usd][gpu][mesh][instancing]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const std::string square =
        "        int[] faceVertexCounts = [4]\n"
        "        int[] faceVertexIndices = [0, 1, 2, 3]\n"
        "        point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]\n"
        "        uniform token subdivisionScheme = \"none\"\n"
        "        color3f[] primvars:displayColor = [(0.3, 0.7, 0.5)] ( interpolation = \"constant\" )\n";
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    double3 xformOp:translate = (0.5, 1, 9)\n"
                               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    const fs::path instanced = scratch("instancer.usda");
    {
        std::ofstream out(instanced);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def PointInstancer \"Many\"\n{\n"
               "    rel prototypes = [</Many/Prototypes/Square>]\n"
               "    int[] protoIndices = [0, 0, 0]\n"
               "    point3f[] positions = [(-2, 0, 0), (0, 1, -1), (2, -0.5, 0.5)]\n"
               "    quath[] orientations = [(1, 0, 0, 0), (0.7071068, 0, 0.7071068, 0), (0.9238795, 0.3826834, 0, 0)]\n"
               "    float3[] scales = [(1, 1, 1), (2, 1, 1), (1, 1.5, 1)]\n"
               "    double3 xformOp:rotateXYZ = (0, 10, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n"
               "    def Scope \"Prototypes\"\n    {\n"
               "        def Mesh \"Square\"\n        {\n" << square << "        }\n    }\n}\n" << camera;
    }
    const fs::path authored = scratch("authored.usda");
    {
        std::ofstream out(authored);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Xform \"Many\"\n{\n"
               "    double3 xformOp:rotateXYZ = (0, 10, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n";
        // translate, orient (as a rotation about one axis), scale: the
        // instancer's T * R * S, authored as ops.
        const char* ops[3] = {
            "        double3 xformOp:translate = (-2, 0, 0)\n"
            "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n",
            "        double3 xformOp:translate = (0, 1, -1)\n        double xformOp:rotateY = 90\n"
            "        float3 xformOp:scale = (2, 1, 1)\n"
            "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateY\", \"xformOp:scale\"]\n",
            "        double3 xformOp:translate = (2, -0.5, 0.5)\n        double xformOp:rotateX = 45\n"
            "        float3 xformOp:scale = (1, 1.5, 1)\n"
            "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\", \"xformOp:scale\"]\n"};
        for (int k = 0; k < 3; ++k) {
            out << "    def Mesh \"Square" << k << "\"\n    {\n" << ops[k] << square << "    }\n";
        }
        out << "}\n" << camera;
    }
    const uint32_t w = 200;
    const uint32_t h = 150;
    auto a = usd::StageRenderer::open(instanced);
    auto b = usd::StageRenderer::open(authored);
    if (!a) FAIL(a.error().toString());
    if (!b) FAIL(b.error().toString());
    auto imageA = (*a)->render("/Camera", 0.0, w, h);
    auto imageB = (*b)->render("/Camera", 0.0, w, h);
    if (!imageA) FAIL(imageA.error().toString());
    if (!imageB) FAIL(imageB.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = imageA->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto bufferA = gpu::Buffer::create(*gpu->device, desc, imageA->rgba.data());
    auto bufferB = gpu::Buffer::create(*gpu->device, desc, imageB->rgba.data());
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    auto diff = render::compareHdr(*gpu->library, *bufferA, *bufferB, w, h);
    REQUIRE(diff);
    auto blank = std::vector<float>(imageA->rgba.size(), 0.0F);
    auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
    REQUIRE(blankBuffer);
    auto drawn = render::compareHdr(*gpu->library, *bufferA, *blankBuffer, w, h);
    REQUIRE(drawn);
    std::printf("  PointInstancer against authored xforms: relMSE %.2e, p99 relative %.2e (against blank: relMSE %.2e)\n",
                diff->relMse, diff->p99Relative, drawn->relMse);
    CHECK(drawn->relMse > 1.0);    // something was drawn
    CHECK(diff->relMse < 1e-5);    // half-precision orientations
}

TEST_CASE("Hydra gets the same ids whichever route finds the meshes", "[usd][gpu][mesh][visibility]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries, to compare all three routes");
    }
    const fs::path path = scratch("routes.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               // Instanced squares, turned and scaled; a single-sided floor
               // facing the camera and a wall behind it facing away; a
               // mirrored single-sided square.
               "def PointInstancer \"Many\"\n{\n"
               "    rel prototypes = [</Many/Prototypes/Square>]\n"
               "    int[] protoIndices = [0, 0, 0, 0]\n"
               "    point3f[] positions = [(-2, 0, 0), (0, 1, -1), (2, -0.5, 0.5), (0.5, -1.2, 1)]\n"
               "    quath[] orientations = [(1, 0, 0, 0), (0.7071068, 0, 0.7071068, 0), (0.9238795, 0.3826834, 0, 0),"
               " (0.9659258, 0, 0, 0.258819)]\n"
               "    float3[] scales = [(1, 1, 1), (2, 1, 1), (1, 1.5, 1), (0.7, 0.7, 0.7)]\n"
               "    def Scope \"Prototypes\"\n    {\n"
               "        def Mesh \"Square\"\n        {\n"
               "            int[] faceVertexCounts = [4]\n"
               "            int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "            point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]\n"
               "            uniform token subdivisionScheme = \"none\"\n"
               "            uniform bool doubleSided = 1\n"
               "        }\n    }\n}\n"
               "def Mesh \"Ground\"\n{\n"
               "    int[] faceVertexCounts = [4, 4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3, 3, 5, 4, 2]\n"
               "    point3f[] points = [(-4, -2, 3), (4, -2, 3), (4, -2, -3), (-4, -2, -3), (4, 1, -5), (-4, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "}\n"
               "def Mesh \"Mirrored\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    double3 xformOp:translate = (-1.5, 1.5, 1)\n    float3 xformOp:scale = (-1, 1, 1)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n"
               "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    double3 xformOp:translate = (0.5, 0.5, 9)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->requestOutputs({"primId", "instanceId", "elementId"});
    const uint32_t w = 320;
    const uint32_t h = 240;
    struct Frame {
        gpu::Buffer ids;   // primId, instanceId, elementId: three words a pixel
        uint64_t    covered = 0;
    };
    const auto frame = [&](const char* route) {
        REQUIRE((*renderer)->setMeshVisibility(route));
        REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
        std::vector<uint8_t> bytes;
        for (const char* aov : {"primId", "instanceId", "elementId"}) {
            auto mapped = (*renderer)->mappedOutput(aov);
            REQUIRE(mapped);
            REQUIRE(mapped->size() == size_t{w} * h * 4);
            bytes.insert(bytes.end(), mapped->begin(), mapped->end());
        }
        gpu::BufferDesc desc;
        desc.bytes = bytes.size();
        desc.elementBytes = 4;
        auto ids = gpu::Buffer::create(*gpu->device, desc, bytes.data());
        REQUIRE(ids);
        // Coverage: the primId plane against a cleared one (-1 everywhere).
        const std::vector<int32_t> cleared(size_t{w} * h, -1);
        gpu::BufferDesc plane;
        plane.bytes = cleared.size() * 4;
        plane.elementBytes = 4;
        auto blank = gpu::Buffer::create(*gpu->device, plane, cleared.data());
        auto prims = gpu::Buffer::create(*gpu->device, plane, bytes.data());
        REQUIRE(blank);
        REQUIRE(prims);
        auto covered = render::countDifferent(*gpu->library, *prims, *blank, w * h);
        REQUIRE(covered);
        return Frame{std::move(*ids), *covered};
    };
    const Frame raster = frame("raster");
    const Frame rays = frame("rays");
    const Frame bvh = frame("bvh");
    const Frame automatic = frame("automatic");
    auto raysDiffer = render::countDifferent(*gpu->library, raster.ids, rays.ids, 3 * w * h);
    auto bvhDiffer = render::countDifferent(*gpu->library, raster.ids, bvh.ids, 3 * w * h);
    auto automaticDiffer = render::countDifferent(*gpu->library, rays.ids, automatic.ids, 3 * w * h);
    REQUIRE(raysDiffer);
    REQUIRE(bvhDiffer);
    REQUIRE(automaticDiffer);
    std::printf("  ids through Hydra, %u x %u, raster covering %llu: rays differ in %llu words, the BVH in %llu, "
                "automatic in %llu\n",
                w, h, static_cast<unsigned long long>(raster.covered), static_cast<unsigned long long>(*raysDiffer),
                static_cast<unsigned long long>(*bvhDiffer), static_cast<unsigned long long>(*automaticDiffer));
    CHECK(raster.covered > uint64_t{w} * h / 8);
    CHECK(*automaticDiffer == 0);   // this device has ray queries: automatic is rays
    // Raster samples pixel centres as the rays do, so they part only where a
    // centre falls on an edge -- here along the grazing floor: measured 15
    // and 6 words of 3 x 76800.
    CHECK(*raysDiffer * 1000 <= raster.covered);
    CHECK(*bvhDiffer * 1000 <= raster.covered);
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
    const UsdPrimDefinition* lighting = registry.FindAppliedAPIPrimDefinition(TfToken("LrtSplatLightingAPI"));
    REQUIRE(lighting != nullptr);
    // Baked is the default: a capture shows the light it was captured under
    // until something asks otherwise.
    VtValue relight;
    CHECK(lighting->GetAttributeFallbackValue(TfToken("primvars:lrt:splat:relight"), &relight));
    CHECK(relight.IsHolding<bool>());
    CHECK(!relight.UncheckedGet<bool>());

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

TEST_CASE("a UsdLux light through Hydra lights a Lambert plane as the closed form says",
          "[usd][gpu][mesh][lights]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("light_sphere.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\" (\n    prepend apiSchemas = [\"ShadowAPI\"]\n)\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    // The same stage with no light at all: the headlight, which the analytic
    // square checks exactly. If this is 0 and 0, the material is Lambert of
    // albedo 0.8 and whatever the lit image does is the light's doing.
    {
        const fs::path dark = scratch("light_none.usda");
        {
            std::ifstream in(path);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            const size_t at = text.find("def SphereLight");
            const size_t end = text.find("def Scope \"Materials\"");
            REQUIRE(at != std::string::npos);
            REQUIRE(end != std::string::npos);
            text.erase(at, end - at);
            std::ofstream out(dark);
            out << text;
        }
        auto plain = usd::StageRenderer::open(dark);
        if (!plain) FAIL(plain.error().toString());
        auto lit = (*plain)->render("/Camera", 0.0, w, h);
        if (!lit) FAIL(lit.error().toString());
        const auto c = squareMismatches(*gpu, *lit, {0.8F, 0.8F, 0.8F});
        std::printf("  the same plane by the headlight: %u covered, %u coverage and %u colour mismatches\n", c[2],
                    c[0], c[1]);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    // The closed form is what the estimator converges to: ask for enough
    // samples that what is left is the light, not the noise.
    (*renderer)->setLightSamples(4096);
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(colours->rhi());
            cursor["depth"].setBinding(depth->rhi());
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
            const float radiance[4] = {1.0F, 1.0F, 1.0F, 0.02F};
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
    float probe[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(probe), probe));
    const float relative = probe[0];
    // The centre of the plane sees the sphere on its axis: d = 2, r = 0.4,
    // so E = pi r^2 / d^2 and a Lambert surface returns albedo E / pi.
    const float* centre = image->rgba.data() + (size_t{h} / 2 * w + w / 2) * 4;
    const float* offAxis = image->rgba.data() + (size_t{h} / 2 * w + w / 2 + 20) * 4;
    const float expected = 0.8F * 0.4F * 0.4F / (2.0F * 2.0F);
    std::printf("  UsdLuxSphereLight: %u pixels, %u beyond 2%%, worst %.4f; centre %.5f (closed form %.5f), "
                "20 px off axis %.5f against %.5f at (%.3f, %.3f, %.3f)\n",
                c[0], c[1], double(relative), double(centre[0]), double(expected), double(offAxis[0]),
                double(probe[1]), double(probe[2]), double(probe[3]), double(probe[4]));
    CHECK(c[0] > 800);
    CHECK(c[1] == 0);
}

TEST_CASE("a dome light's image lights a Lambert plane, and shows where nothing is drawn",
          "[usd][gpu][mesh][lights]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    // A lat-long image of one colour: whatever the mapping does, every
    // direction reads the same radiance, so the closed form still holds.
    const fs::path png = scratch("dome.png");
    {
        fs::remove(png);
        std::vector<uint32_t> texels(size_t{32} * 16, 0xFFCCCCCCu);   // ABGR, 0xCC per channel
        HioImageSharedPtr made = HioImage::OpenForWriting(png.string());
        REQUIRE(made);
        HioImage::StorageSpec spec;
        spec.width = 32;
        spec.height = 16;
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(made->Write(spec));
    }
    // Requested as auto, so an 8-bit image is read as sRGB.
    const float code = 0xCC / 255.0F;
    const float linear = std::pow((code + 0.055F) / 1.055F, 2.4F);
    const fs::path path = scratch("light_dome.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def DomeLight \"Sky\"\n{\n"
               "    float inputs:intensity = 1\n"
               "    color3f inputs:color = (1, 1, 1)\n"
            << "    asset inputs:texture:file = @" << png.string() << "@\n"
            << "}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 161;
    const uint32_t h = 121;
    (*renderer)->setLightSamples(std::getenv("LRT_DOME_SAMPLES") != nullptr ? 16384u : 1024u);
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(colours->rhi());
            cursor["depth"].setBinding(depth->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor p = cursor["plane"];
            p["kind"].setData(uint32_t{5});   // a dome
            p["vertices"].setData(uint32_t{512});
            p["row0"].setData(toWorld.data(), sizeof(float) * 4);
            p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            const float centre[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.0F};
            const float axisY[4] = {0.0F, 1.0F, 0.0F, 0.0F};
            const float normal[4] = {0.0F, 0.0F, 1.0F, 0.8F};
            const float radiance[4] = {linear, linear, linear, 0.02F};
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
    float probe[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(probe), probe));
    // A corner of the image sees no geometry: it shows the dome itself.
    const float* corner = image->rgba.data();
    std::printf("  dome image: %u pixels, %u beyond 2%%, worst %.4f; background %.4f (image %.4f)\n", c[0], c[1],
                double(probe[0]), double(corner[0]), double(linear));
    CHECK(c[0] > 800);
    CHECK(c[1] == 0);
    CHECK(corner[0] == Catch::Approx(linear).margin(0.01F));
}

/// The scene index's half of light linking, through Hydra. Hidden until
/// the chain's order was found: HdsiLightLinkingSceneIndex builds its
/// collection cache from the added-prim notices that pass through it, and
/// StageRenderer gave the stage to UsdImaging's scene indices before this
/// renderer's filters existed, so they never heard of the stage's prims and
/// every category came out empty. The stage is now given once the render
/// index observes the chain.
TEST_CASE("a UsdLux light's collection reaches only what it includes",
          "[usd][gpu][mesh][lights][linking]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("light_linked.usda");
    {
        std::ofstream out(path);
        // Two squares side by side, one material, and a light whose collection
        // includes the left one alone. USD resolves that into a category the
        // left square carries, which is what the engine tests.
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
        for (int k = 0; k < 2; ++k) {
            const float x = k == 0 ? -1.0F : 1.0F;
            out << "def Mesh \"" << (k == 0 ? "Left" : "Right") << "\" (\n"
                   "    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                << "    point3f[] points = [(" << x - 0.9F << ", -0.9, -5), (" << x + 0.9F << ", -0.9, -5), ("
                << x + 0.9F << ", 0.9, -5), (" << x - 0.9F << ", 0.9, -5)]\n"
                << "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Materials/Mat>\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def SphereLight \"Key\" (\n    prepend apiSchemas = [\"ShadowAPI\"]\n)\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    uniform token collection:lightLink:mode = \"expression\"\n"
               "    uniform pathExpression collection:lightLink:membershipExpression = \"/Left\"\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 160;
    const uint32_t h = 120;
    (*renderer)->setLightSamples(16);
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto counters = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_check", "lightLinkCounters");
    if (!counters) FAIL(counters.error().toString());
    gpu::Buffer halves = test::uintBuffer(*gpu->device, 4, "halves");
    {
        gpu::CommandBatch batch(*gpu->device);
        counters->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["image"].setBinding(colours->rhi());
            cursor["imageDepth"].setBinding(depth->rhi());
            cursor["halves"].setBinding(halves.rhi());
            cursor["params"]["thetaBins"].setData(h);
            cursor["params"]["phiBins"].setData(w);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t n[4] = {0, 0, 0, 0};
    REQUIRE(halves.read(*gpu->device, 0, sizeof(n), n));
    std::printf("  collection:lightLink includes /Left: left %u drawn %u lit, right %u drawn %u lit\n", n[0], n[1],
                n[2], n[3]);
    CHECK(n[0] > 1000);
    CHECK(n[1] == n[0]);   // the light reaches what its collection includes
    CHECK(n[2] > 1000);
    CHECK(n[3] == 0);      // and nothing else
}

// A stage without lights: the raster lights it with the headlight, and the
// path tracer drew it black. Both now light it alike -- the headlight at the
// first vertex, nothing after -- so an unlit asset looks the same whichever
// technique opens it, bounces or not.
TEST_CASE("a stage without lights draws the same under rt as under the raster's headlight",
          "[usd][gpu][mesh][path][headlight]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("unlit.usda");
    {
        std::ofstream out(path);
        out << kSquareStage;
    }
    const uint32_t w = 160, h = 120;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto raster = (*renderer)->render("/Camera", 0.0, w, h, "raster");
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathBounces(3);
    auto traced = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!raster) FAIL(raster.error().toString());
    if (!traced) FAIL(traced.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = raster->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto a = gpu::Buffer::create(*gpu->device, desc, raster->rgba.data());
    auto b = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
    REQUIRE(a);
    REQUIRE(b);
    auto diff = render::compareImages(*gpu->library, *a, *b, w, h);
    REQUIRE(diff);
    const uint64_t lit = [&] {
        gpu::BufferDesc zeros = desc;
        auto empty = gpu::Buffer::create(*gpu->device, zeros);
        REQUIRE(empty);
        auto words = render::countDifferent(*gpu->library, *b, *empty, w * h * 4);
        REQUIRE(words);
        return *words;
    }();
    std::printf("  unlit stage, rt against raster: p99 %u, max %u, %llu pixels beyond 2; rt words lit %llu\n",
                diff->p99, diff->max, static_cast<unsigned long long>(diff->over2), static_cast<unsigned long long>(lit));
    CHECK(lit > 8000);
    CHECK(diff->max <= 1);
}

// Multiple importance sampling (M6): next event estimation and the
// material's own sampling, weighed by the power heuristic, on the case that
// needs both -- a glossy metal floor under a large rect light, whose
// highlight light sampling alone finds by luck. Two things are checked. That
// it is the same image: a deep frame with MIS against a deep frame without,
// within the noise either still carries. And that it is a better estimate of
// it: at equal paths, the error against the deep frame without MIS falls
// several times over.
TEST_CASE("MIS weighs light and material sampling into the same image with less noise on a glossy floor",
          "[usd][gpu][mesh][path][mis]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    struct Light {
        const char* name;
        const char* prim;
        double      atLeast;   // how many times less error MIS must reach
        const char* surface = "            float inputs:metallic = 1\n            float inputs:roughness = 0.2\n";
        /// How far the deep frames may part, over the shallow light-sampling
        /// frame's error: light sampling of a dome on a glossy floor is heavy
        /// tailed, and its deep frame carries fireflies that a relative MSE
        /// weighs by their square -- measured apart by 3.4e-2 at 8192 against
        /// 32768 paths while MIS at the two agreed to 1.8e-6. Its sameness
        /// is the rough floor's to show, where light sampling converges.
        /// Two independent deep frames of 8192 paths part by their summed
        /// noise, errNee * 32 / 8192 each: errNee / 128, a factor of two
        /// under the rough floor's bound.
        double      sameOver = 32.0;
    };
    const Light cases[] = {
        {"rect", "def RectLight \"Panel\"\n{\n"
                 "    float inputs:intensity = 4\n    float inputs:width = 4\n    float inputs:height = 2\n"
                 "    double3 xformOp:translate = (0, 2.5, -4)\n"
                 "    double xformOp:rotateX = -90\n"
                 "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n", 3.0},
        {"dome", "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n", 3.0,
         "            float inputs:metallic = 1\n            float inputs:roughness = 0.2\n", 2.0},
        {"dome, rough", "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n", 0.8,
         "            float inputs:metallic = 0\n            float inputs:roughness = 1\n", 64.0},
        {"sphere", "def SphereLight \"Bulb\"\n{\n"
                   "    float inputs:intensity = 20\n    float inputs:radius = 0.8\n"
                   "    double3 xformOp:translate = (0, 2.5, -4)\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n", 1.5},
    };
    for (const Light& light : cases) {
    std::string file = std::string("mis_") + light.name + ".usda";
    std::replace(file.begin(), file.end(), ' ', '_');
    std::replace(file.begin(), file.end(), ',', '_');
    const fs::path path = scratch(file);
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Materials/Metal>\n}\n"
            << light.prim
            << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.2, 2)\n"
               "    double xformOp:rotateX = -12\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Metal\"\n    {\n"
               "        token outputs:surface.connect = </Materials/Metal/Preview.outputs:surface>\n"
               "        def Shader \"Preview\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.9, 0.9, 0.9)\n"
            << light.surface
            << "            token outputs:surface\n        }\n    }\n}\n";
    }
    const uint32_t w = 64, h = 48;
    const auto frame = [&](bool mis, uint32_t total) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathMis(mis);
        (*renderer)->setPathSamples(std::min<uint32_t>(total, 256));
        (*renderer)->setPathTotal(total);
        (*renderer)->setPathBounces(1);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer deepNee = frame(false, 8192);
    const gpu::Buffer deepMis = frame(true, 8192);
    const gpu::Buffer nee = frame(false, 32);
    const gpu::Buffer mis = frame(true, 32);
    auto same = render::compareHdr(*gpu->library, deepMis, deepNee, w, h);
    auto errNee = render::compareHdr(*gpu->library, nee, deepNee, w, h);
    auto errMis = render::compareHdr(*gpu->library, mis, deepNee, w, h);
    REQUIRE(same);
    REQUIRE(errNee);
    REQUIRE(errMis);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
    REQUIRE(empty);
    auto lit = render::compareHdr(*gpu->library, deepNee, *empty, w, h);
    REQUIRE(lit);
    std::printf("  %-11s: deep MIS against deep light sampling relMSE %.3e; at 32 paths against the deep frame, light "
                "sampling alone %.3e, MIS %.3e (%.1f times less); the frame against blank %.2e\n",
                light.name, same->relMse, errNee->relMse, errMis->relMse,
                errNee->relMse / std::max(errMis->relMse, 1e-30), lit->relMse);
    CHECK(lit->relMse > 0.1);
    // Same image: the deep frames differ by their own noise, which the
    // shallow light-sampling frame carries 256 times more of.
    CHECK(same->relMse < errNee->relMse / light.sameOver);
    CHECK(errMis->relMse < errNee->relMse / light.atLeast);
    }
}

// The traced technique over a mesh, through Hydra: what used to trace splats
// and return now path traces the surfaces. A plane alone under one light has
// nothing for a bounce to find, so the traced frame and the raster frame are
// two estimators of the same direct light and must agree to within the noise
// the paths still carry. This says nothing about accumulation: every frame
// here is gathered in one pass.
TEST_CASE("the rt technique path traces a mesh and agrees with the raster's direct light",
          "[usd][gpu][mesh][path]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("rt_surface.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setLightSamples(64);
    // Enough paths that what is left between the two frames is the estimator
    // and not the noise: Monte Carlo error falls as 1/sqrt(N).
    (*renderer)->setPathSamples(1024);
    (*renderer)->setPathBounces(1);
    auto rasterised = (*renderer)->render("/Camera", 0.0, w, h);
    if (!rasterised) FAIL(rasterised.error().toString());
    auto traced = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!traced) FAIL(traced.error().toString());
    // That the traced frame drew the surface where the surface is. Only the
    // coverage is read here: squareMismatches compares colour against albedo
    // times the cosine to the eye, which is the headlight's answer and not
    // this scene's, and the closed form of a sphere light is what the M5 case
    // checks with lrt/test/lambert_irradiance. What this case is for is the
    // line below it.
    const auto counts = squareMismatches(*gpu, *traced, {0.8F, 0.8F, 0.8F});
    std::printf("  the rt technique over a mesh: %u covered, %u coverage mismatches\n", counts[2], counts[0]);
    CHECK(counts[2] > 8000);
    CHECK(counts[0] == 0);
    gpu::BufferDesc desc;
    desc.bytes = rasterised->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto rasterRgba = gpu::Buffer::create(*gpu->device, desc, rasterised->rgba.data());
    auto tracedRgba = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
    REQUIRE(rasterRgba);
    REQUIRE(tracedRgba);
    auto diff = render::compareImages(*gpu->library, *rasterRgba, *tracedRgba, w, h);
    REQUIRE(diff);
    std::printf("  rt against raster over the same light: p99 %u, max %u, %llu pixels beyond 2\n", diff->p99,
                diff->max, static_cast<unsigned long long>(diff->over2));
    CHECK(diff->p99 <= 8);
}

// Progressive accumulation, through the whole chain: the delegate's settings,
// the render pass, the engine's mean. Drawing the same camera again is the same
// frame continued and its paths are added; moving the camera or changing a
// setting is a different frame and the mean starts over. The second half is
// what says the revision is armed rather than decorative -- a revision nothing
// ever raises would let the mean keep accumulating over a scene that changed,
// and no image would look wrong enough to say so.
TEST_CASE("a path traced frame accumulates over passes and starts again when it must",
          "[usd][gpu][mesh][path]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("rt_progressive.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 96;
    const uint32_t h = 72;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathBounces(1);
    (*renderer)->setPathTotal(32);
    // Eight passes of four paths reach the thirty-two it was told to gather.
    std::vector<uint32_t> held;
    for (uint32_t pass = 0; pass < 8; ++pass) {
        if (auto drawn = (*renderer)->draw("/Camera", 0.0, w, h, "rt"); !drawn) {
            FAIL(drawn.error().toString());
        }
        held.push_back((*renderer)->pathAccumulated());
    }
    std::printf("  paths a pixel held after each pass:");
    for (uint32_t n : held) {
        std::printf(" %u", n);
    }
    std::printf("\n  converged: %s\n", (*renderer)->pathConverged() ? "yes" : "no");
    for (uint32_t pass = 0; pass < 8; ++pass) {
        CHECK(held[pass] == 4 * (pass + 1));
    }
    CHECK((*renderer)->pathConverged());

    // A camera somewhere else: a different frame, and the mean starts again.
    render::Camera moved = render::Camera::lookingAt({0.5, 0.25, 0.0}, {0.0, 0.0, -1.0});
    moved.lens.focal = 35.0;
    moved.lens.haperture = 24.576;
    if (auto drawn = (*renderer)->draw(moved, 0.0, w, h, "rt"); !drawn) {
        FAIL(drawn.error().toString());
    }
    const uint32_t afterMove = (*renderer)->pathAccumulated();
    std::printf("  after the camera moved: %u\n", afterMove);
    CHECK(afterMove == 4);
    CHECK_FALSE((*renderer)->pathConverged());

    // And a setting that changes what a path finds: the same again.
    for (uint32_t pass = 0; pass < 3; ++pass) {
        if (auto drawn = (*renderer)->draw(moved, 0.0, w, h, "rt"); !drawn) {
            FAIL(drawn.error().toString());
        }
    }
    const uint32_t beforeSetting = (*renderer)->pathAccumulated();
    (*renderer)->setPathBounces(2);
    if (auto drawn = (*renderer)->draw(moved, 0.0, w, h, "rt"); !drawn) {
        FAIL(drawn.error().toString());
    }
    const uint32_t afterSetting = (*renderer)->pathAccumulated();
    std::printf("  before the bounce count changed: %u, after: %u\n", beforeSetting, afterSetting);
    CHECK(beforeSetting == 16);
    CHECK(afterSetting == 4);

    // An image, not a viewport: render() draws until the total is gathered.
    (*renderer)->setPathBounces(1);
    (*renderer)->setPathTotal(32);
    auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!image) FAIL(image.error().toString());
    std::printf("  render() with a total of 32 and 4 a pass left %u paths gathered\n",
                (*renderer)->pathAccumulated());
    CHECK((*renderer)->pathAccumulated() == 32);
    CHECK((*renderer)->pathConverged());

    // Adaptive: with a total no image would reach, render() ends when every
    // covered pixel's error is below the target -- the other gate.
    (*renderer)->setPathTotal(100000);
    (*renderer)->setPathAdaptive(true);
    (*renderer)->setPathError(0.1F);
    auto adaptive = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!adaptive) FAIL(adaptive.error().toString());
    std::printf("  adaptive at 10%% against a total of 100000: gathered %u a pixel at most, converged %s\n",
                (*renderer)->pathAccumulated(), (*renderer)->pathConverged() ? "yes" : "no");
    CHECK((*renderer)->pathConverged());
    CHECK((*renderer)->pathAccumulated() < 100000);
}

// The denoiser from the engine: lrt:denoise runs OIDN over a path traced
// frame once it has gathered lrt:pathTotal, in place. Two frames of the same
// stage and paths, one with the setting and one without, must differ once the
// total is reached -- and must not differ while it is not, which is the gate.
TEST_CASE("lrt:denoise runs once a path traced frame is gathered, and not before",
          "[usd][gpu][mesh][path][denoise]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!technique::denoiserBuilt()) {
        SKIP("built without OIDN");
    }
    if (auto probe = technique::Denoiser::create(*gpu->library); !probe) {
        SKIP(std::string("no denoiser here: ") + probe.error().toString());
    }
    const fs::path path = scratch("rt_denoise.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 96;
    const uint32_t h = 72;
    const auto frame = [&](bool denoise, uint32_t total) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(4);
        (*renderer)->setPathBounces(1);
        (*renderer)->setPathTotal(total);
        (*renderer)->setDenoise(denoise);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const uint32_t words = w * h * 4;
    // Gathered in one pass: the denoiser runs, and the frame is not the mean.
    const gpu::Buffer plain = frame(false, 4);
    const gpu::Buffer denoised = frame(true, 4);
    auto changed = render::countDifferent(*gpu->library, plain, denoised, words);
    REQUIRE(changed);
    // Not yet gathered: the gate holds, and the two frames are the same frame.
    // render() draws until the total, so this half is one draw() -- four of
    // sixty-four paths -- read back as the host maps the colour output.
    const auto drawnOnce = [&](bool denoise) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(4);
        (*renderer)->setPathBounces(1);
        (*renderer)->setPathTotal(64);
        (*renderer)->setDenoise(denoise);
        if (auto drawn = (*renderer)->draw("/Camera", 0.0, w, h, "rt"); !drawn) {
            FAIL(drawn.error().toString());
        }
        CHECK((*renderer)->pathAccumulated() == 4);
        CHECK_FALSE((*renderer)->pathConverged());
        auto bytes = (*renderer)->mappedOutput("color");
        if (!bytes) FAIL(bytes.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = bytes->size();
        desc.elementBytes = 4;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, bytes->data());
        REQUIRE(buffer);
        return std::pair<gpu::Buffer, uint32_t>(*buffer, static_cast<uint32_t>(bytes->size() / 4));
    };
    const auto [plainEarly, earlyWords] = drawnOnce(false);
    const auto [gated, gatedWords] = drawnOnce(true);
    REQUIRE(earlyWords == gatedWords);
    auto unchanged = render::countDifferent(*gpu->library, plainEarly, gated, earlyWords);
    REQUIRE(unchanged);
    std::printf("  denoise at the total: %llu of %u words changed; before the total: %llu\n",
                static_cast<unsigned long long>(*changed), words, static_cast<unsigned long long>(*unchanged));
    CHECK(*changed > 0);
    CHECK(*unchanged == 0);
}

// The camera's exposure: UsdGeomCamera's `exposure`, in stops, reaches the
// engine through HdCamera and scales the composed frame by 2^stops -- exactly,
// since a power of two is an exponent bump in float. So the frame with
// exposure 1 authored must be the frame without it, doubled on the device.
TEST_CASE("a camera's exposure scales the frame by a power of two, exactly", "[usd][gpu][mesh][camera]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto stageWith = [&](const char* name, const char* exposureLine) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        std::string stage = kSquareStage;
        // kSquareStage authors /Camera; the exposure goes in beside its focal length.
        const size_t at = stage.find("float focalLength");
        REQUIRE(at != std::string::npos);
        stage.insert(at, exposureLine);
        out << stage;
        return path;
    };
    const fs::path plainPath = stageWith("exposure_plain.usda", "");
    const fs::path stopPath = stageWith("exposure_one_stop.usda", "float exposure = 1\n        ");
    const uint32_t w = 96;
    const uint32_t h = 72;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    gpu::Buffer plain = frame(plainPath);
    const gpu::Buffer stop = frame(stopPath);
    // Double the plain frame on the device with the same kernel the engine
    // uses, and the two must be the same words.
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/technique/exposure", "applyExposure");
    if (!made) FAIL(made.error().toString());
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(plain.rhi());
            cursor["params"]["scale"].setData(2.0F);
            cursor["params"]["pixels"].setData(w * h);
        });
        REQUIRE(batch.submit(true));
    }
    auto differing = render::countDifferent(*gpu->library, plain, stop, w * h * 4);
    REQUIRE(differing);
    std::printf("  exposure 1 authored against the plain frame doubled: %llu of %u words differ\n",
                static_cast<unsigned long long>(*differing), w * h * 4);
    CHECK(*differing == 0);
}

// A UsdLux cylinder light through Hydra: radius and length as authored, the
// axis along the prim's x, lighting a Lambert plane as the closed form says.
TEST_CASE("a UsdLux cylinder light through Hydra lights a Lambert plane as the closed form says",
          "[usd][gpu][mesh][lights][cylinder]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("light_cylinder.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def CylinderLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.3\n"
               "    float inputs:length = 1.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 81;
    const uint32_t h = 61;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setLightSamples(65536);   // a one-sided curved emitter: sigma ~0.6% here, see the technique case
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    test::dumpPpm("cylinder_light_hydra", image->rgba.data(), w, h);

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(colours->rhi());
            cursor["depth"].setBinding(depth->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor p = cursor["plane"];
            p["kind"].setData(uint32_t{6});
            p["vertices"].setData(uint32_t{512});
            p["row0"].setData(toWorld.data(), sizeof(float) * 4);
            p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            const float centre[4] = {0.0F, 0.0F, -3.0F, 0.0F};
            const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.3F};
            const float axisY[4] = {0.0F, 1.0F, 0.0F, 1.4F};
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
    }
    uint32_t c[2] = {0, 0};
    float relative = 0.0F;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(relative), &relative));
    std::printf("  cylinder light through Hydra: %u pixels, %u beyond 3%%, worst %.4f\n", c[0], c[1],
                double(relative));
    CHECK(c[0] > 2000);
    CHECK(c[1] == 0);
}

// A UsdLux IES profile through Hydra: a cutoff at 20 degrees, one inside and
// zero outside, on a small sphere light over the plane. Inside the cone the
// frame must be the frame without the profile, word for word; outside it
// must be black. A band about the cutoff is the interpolation's and skipped.
TEST_CASE("a UsdLux IES profile shapes a light through Hydra: lit inside its cone as without, black outside",
          "[usd][gpu][mesh][lights][ies]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path ies = scratch("cutoff20.ies");
    {
        std::ofstream out(ies);
        out << "IESNA:LM-63-2002\n[TEST] cutoff 20\nTILT=NONE\n1 1000 1 37 1 1 2 0 0 0\n1 1 0\n";
        for (int k = 0; k < 37; ++k) out << (k * 5) << (k == 36 ? "\n" : " ");
        out << "0\n";
        for (int k = 0; k < 37; ++k) out << (k * 5 <= 20 ? "1" : "0") << (k == 36 ? "\n" : " ");
    }
    const auto stageWith = [&](const char* name, bool shaped) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << kSquareStage << "def SphereLight \"Key\"";
        if (shaped) out << " (\n    prepend apiSchemas = [\"ShapingAPI\"]\n)";
        out << "\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.05\n"
               "    float inputs:intensity = 400\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n";
        if (shaped) out << "    asset inputs:shaping:ies:file = @" << ies.string() << "@\n";
        out << "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
        return path;
    };
    const uint32_t w = 161;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setLightSamples(64);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        return *image;
    };
    const usd::StageImage plain = frame(stageWith("ies_plain.usda", false));
    const usd::StageImage shaped = frame(stageWith("ies_cutoff.usda", true));
    test::dumpPpm("ies_cutoff_hydra", shaped.rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = plain.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto with = gpu::Buffer::create(*gpu->device, desc, shaped.rgba.data());
    auto without = gpu::Buffer::create(*gpu->device, desc, plain.rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, shaped.depth, "depth");
    REQUIRE(with);
    REQUIRE(without);
    REQUIRE(depth);
    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/ies_check", "iesCutoff");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 6, "ies.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 2, "ies.worst");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["with"].setBinding(with->rhi());
            cursor["without"].setBinding(without->rhi());
            cursor["depth"].setBinding(depth->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor c = cursor["check"];
            const float centre[4] = {0.0F, 0.0F, -3.0F, 20.0F};
            c["lightCentre"].setData(centre, sizeof(centre));
            c["row0"].setData(toWorld.data(), sizeof(float) * 4);
            c["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            c["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[6] = {};
    float e[2] = {};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(e), e));
    std::printf("  IES cutoff through Hydra: inside the cone %u of %u pixels differ from the unshaped frame; outside "
                "it %u of %u are lit (largest %.3e, farthest at %.1f degrees)\n",
                c[2], c[4], c[3], c[5], double(e[0]), double(e[1]));
    CHECK(c[4] > 100);
    CHECK(c[5] > 100);
    CHECK(c[2] == 0);
    CHECK(c[3] == 0);
}


// Light instancing through Hydra: a PointInstancer whose prototype is a
// sphere light, against the same three lights authored one by one, on the
// square. The frames must agree to what a differently ordered light table
// allows -- the instancer's copies are the authored lights, placed by the
// device.
TEST_CASE("a PointInstancer of lights lights the scene as its instances authored one by one",
          "[usd][gpu][mesh][lights][instancing]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const char* positions[3] = {"(-1, 0.3, -3)", "(0, -0.5, -2.5)", "(1.2, 0.2, -3.5)"};
    const std::string light =
        "    bool inputs:shadow:enable = 0\n"
        "    float inputs:radius = 0.1\n"
        "    float inputs:intensity = 60\n"
        "    bool inputs:normalize = 0\n"
        "    color3f inputs:color = (1, 0.9, 0.8)\n";
    const std::string materials =
        "def Scope \"Materials\"\n{\n"
        "    def Material \"Mat\"\n    {\n"
        "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
        "        def Shader \"Surface\"\n        {\n"
        "            uniform token info:id = \"ND_surface\"\n"
        "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
        "            token outputs:out\n        }\n"
        "        def Shader \"Diffuse\"\n        {\n"
        "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
        "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
        "            float inputs:roughness = 0\n"
        "            token outputs:out\n        }\n    }\n}\n";
    const fs::path instanced = scratch("light_instancer.usda");
    {
        std::ofstream out(instanced);
        out << kSquareStage << "def PointInstancer \"Many\"\n{\n"
            << "    rel prototypes = [</Many/Prototypes/Key>]\n"
            << "    int[] protoIndices = [0, 0, 0]\n"
            << "    point3f[] positions = [" << positions[0] << ", " << positions[1] << ", " << positions[2] << "]\n"
            << "    def Scope \"Prototypes\"\n    {\n"
            << "        def SphereLight \"Key\"\n        {\n" << light << "        }\n    }\n}\n" << materials;
    }
    const fs::path authored = scratch("light_authored.usda");
    {
        std::ofstream out(authored);
        out << kSquareStage;
        for (int k = 0; k < 3; ++k) {
            out << "def SphereLight \"Key" << k << "\"\n{\n" << light
                << "    double3 xformOp:translate = " << positions[k] << "\n"
                << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        }
        out << materials;
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setLightSamples(64);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        return *image;
    };
    const usd::StageImage a = frame(instanced);
    const usd::StageImage b = frame(authored);
    test::dumpPpm("light_instancer_hydra", a.rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = a.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto bufferA = gpu::Buffer::create(*gpu->device, desc, a.rgba.data());
    auto bufferB = gpu::Buffer::create(*gpu->device, desc, b.rgba.data());
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    auto diff = render::compareHdr(*gpu->library, *bufferA, *bufferB, w, h);
    REQUIRE(diff);
    std::vector<float> blank(a.rgba.size(), 0.0F);
    auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
    REQUIRE(blankBuffer);
    auto drawn = render::compareHdr(*gpu->library, *bufferA, *blankBuffer, w, h);
    REQUIRE(drawn);
    std::printf("  PointInstancer of sphere lights against authored lights: relMSE %.2e, max relative %.2e "
                "(against blank: relMSE %.2e)\n",
                diff->relMse, diff->maxRelative, drawn->relMse);
    CHECK(drawn->relMse > 1.0);
    CHECK(diff->relMse < 1e-8);
}

// UsdGeomCamera's diaphragm through Hydra: fStop and focusDistance reach the
// path tracer's own primary rays. With the square in focus the frame is the
// pinhole camera's, exactly; with the focus in front of it the square's
// edges blur, which the frame shows as a difference. Distortion is checked
// to the pixel in the technique; here it only has to arrive.
TEST_CASE("a UsdGeomCamera's fStop and focusDistance reach the path tracer through Hydra",
          "[usd][gpu][mesh][path][camera]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const std::string rest =
        "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n"
        "def Scope \"Materials\"\n{\n"
        "    def Material \"Mat\"\n    {\n"
        "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
        "        def Shader \"Surface\"\n        {\n"
        "            uniform token info:id = \"ND_surface\"\n"
        "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
        "            token outputs:out\n        }\n"
        "        def Shader \"Diffuse\"\n        {\n"
        "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
        "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
        "            float inputs:roughness = 0\n"
        "            token outputs:out\n        }\n    }\n}\n";
    const auto stageWith = [&](const char* name, const char* lens) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        // kSquareStage's camera, with the lens authored inside it.
        std::string text = kSquareStage;
        const size_t at = text.find("    float2 clippingRange");
        REQUIRE(at != std::string::npos);
        text.insert(at, lens);
        out << text << rest;
        return path;
    };
    // An even width: with the square's triangles meeting on a diagonal
    // through the frame's centre, an odd width put pixel centres exactly on
    // that shared edge, and one lens ray in four converging on such a point
    // fell through the seam.
    const uint32_t w = 160;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(4);
        (*renderer)->setPathTotal(4);
        (*renderer)->setPathBounces(0);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        return *image;
    };
    const usd::StageImage pinhole = frame(stageWith("lens_pinhole.usda", ""));
    const usd::StageImage inFocus =
        frame(stageWith("lens_focus5.usda", "    float fStop = 8\n    float focusDistance = 5\n"));
    const usd::StageImage outOfFocus =
        frame(stageWith("lens_focus2.usda", "    float fStop = 8\n    float focusDistance = 2.5\n"));
    const usd::StageImage distorted = frame(stageWith(
        "lens_distorted.usda", "    token lensDistortion:type = \"standard\"\n    float lensDistortion:k1 = 0.3\n"));
    test::dumpPpm("dof_hydra", outOfFocus.rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = pinhole.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto a = gpu::Buffer::create(*gpu->device, desc, pinhole.rgba.data());
    auto b = gpu::Buffer::create(*gpu->device, desc, inFocus.rgba.data());
    auto c = gpu::Buffer::create(*gpu->device, desc, outOfFocus.rgba.data());
    auto d = gpu::Buffer::create(*gpu->device, desc, distorted.rgba.data());
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(c);
    REQUIRE(d);
    auto focused = render::compareHdr(*gpu->library, *a, *b, w, h);
    auto blurred = render::compareHdr(*gpu->library, *a, *c, w, h);
    auto bent = render::compareHdr(*gpu->library, *a, *d, w, h);
    REQUIRE(focused);
    REQUIRE(blurred);
    REQUIRE(bent);
    std::printf("  UsdGeomCamera lens through Hydra: in focus relMSE %.2e (p99 relative %.2e, max %.2e) against the "
                "pinhole; out of focus %.2e; distorted %.2e\n",
                focused->relMse, focused->p99Relative, focused->maxRelative, blurred->relMse, bent->relMse);
    CHECK(focused->relMse < 1e-8);
    CHECK(blurred->relMse > 1e-3);
    CHECK(bent->relMse > 1e-3);
}

// A mesh whose points are time sampled, through Hydra: at the second time
// Hydra hands the delegate new points and the same topology, the engine
// rebuilds the mesh under the same key, and the scene takes it as a
// deformation -- the pools' generation stands while their positions revision
// rises -- so the hardware and compute structures are refit. Checked by the
// three routes agreeing on the deformed frame, and by that frame differing
// from the first time's.
TEST_CASE("time sampled points deform a mesh in place through Hydra, and every route sees the deformation",
          "[usd][gpu][mesh][deformation][refit]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries, to compare all three routes");
    }
    const int n = 12;
    const auto grid = [&](double amplitude) {
        std::string out = "[";
        for (int y = 0; y <= n; ++y) {
            for (int x = 0; x <= n; ++x) {
                const double fx = static_cast<double>(x) / n * 4.0 - 2.0;
                const double fy = static_cast<double>(y) / n * 3.0 - 1.5;
                const double fz = -6.0 + amplitude * std::sin(fx * 1.9) * std::cos(fy * 2.3);
                char buffer[96];
                std::snprintf(buffer, sizeof(buffer), "%s(%.4f, %.4f, %.4f)", (x == 0 && y == 0) ? "" : ", ", fx, fy,
                              fz);
                out += buffer;
            }
        }
        return out + "]";
    };
    std::string counts = "[";
    std::string indices = "[";
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int p = y * (n + 1) + x;
            char buffer[96];
            std::snprintf(buffer, sizeof(buffer), "%s4", (x == 0 && y == 0) ? "" : ", ");
            counts += buffer;
            // Counter-clockwise seen from +z, where the camera is: single sided
            // by USD's default, the sheet faces it.
            std::snprintf(buffer, sizeof(buffer), "%s%d, %d, %d, %d", (x == 0 && y == 0) ? "" : ", ", p, p + 1,
                          p + n + 2, p + n + 1);
            indices += buffer;
        }
    }
    counts += "]";
    indices += "]";
    const fs::path path = scratch("deforming.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Sheet\"\n{\n"
            << "    int[] faceVertexCounts = " << counts << "\n"
            << "    int[] faceVertexIndices = " << indices << "\n"
            << "    point3f[] points.timeSamples = {\n        0: " << grid(0.0) << ",\n        1: " << grid(1.2)
            << ",\n    }\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.6, 0.7, 0.3)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t w = 200;
    const uint32_t h = 150;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const auto frame = [&](double time, const char* route) {
        REQUIRE((*renderer)->setMeshVisibility(route));
        auto image = (*renderer)->render("/Camera", time, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer flat = frame(0.0, "raster");
    const gpu::Buffer flatRays = frame(0.0, "rays");
    const uint64_t generation = (*renderer)->meshGeneration();
    const uint64_t revision = (*renderer)->meshPositionsRevision();
    const gpu::Buffer bentRaster = frame(1.0, "raster");
    CHECK((*renderer)->meshGeneration() == generation);
    CHECK((*renderer)->meshPositionsRevision() > revision);
    const gpu::Buffer bentRays = frame(1.0, "rays");
    const gpu::Buffer bentBvh = frame(1.0, "bvh");
    CHECK((*renderer)->meshGeneration() == generation);
    auto moved = render::compareImages(*gpu->library, flat, bentRaster, w, h);
    auto baseline = render::compareImages(*gpu->library, flat, flatRays, w, h);
    auto rays = render::compareImages(*gpu->library, bentRaster, bentRays, w, h);
    auto bvh = render::compareImages(*gpu->library, bentRaster, bentBvh, w, h);
    REQUIRE(moved);
    REQUIRE(baseline);
    REQUIRE(rays);
    REQUIRE(bvh);
    // The routes differ along triangle edges (the rasteriser's coverage rule
    // against a ray at the pixel's centre); the interior agrees, which the
    // technique's check counts exactly. Here the deformed frame's edge
    // disagreement must be no more than the flat frame's kind: a stale
    // structure would disagree over the whole sheet.
    std::printf("  time 1 against time 0: %llu pixels beyond 2 (max %u); raster against rays: %llu beyond 2 at "
                "time 1, %llu at time 0; against the compute BVH %llu; generation %llu, positions revision %llu "
                "-> %llu\n",
                static_cast<unsigned long long>(moved->over2), moved->max,
                static_cast<unsigned long long>(rays->over2), static_cast<unsigned long long>(baseline->over2),
                static_cast<unsigned long long>(bvh->over2), static_cast<unsigned long long>(generation),
                static_cast<unsigned long long>(revision),
                static_cast<unsigned long long>((*renderer)->meshPositionsRevision()));
    CHECK(moved->over2 > uint64_t{w} * h / 20);
    CHECK(rays->over2 <= 2 * baseline->over2 + 50);
    CHECK(bvh->over2 <= 2 * baseline->over2 + 50);
    CHECK(rays->over2 < uint64_t{w} * h / 50);
    CHECK(bvh->over2 < uint64_t{w} * h / 50);
}

// Motion blur through Hydra: a square sliding between two frames under a
// camera whose shutter is open about the frame, path traced in eight
// slices. The blurred frame differs from the frame the same stage gives
// with the shutter closed; and the same with the square's points sliding
// instead of its transform. The staircase itself is checked in the
// technique; here the shutter, the samples and the buckets have to arrive.
TEST_CASE("a camera's shutter blurs a moving mesh through Hydra, rigid and deforming",
          "[usd][gpu][mesh][path][motion]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const std::string sun =
        "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
    const auto stageWith = [&](const char* name, bool shutter, bool deform) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n";
        if (deform) {
            out << "    point3f[] points.timeSamples = {\n"
                   "        0: [(-1.5, -1, -5), (-0.5, -1, -5), (-0.5, 1, -5), (-1.5, 1, -5)],\n"
                   "        1: [(0.5, -1, -5), (1.5, -1, -5), (1.5, 1, -5), (0.5, 1, -5)],\n    }\n";
        } else {
            out << "    point3f[] points = [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)]\n"
                   "    double3 xformOp:translate.timeSamples = {\n        0: (-1, 0, 0),\n        1: (1, 0, 0),\n    }\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        }
        out << "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (shutter) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        }
        out << "}\n" << sun;
        return path;
    };
    const uint32_t w = 160;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    for (const bool deform : {false, true}) {
        const gpu::Buffer sharp = frame(stageWith(deform ? "blur_points_closed.usda" : "blur_closed.usda", false, deform));
        const gpu::Buffer blurred = frame(stageWith(deform ? "blur_points_open.usda" : "blur_open.usda", true, deform));
        auto diff = render::compareHdr(*gpu->library, sharp, blurred, w, h);
        REQUIRE(diff);
        std::vector<float> blank(static_cast<size_t>(w) * h * 4, 0.0F);
        gpu::BufferDesc desc;
        desc.bytes = blank.size() * sizeof(float);
        desc.elementBytes = 16;
        auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
        REQUIRE(blankBuffer);
        auto drawn = render::compareHdr(*gpu->library, blurred, *blankBuffer, w, h);
        REQUIRE(drawn);
        std::printf("  %s through Hydra: shutter open against closed relMSE %.2e (blurred frame against blank %.2e)\n",
                    deform ? "sliding points" : "sliding transform", diff->relMse, drawn->relMse);
        CHECK(drawn->relMse > 1.0);
        CHECK(diff->relMse > 1e-2);
    }
}

// A PointInstancer whose positions move under the shutter blurs its
// instances as the same squares authored one by one, each sliding by its
// own transform: the set's chain is composed at the two samples and its
// records copied per shutter slice between them, as a single instance's
// are. And the blur is there: against the shutter closed, the frames differ.
TEST_CASE("a moving PointInstancer's instances blur as the same prims authored one by one",
          "[usd][gpu][mesh][path][motion][instancing]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const char* square = "    int[] faceVertexCounts = [4]\n"
                         "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                         "    point3f[] points = [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0.4, 0.4, 0), (-0.4, 0.4, 0)]\n"
                         "    uniform token subdivisionScheme = \"none\"\n"
                         "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n";
    const auto stage = [&](const char* name, bool instanced, bool shutter) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n";
        if (instanced) {
            out << "def PointInstancer \"Squares\"\n{\n"
                   "    int[] protoIndices = [0, 0]\n"
                   "    point3f[] positions.timeSamples = {\n"
                   "        0: [(-1.5, -0.5, -5), (0.5, 0.6, -6)],\n"
                   "        1: [(-0.5, -0.5, -5), (1.5, 0.9, -6)],\n    }\n"
                   "    rel prototypes = </Squares/Proto>\n"
                   "    def Mesh \"Proto\"\n    {\n" << square << "    }\n}\n";
        } else {
            const char* from[] = {"(-1.5, -0.5, -5)", "(0.5, 0.6, -6)"};
            const char* to[] = {"(-0.5, -0.5, -5)", "(1.5, 0.9, -6)"};
            for (int k = 0; k < 2; ++k) {
                out << "def Mesh \"Square" << k << "\"\n{\n" << square
                    << "    double3 xformOp:translate.timeSamples = {\n        0: " << from[k] << ",\n        1: " << to[k]
                    << ",\n    }\n"
                       "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
            }
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (shutter) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        }
        out << "}\n"
               "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
        return path;
    };
    const uint32_t w = 160, h = 120;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer instancedBlur = frame(stage("instancer_blur.usda", true, true));
    const gpu::Buffer singlesBlur = frame(stage("instancer_blur_singles.usda", false, true));
    const gpu::Buffer instancedSharp = frame(stage("instancer_sharp.usda", true, false));
    auto same = render::compareHdr(*gpu->library, instancedBlur, singlesBlur, w, h);
    auto blurred = render::compareHdr(*gpu->library, instancedBlur, instancedSharp, w, h);
    REQUIRE(same);
    REQUIRE(blurred);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
    REQUIRE(empty);
    auto drawn = render::compareHdr(*gpu->library, instancedBlur, *empty, w, h);
    REQUIRE(drawn);
    std::printf("  a moving instancer against the same squares one by one: relMSE %.2e (max relative %.2e); against "
                "the shutter closed %.2e; against blank %.2e\n",
                same->relMse, same->maxRelative, blurred->relMse, drawn->relMse);
    CHECK(drawn->relMse > 1.0);
    CHECK(blurred->relMse > 1e-2);
    CHECK(same->maxRelative < 1e-3);
}

// A camera that moves under the shutter: everything it sees blurs as if the
// scene moved the other way. Two squares under a sun that lights by
// direction alone (no shadows), so where a square is does not change its
// shade -- the camera sliding +x over still squares must draw what a still
// camera draws of the squares sliding -x, slice by slice. And the blur is
// there: against the same camera with the shutter closed, the frames differ.
TEST_CASE("a camera moving under the shutter blurs the frame as the scene moving the other way does",
          "[usd][gpu][mesh][path][motion][camera]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const char* square = "    int[] faceVertexCounts = [4]\n"
                         "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                         "    point3f[] points = [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0.4, 0.4, 0), (-0.4, 0.4, 0)]\n"
                         "    uniform token subdivisionScheme = \"none\"\n"
                         "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n";
    // cameraMoves: the camera slides from x 0 to 1 over frames 0..1 and the
    // squares stand; otherwise the camera stands and the squares slide 0..-1.
    const auto stage = [&](const char* name, bool cameraMoves, bool shutter) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n";
        const char* at[] = {"(-0.8, -0.3, -5)", "(0.9, 0.5, -6)"};
        const char* moved[] = {"(-1.8, -0.3, -5)", "(-0.1, 0.5, -6)"};
        for (int k = 0; k < 2; ++k) {
            out << "def Mesh \"Square" << k << "\"\n{\n" << square;
            if (cameraMoves) {
                out << "    double3 xformOp:translate = " << at[k] << "\n";
            } else {
                out << "    double3 xformOp:translate.timeSamples = {\n        0: " << at[k] << ",\n        1: " << moved[k]
                    << ",\n    }\n";
            }
            out << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (cameraMoves) {
            out << "    double3 xformOp:translate.timeSamples = {\n        0: (0, 0, 0),\n        1: (1, 0, 0),\n    }\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        }
        if (shutter) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        }
        out << "}\n"
               "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
        return path;
    };
    const uint32_t w = 160, h = 120;
    // Drawn at frame 0.5 with the camera's frame position subtracted: the
    // still-camera stage at 0.5 has the squares half way along, and the
    // moving camera's frame view is half way along too.
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer cameraBlur = frame(stage("camera_blur.usda", true, true));
    const gpu::Buffer sceneBlur = frame(stage("camera_blur_scene.usda", false, true));
    const gpu::Buffer cameraSharp = frame(stage("camera_sharp.usda", true, false));
    auto same = render::compareHdr(*gpu->library, cameraBlur, sceneBlur, w, h);
    auto blurred = render::compareHdr(*gpu->library, cameraBlur, cameraSharp, w, h);
    REQUIRE(same);
    REQUIRE(blurred);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
    REQUIRE(empty);
    auto drawn = render::compareHdr(*gpu->library, cameraBlur, *empty, w, h);
    REQUIRE(drawn);
    std::printf("  a moving camera against the scene moving the other way: relMSE %.2e (max relative %.2e); "
                "against the shutter closed %.2e; against blank %.2e\n",
                same->relMse, same->maxRelative, blurred->relMse, drawn->relMse);
    CHECK(drawn->relMse > 1.0);
    CHECK(blurred->relMse > 1e-2);
    CHECK(same->maxRelative < 1e-3);
}

// A light that moves under the shutter: its light and its shadows move with
// it, slice by slice. The same instants arranged the other way round -- the
// light still and everything else (the camera, a floor, an occluder)
// sliding the opposite way -- are the same relative scene at every time, so
// the two frames must agree; and against the light standing still, they
// must not.
TEST_CASE("a light moving under the shutter lights and shadows as the scene moving the other way does",
          "[usd][gpu][mesh][path][motion][lights]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    // LightMoves: the light slides x 0 -> 2 and all else stands at x 0.
    // SceneMoves: the light stands at x 1 and the camera, floor and occluder
    // slide x 1 -> -1, so that everything less the light is -2t either way.
    // Still: the light at its frame position, x 1, nothing moving.
    enum class Arrangement { LightMoves, SceneMoves, Still };
    const auto stage = [&](const char* name, Arrangement arrangement) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        const bool sceneMoves = arrangement == Arrangement::SceneMoves;
        const auto slide = [&](const char* at, const char* to) {
            std::string text;
            if (sceneMoves) {
                text = std::string("    double3 xformOp:translate.timeSamples = {\n        0: ") + at + ",\n        1: " + to +
                       ",\n    }\n";
            } else {
                text = "    double3 xformOp:translate = (0, 0, 0)\n";
            }
            return text + "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        };
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-8, -1, 0), (8, -1, 0), (8, -1, -12), (-8, -1, -12)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n"
            << slide("(1, 0, 0)", "(-1, 0, 0)") << "}\n"
            << "def Mesh \"Occluder\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, 0.2, -4.5), (0.5, 0.2, -4.5), (0.5, 0.2, -5.5), (-0.5, 0.2, -5.5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n"
            << slide("(1, 0, 0)", "(-1, 0, 0)") << "}\n"
            << "def SphereLight \"Bulb\"\n{\n    float inputs:intensity = 30\n    float inputs:radius = 0.2\n";
        if (arrangement == Arrangement::LightMoves) {
            out << "    double3 xformOp:translate.timeSamples = {\n        0: (0, 2, -5),\n        1: (2, 2, -5),\n    }\n";
        } else {
            out << "    double3 xformOp:translate = (1, 2, -5)\n";
        }
        out << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        if (sceneMoves) {
            out << "    double3 xformOp:translate.timeSamples = {\n        0: (1, 1, 1),\n        1: (-1, 1, 1),\n    }\n";
        } else {
            out << "    double3 xformOp:translate = (0, 1, 1)\n";
        }
        out << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 128, h = 96;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(128);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer lightMoves = frame(stage("light_moves.usda", Arrangement::LightMoves));
    const gpu::Buffer sceneMoves = frame(stage("light_moves_scene.usda", Arrangement::SceneMoves));
    const gpu::Buffer still = frame(stage("light_still.usda", Arrangement::Still));
    auto same = render::compareHdr(*gpu->library, lightMoves, sceneMoves, w, h);
    auto moved = render::compareHdr(*gpu->library, lightMoves, still, w, h);
    REQUIRE(same);
    REQUIRE(moved);
    std::printf("  a moving light against the scene moving the other way: relMSE %.2e (p99 relative %.2e); against the "
                "light still %.2e\n",
                same->relMse, same->p99Relative, moved->relMse);
    // The blur is a soft shadow sweeping part of the floor: small over the
    // frame (2.8e-4), and real -- with the light's samples withheld from the
    // kernel, the moving light drew the still frame to 3.7e-13 and parted
    // from the other arrangement by 3.1e-4.
    CHECK(moved->relMse > 1e-5);
    CHECK(same->relMse < moved->relMse / 1000.0);
}

// Velocities: a mesh that authors one sample of points and `velocities`
// must blur as one that authors the two samples those velocities reach --
// UsdGeom's velocity interpolation, resolved by hdsi's scene index ahead
// of the delegate, so the shutter samples read the same either way.
TEST_CASE("authored velocities blur a mesh as the equivalent time samples do", "[usd][gpu][mesh][path][motion][velocity]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    // 24 time codes a second: a velocity of 48 units a second slides the
    // square 2 units a frame, from x -1 at frame 0 to x 1 at frame 1. The
    // velocity stage authors its one sample at the frame drawn, 0.5: the
    // scene index extrapolates from the sample the frame reads.
    const auto stageWith = [&](const char* name, bool velocities) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n"
               "    timeCodesPerSecond = 24\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n";
        if (velocities) {
            out << "    point3f[] points.timeSamples = {\n"
                   "        0.5: [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)],\n    }\n"
                   "    vector3f[] velocities.timeSamples = {\n"
                   "        0.5: [(48, 0, 0), (48, 0, 0), (48, 0, 0), (48, 0, 0)],\n    }\n";
        } else {
            out << "    point3f[] points.timeSamples = {\n"
                   "        0: [(-1.5, -1, -5), (-0.5, -1, -5), (-0.5, 1, -5), (-1.5, 1, -5)],\n"
                   "        1: [(0.5, -1, -5), (1.5, -1, -5), (1.5, 1, -5), (0.5, 1, -5)],\n    }\n";
        }
        out << "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double shutter:open = -0.25\n    double shutter:close = 0.25\n}\n"
               "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
        return path;
    };
    const uint32_t w = 160;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer sampled = frame(stageWith("velocity_samples.usda", false));
    const gpu::Buffer velocities = frame(stageWith("velocity_authored.usda", true));
    auto diff = render::compareHdr(*gpu->library, sampled, velocities, w, h);
    REQUIRE(diff);
    std::printf("  velocities against the equivalent samples: relMSE %.2e, max relative %.2e\n", diff->relMse,
                diff->maxRelative);
    CHECK(diff->relMse < 1e-6);
}

// Skinning through Hydra: a square bound to one joint of a two-joint
// skeleton whose animation slides that joint. usdSkelImaging hands the
// delegate the skinning as ext computation prims; the engine runs them on
// the device (geom::Skinner) and builds the mesh from the skinned points.
// With every weight on the sliding joint the skinned square is the square
// authored with that slide as its transform -- to the pixel, both routes --
// and at the rest pose it is the square as authored. The skinned mesh keeps
// its topology key across frames, so the animation is a refit, not a
// repack.
TEST_CASE("a skeleton's animation skins a mesh on the device, as the authored transform draws it",
          "[usd][gpu][mesh][skinning]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const std::string square =
        "    int[] faceVertexCounts = [4]\n"
        "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
        "    point3f[] points = [(-0.5, -0.5, -5), (0.5, -0.5, -5), (0.5, 0.5, -5), (-0.5, 0.5, -5)]\n"
        "    uniform token subdivisionScheme = \"none\"\n"
        "    color3f[] primvars:displayColor = [(0.3, 0.7, 0.5)] ( interpolation = \"constant\" )\n";
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    float2 clippingRange = (0.1, 1000)\n}\n";
    const fs::path skinned = scratch("skinned.usda");
    {
        std::ofstream out(skinned);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def SkelRoot \"Root\"\n{\n"
               "    def Skeleton \"Skel\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
               "        uniform token[] joints = [\"root\", \"root/arm\"]\n"
               "        uniform matrix4d[] bindTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ), "
               "( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        uniform matrix4d[] restTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ), "
               "( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        rel skel:animationSource = </Root/Anim>\n    }\n"
               "    def SkelAnimation \"Anim\"\n    {\n"
               "        uniform token[] joints = [\"root/arm\"]\n"
               "        float3[] translations.timeSamples = {\n            0: [(0, 0, 0)],\n            1: [(1.2, 0.4, 0)],\n        }\n"
               "        quatf[] rotations = [(1, 0, 0, 0)]\n"
               "        half3[] scales = [(1, 1, 1)]\n    }\n"
               "    def Mesh \"Square\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
            << square
            << "        rel skel:skeleton = </Root/Skel>\n"
               "        int[] primvars:skel:jointIndices = [1, 1, 1, 1] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        float[] primvars:skel:jointWeights = [1, 1, 1, 1] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        matrix4d primvars:skel:geomBindTransform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )\n"
               "    }\n}\n"
            << camera;
    }
    const auto plain = [&](const char* name, const char* translate) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
            << square << "    double3 xformOp:translate = " << translate << "\n"
            << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
            << camera;
        return path;
    };
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto frame = [&](usd::StageRenderer& renderer, double time, const char* route) {
        REQUIRE(renderer.setMeshVisibility(route));
        auto image = renderer.render("/Camera", time, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    auto skel = usd::StageRenderer::open(skinned);
    auto rest = usd::StageRenderer::open(plain("skinned_rest.usda", "(0, 0, 0)"));
    auto moved = usd::StageRenderer::open(plain("skinned_moved.usda", "(1.2, 0.4, 0)"));
    if (!skel) FAIL(skel.error().toString());
    if (!rest) FAIL(rest.error().toString());
    if (!moved) FAIL(moved.error().toString());
    const gpu::Buffer skelRest = frame(**skel, 0.0, "raster");
    const uint64_t generation = (*skel)->meshGeneration();
    const gpu::Buffer skelMoved = frame(**skel, 1.0, "raster");
    const gpu::Buffer authoredRest = frame(**rest, 0.0, "raster");
    const gpu::Buffer authoredMoved = frame(**moved, 0.0, "raster");
    auto atRest = render::compareImages(*gpu->library, skelRest, authoredRest, w, h);
    auto atOne = render::compareImages(*gpu->library, skelMoved, authoredMoved, w, h);
    auto changed = render::compareImages(*gpu->library, skelRest, skelMoved, w, h);
    REQUIRE(atRest);
    REQUIRE(atOne);
    REQUIRE(changed);
    std::printf("  skinned square: at rest %llu pixels beyond 2 of the authored square (max %u); slid %llu (max %u); "
                "the slide changes %llu pixels; generation %llu -> %llu\n",
                static_cast<unsigned long long>(atRest->over2), atRest->max,
                static_cast<unsigned long long>(atOne->over2), atOne->max,
                static_cast<unsigned long long>(changed->over2), static_cast<unsigned long long>(generation),
                static_cast<unsigned long long>((*skel)->meshGeneration()));
    CHECK(changed->over2 > 1000);
    CHECK(atRest->max == 0);
    CHECK(atOne->max == 0);
    CHECK((*skel)->meshGeneration() == generation);
    if (gpu->device->caps().rayQuery && gpu->device->caps().accelerationStructure) {
        const gpu::Buffer rays = frame(**skel, 1.0, "rays");
        auto viaRays = render::compareImages(*gpu->library, rays, authoredMoved, w, h);
        REQUIRE(viaRays);
        std::printf("  and through rays: %llu beyond 2 (max %u)\n", static_cast<unsigned long long>(viaRays->over2),
                    viaRays->max);
        CHECK(viaRays->over2 < uint64_t{w} * h / 50);
    }
}

// Blend shapes through Hydra: a shape with an inbetween, weighed by the
// animation. At weight 1 the square is the square authored with the
// shape's offsets; at weight 0.5 it is the inbetween authored at 0.5 -- not
// half the shape's offsets, which is what proves the inbetween is resolved
// -- both to the pixel.
TEST_CASE("blend shapes and their inbetweens deform a mesh through Hydra as the authored shapes draw it",
          "[usd][gpu][mesh][skinning][blendshapes]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto squareWith = [](const char* p0, const char* p1, const char* p2, const char* p3) {
        return std::string("    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                           "    point3f[] points = [") +
               p0 + ", " + p1 + ", " + p2 + ", " + p3 +
               "]\n    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.3, 0.7, 0.5)] ( interpolation = \"constant\" )\n";
    };
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    float2 clippingRange = (0.1, 1000)\n}\n";
    const fs::path shaped = scratch("blendshape.usda");
    {
        std::ofstream out(shaped);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 2\n)\n"
               "def SkelRoot \"Root\"\n{\n"
               "    def Skeleton \"Skel\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
               "        uniform token[] joints = [\"root\"]\n"
               "        uniform matrix4d[] bindTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        uniform matrix4d[] restTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        rel skel:animationSource = </Root/Anim>\n    }\n"
               "    def SkelAnimation \"Anim\"\n    {\n"
               "        uniform token[] joints = [\"root\"]\n"
               "        float3[] translations = [(0, 0, 0)]\n"
               "        quatf[] rotations = [(1, 0, 0, 0)]\n"
               "        half3[] scales = [(1, 1, 1)]\n"
               "        uniform token[] blendShapes = [\"puff\"]\n"
               "        float[] blendShapeWeights.timeSamples = {\n            0: [0],\n            1: [0.5],\n            2: [1],\n        }\n"
               "    }\n"
               "    def Mesh \"Square\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
            << squareWith("(-0.5, -0.5, -5)", "(0.5, -0.5, -5)", "(0.5, 0.5, -5)", "(-0.5, 0.5, -5)")
            << "        rel skel:skeleton = </Root/Skel>\n"
               "        uniform token[] skel:blendShapes = [\"puff\"]\n"
               "        rel skel:blendShapeTargets = [</Root/Square/Puff>]\n"
               "        int[] primvars:skel:jointIndices = [0, 0, 0, 0] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        float[] primvars:skel:jointWeights = [1, 1, 1, 1] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        def BlendShape \"Puff\"\n        {\n"
               "            uniform vector3f[] offsets = [(0.8, 0, 0), (0, 0.6, 0)]\n"
               "            uniform int[] pointIndices = [1, 2]\n"
               "            uniform vector3f[] inbetweens:Half:offsets = [(0.1, -0.3, 0), (-0.2, 0.1, 0)] (\n"
               "                weight = 0.5\n            )\n        }\n"
               "    }\n}\n"
            << camera;
    }
    const auto plain = [&](const char* name, const std::string& square) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\ndef Mesh \"Square\"\n{\n" << square << "}\n" << camera;
        return path;
    };
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto frame = [&](usd::StageRenderer& renderer, double time) {
        auto image = renderer.render("/Camera", time, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    auto skel = usd::StageRenderer::open(shaped);
    // Weight 1: the shape's offsets on points 1 and 2. Weight 0.5: the
    // inbetween's own offsets, as authored at 0.5.
    auto full = usd::StageRenderer::open(plain(
        "blendshape_full.usda", squareWith("(-0.5, -0.5, -5)", "(1.3, -0.5, -5)", "(0.5, 1.1, -5)", "(-0.5, 0.5, -5)")));
    auto half = usd::StageRenderer::open(plain(
        "blendshape_half.usda", squareWith("(-0.5, -0.5, -5)", "(0.6, -0.8, -5)", "(0.3, 0.6, -5)", "(-0.5, 0.5, -5)")));
    if (!skel) FAIL(skel.error().toString());
    if (!full) FAIL(full.error().toString());
    if (!half) FAIL(half.error().toString());
    const gpu::Buffer rest = frame(**skel, 0.0);
    const gpu::Buffer atHalf = frame(**skel, 1.0);
    const gpu::Buffer atFull = frame(**skel, 2.0);
    const gpu::Buffer authoredFull = frame(**full, 0.0);
    const gpu::Buffer authoredHalf = frame(**half, 0.0);
    auto fullDiff = render::compareImages(*gpu->library, atFull, authoredFull, w, h);
    auto halfDiff = render::compareImages(*gpu->library, atHalf, authoredHalf, w, h);
    auto moved = render::compareImages(*gpu->library, rest, atFull, w, h);
    REQUIRE(fullDiff);
    REQUIRE(halfDiff);
    REQUIRE(moved);
    std::printf("  blend shape at weight 1: %llu pixels beyond 2 of the authored shape (max %u); at 0.5, %llu beyond "
                "the authored inbetween (max %u); the shape changes %llu pixels\n",
                static_cast<unsigned long long>(fullDiff->over2), fullDiff->max,
                static_cast<unsigned long long>(halfDiff->over2), halfDiff->max,
                static_cast<unsigned long long>(moved->over2));
    CHECK(moved->over2 > 1000);
    CHECK(fullDiff->max == 0);
    CHECK(halfDiff->max == 0);
}

// hdsi's conversions ahead of the delegate: a UsdGeomSphere becomes a mesh
// (checked against the analytic sphere, to the tessellation's chord), a
// TetMesh becomes its surface triangles and a degree-one NurbsPatch its
// quad -- each drawn as the mesh authored by hand, to the pixel.
TEST_CASE("implicit surfaces, tetrahedral meshes and NURBS patches arrive as meshes through Hydra",
          "[usd][gpu][mesh][hdsi]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    float2 clippingRange = (0.1, 1000)\n}\n";
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto image = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto out = (*renderer)->render("/Camera", 0.0, w, h);
        if (!out) FAIL(out.error().toString());
        return *out;
    };
    const auto buffer = [&](const usd::StageImage& img) {
        gpu::BufferDesc desc;
        desc.bytes = img.rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, img.rgba.data());
        REQUIRE(made);
        return *made;
    };
    // The depth as an image, for a comparison that does not care which way
    // a face winds: the same triangles at the same places, however shaded.
    const auto depthImage = [&](const usd::StageImage& img) {
        std::vector<float> packed(img.depth.size() * 4, 1.0F);
        for (size_t i = 0; i < img.depth.size(); ++i) {
            packed[i * 4] = img.depth[i];
        }
        gpu::BufferDesc desc;
        desc.bytes = packed.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, packed.data());
        REQUIRE(made);
        return *made;
    };
    // The sphere, against the analytic one: hdsi tessellates with 10 axial
    // and 10 radial segments, so the chord's sag is r (1 - cos(pi / 10)).
    {
        const fs::path path = scratch("hdsi_sphere.usda");
        {
            std::ofstream out(path);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Sphere \"Ball\"\n{\n    double radius = 1.2\n"
                   "    double3 xformOp:translate = (0, 0, -5)\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
                << camera;
        }
        const usd::StageImage img = image(path);
        test::dumpPpm("hdsi_sphere", img.rgba.data(), w, h);
        render::Camera cam;
        cam.lens.focal = 35.0;
        cam.lens.haperture = 24.576;
        const render::Projection projection = render::projectionFor(cam, w, h);
        auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, img.depth, "depth");
        REQUIRE(depth);
        const gpu::Buffer colours = buffer(img);
        auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/visibility_check", "sphereCheck");
        if (!check) FAIL(check.error().toString());
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "sphere.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "sphere.worst");
        const float radius = 1.2F;
        const float sag = radius * (1.0F - std::cos(3.14159265F / 10.0F));
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(colours.rhi());
                cursor["depth"].setBinding(depth->rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                technique::setCamera(cursor["camera"], projection, w, h);
                cursor["sphere"]["z"].setData(5.0F);
                cursor["sphere"]["radius"].setData(radius);
                cursor["sphere"]["sag"].setData(sag);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[3] = {0, 0, 0};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(e), &e));
        std::printf("  UsdGeomSphere as a mesh: %u of %u pixels judged wrong outside the chord band, %u squarely "
                    "covered, depth within %.4f of the analytic sphere (sag %.4f)\n",
                    c[0], c[1], c[2], static_cast<double>(e), static_cast<double>(sag));
        CHECK(c[2] > 2000);
        CHECK(c[0] == 0);
        CHECK(e <= sag / 0.6F + 1e-4F);
    }
    // A tetrahedron as a TetMesh, against its four faces authored as a mesh.
    {
        const char* points = "[(-1, -0.6, -5), (1, -0.6, -5), (0, -0.6, -6.5), (0, 0.9, -5.5)]";
        const fs::path tet = scratch("hdsi_tet.usda");
        {
            std::ofstream out(tet);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def TetMesh \"Tet\"\n{\n"
                << "    point3f[] points = " << points << "\n"
                << "    int4[] tetVertexIndices = [(0, 1, 2, 3)]\n"
                   "    int3[] surfaceFaceVertexIndices = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]\n"
                   "    color3f[] primvars:displayColor = [(0.6, 0.5, 0.2)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        const fs::path faces = scratch("hdsi_tet_faces.usda");
        {
            std::ofstream out(faces);
            // The surface hdsi derives: each face wound outward.
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Mesh \"Tet\"\n{\n"
                << "    point3f[] points = " << points << "\n"
                << "    int[] faceVertexCounts = [3, 3, 3, 3]\n"
                   "    int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    bool doubleSided = 1\n"
                   "    color3f[] primvars:displayColor = [(0.6, 0.5, 0.2)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        // Depth against depth: hdsi winds the surface its own way, and the
        // headlight shades a face by the side it sees.
        const usd::StageImage a = image(tet);
        const usd::StageImage b = image(faces);
        const gpu::Buffer depthA = depthImage(a);
        const gpu::Buffer depthB = depthImage(b);
        auto diff = render::compareHdr(*gpu->library, depthA, depthB, w, h);
        REQUIRE(diff);
        std::vector<float> blank(static_cast<size_t>(w) * h * 4, 0.0F);
        gpu::BufferDesc desc;
        desc.bytes = blank.size() * sizeof(float);
        desc.elementBytes = 16;
        auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
        REQUIRE(blankBuffer);
        auto drawn = render::compareImages(*gpu->library, buffer(a), *blankBuffer, w, h);
        REQUIRE(drawn);
        std::printf("  TetMesh against its faces as a mesh, by depth: relMSE %.2e, max relative %.2e; drawn %llu\n",
                    diff->relMse, diff->maxRelative, static_cast<unsigned long long>(drawn->over2));
        CHECK(drawn->over2 > 1000);
        CHECK(diff->relMse < 1e-8);
    }
    // A bilinear NurbsPatch, against its quad.
    {
        const char* points = "[(-1, -0.7, -5), (1, -0.7, -5), (-1, 0.7, -5), (1, 0.7, -5)]";
        const fs::path patch = scratch("hdsi_nurbs.usda");
        {
            std::ofstream out(patch);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def NurbsPatch \"Patch\"\n{\n"
                   "    int uVertexCount = 2\n    int vVertexCount = 2\n"
                   "    int uOrder = 2\n    int vOrder = 2\n"
                   "    double[] uKnots = [0, 0, 1, 1]\n    double[] vKnots = [0, 0, 1, 1]\n"
                << "    point3f[] points = " << points << "\n"
                << "    color3f[] primvars:displayColor = [(0.2, 0.5, 0.7)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        const fs::path quad = scratch("hdsi_nurbs_quad.usda");
        {
            std::ofstream out(quad);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Mesh \"Patch\"\n{\n"
                << "    point3f[] points = " << points << "\n"
                << "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 3, 2]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    bool doubleSided = 1\n"
                   "    color3f[] primvars:displayColor = [(0.2, 0.5, 0.7)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        const gpu::Buffer a = buffer(image(patch));
        const gpu::Buffer b = buffer(image(quad));
        auto diff = render::compareImages(*gpu->library, a, b, w, h);
        REQUIRE(diff);
        std::printf("  NurbsPatch against its quad: %llu pixels beyond 2 (max %u)\n",
                    static_cast<unsigned long long>(diff->over2), diff->max);
        CHECK(diff->max == 0);
    }
}

// BasisCurves through Hydra: a straight linear curve of width w along x is
// a tube -- a cylinder of radius w/2 tessellated with `sides` sides -- so
// its depth on the row through its axis is the front of that cylinder to
// within a facet's sag, and the rows it covers are those within w/2 of the
// axis, to the same sag. The three visibility routes must agree on it.
TEST_CASE("a linear BasisCurves prim draws as a tube of its width through Hydra, on every route",
          "[usd][gpu][mesh][curves]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("curve.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def BasisCurves \"Hair\"\n{\n"
               "    uniform token type = \"linear\"\n"
               "    int[] curveVertexCounts = [2]\n"
               "    point3f[] points = [(-3, 0.2, -5), (3, 0.2, -5)]\n"
               "    float[] widths = [0.4] ( interpolation = \"constant\" )\n"
               "    color3f[] primvars:displayColor = [(0.7, 0.5, 0.3)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t w = 200;
    const uint32_t h = 150;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    render::Camera cam;
    cam.lens.focal = 35.0;
    cam.lens.haperture = 24.576;
    const render::Projection projection = render::projectionFor(cam, w, h);
    const float radius = 0.2F;
    const float sag = radius * (1.0F - std::cos(3.14159265F / 8.0F));   // 8 sides
    for (const char* route : {"raster", "rays", "bvh"}) {
        if (std::string(route) != "raster" && !(gpu->device->caps().rayQuery && gpu->device->caps().accelerationStructure)) {
            continue;
        }
        REQUIRE((*renderer)->setMeshVisibility(route));
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
        REQUIRE(depth);
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto colours = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(colours);
        auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/visibility_check", "cylinderCheck");
        if (!check) FAIL(check.error().toString());
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "cylinder.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "cylinder.worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(colours->rhi());
                cursor["depth"].setBinding(depth->rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                technique::setCamera(cursor["camera"], projection, w, h);
                rhi::ShaderCursor c = cursor["cylinder"];
                c["y"].setData(0.2F);
                c["z"].setData(5.0F);
                c["radius"].setData(radius);
                c["sag"].setData(sag);
                c["xMin"].setData(-3.0F);
                c["xMax"].setData(3.0F);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[3] = {0, 0, 0};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(e), &e));
        std::printf("  %s: %u pixels squarely on the tube, %u coverage wrong beyond the facet band, depth within "
                    "%.4f of the cylinder's front (sag %.4f)\n",
                    route, c[2], c[0], static_cast<double>(e), static_cast<double>(sag));
        CHECK(c[2] > 500);
        CHECK(c[0] == 0);
        CHECK(e <= sag / 0.7F + 1e-3F);
    }
}

// A hair material on a curve through Hydra: chiang_hair_bsdf bound to a
// BasisCurves prim, lit by a sun. The tube draws lit and not black, and
// differently from the same curve under a Lambert material -- the lobe is
// the one shading it, with the tube's own tangent as the fibre's direction.
TEST_CASE("a chiang hair material shades a curve through Hydra", "[usd][gpu][mesh][curves][hair]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto stageWith = [&](const char* name, bool hair) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def BasisCurves \"Hair\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    uniform token type = \"cubic\"\n    uniform token basis = \"bspline\"\n"
               "    int[] curveVertexCounts = [6]\n"
               "    point3f[] points = [(-3, -0.5, -5), (-2, 0.6, -5), (-1, -0.4, -4.5), (1, 0.5, -5.5), (2, -0.6, -5), (3, 0.5, -5)]\n"
               "    float[] widths = [0.5] ( interpolation = \"constant\" )\n"
               "    rel material:binding = </Materials/Mat>\n}\n"
               "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n"
               "    double xformOp:rotateX = -30\n    uniform token[] xformOpOrder = [\"xformOp:rotateX\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Bsdf.outputs:out>\n"
               "            token outputs:out\n        }\n";
        if (hair) {
            out << "        def Shader \"Bsdf\"\n        {\n"
                   "            uniform token info:id = \"ND_chiang_hair_bsdf\"\n"
                   "            color3f inputs:tint_R = (1, 1, 1)\n"
                   "            color3f inputs:tint_TT = (1, 0.8, 0.6)\n"
                   "            color3f inputs:tint_TRT = (1, 0.9, 0.8)\n"
                   "            float3 inputs:absorption_coefficient = (0.2, 0.4, 0.8)\n"
                   "            token outputs:out\n        }\n    }\n}\n";
        } else {
            out << "        def Shader \"Bsdf\"\n        {\n"
                   "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
                   "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
                   "            token outputs:out\n        }\n    }\n}\n";
        }
        return path;
    };
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setLightSamples(16);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        test::dumpPpm(path.stem().string(), image->rgba.data(), w, h);
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer hair = frame(stageWith("hair_chiang.usda", true));
    const gpu::Buffer lambert = frame(stageWith("hair_lambert.usda", false));
    std::vector<float> blank(static_cast<size_t>(w) * h * 4, 0.0F);
    gpu::BufferDesc desc;
    desc.bytes = blank.size() * sizeof(float);
    desc.elementBytes = 16;
    auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
    REQUIRE(blankBuffer);
    auto lit = render::compareHdr(*gpu->library, hair, *blankBuffer, w, h);
    auto differs = render::compareHdr(*gpu->library, hair, lambert, w, h);
    REQUIRE(lit);
    REQUIRE(differs);
    std::printf("  hair material on a curve: against blank relMSE %.2e; against Lambert relMSE %.2e\n", lit->relMse,
                differs->relMse);
    CHECK(lit->relMse > 0.1);
    CHECK(differs->relMse > 1e-2);
}

// Subdivision through Hydra: a catmullClark cube drawn at refine levels 0,
// 1 and 2. The face centre of the +z face is a vertex from level 1 on and
// its limit is one point, so the depth at the pixel looking straight at it
// is the same at levels 1 and 2 to a facet's sag (the limit projection),
// while the flat control face at level 0 sits nearer the camera: the
// surface bulges inward there, so the centre depth grows.
TEST_CASE("a catmullClark cube refines through Hydra, its limit the same at every level", "[usd][gpu][mesh][subdivision]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("subdiv_cube.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Cube\"\n{\n"
               "    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n"
               "    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3]\n"
               "    point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), "
               "(1, 1, 1), (-1, 1, 1)]\n"
               "    uniform token subdivisionScheme = \"catmullClark\"\n"
               "    color3f[] primvars:displayColor = [(0.7, 0.6, 0.4)] ( interpolation = \"constant\" )\n"
               "    double3 xformOp:translate = (0, 0, -6)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    std::array<float, 3> centreDepth{};
    std::array<uint32_t, 3> covered{};
    for (uint32_t level = 0; level < 3; ++level) {
        (*renderer)->setRefineLevel(level);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        centreDepth[level] = image->depth[(h / 2) * w + w / 2];
        test::dumpPpm("subdiv_cube_" + std::to_string(level), image->rgba.data(), w, h);
        for (const float d : image->depth) {
            covered[level] += d > 0.0F ? 1u : 0u;
        }
    }
    std::printf("  refine 0, 1, 2: centre depth %.5f %.5f %.5f; covered %u %u %u pixels\n",
                static_cast<double>(centreDepth[0]), static_cast<double>(centreDepth[1]),
                static_cast<double>(centreDepth[2]), covered[0], covered[1], covered[2]);
    CHECK(std::abs(centreDepth[0] - 5.0F) < 1e-4F);            // the control face at z = -5
    CHECK(centreDepth[1] > centreDepth[0] + 0.05F);           // the limit surface sits inside the cube
    // One surface whatever the level: the centre pixel's ray meets a facet
    // beside the centre vertex, so the two levels differ by a facet's sag
    // (0.004 here), not by a level's worth of motion (0.17).
    CHECK(std::abs(centreDepth[1] - centreDepth[2]) < 0.01F);
    CHECK(covered[1] < covered[0]);                           // the corners pull in
    CHECK(covered[2] < covered[0]);
}

// Coordinate systems through Hydra: UsdShadeCoordSysAPI binds a name on
// the mesh to an xformable prim; hdsi makes a coordSys prim under that
// target, the delegate takes the sprim, and the mesh's Sync reads its
// bindings with the target's transform -- the name and the translation
// arrive exactly. What a material does with them is not yet wired: this
// is the resolution the plan asked for, by the scene index's prim and not
// by parsing paths.
TEST_CASE("a coordinate system bound to a mesh resolves to its target's transform through Hydra",
          "[usd][gpu][mesh][coordSys]") {
    LRT_REQUIRE_GPU(gpu);
    const fs::path path = scratch("coordsys.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Xform \"Frame\"\n{\n"
               "    double3 xformOp:translate = (1, 2, 3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Mesh \"Square\" (\n    prepend apiSchemas = [\"CoordSysAPI:paint\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel coordSys:paint:binding = </Frame>\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto image = (*renderer)->render("/Camera", 0.0, 64, 48);
    if (!image) FAIL(image.error().toString());
    const std::vector<usd::CoordSysBinding> bindings = (*renderer)->coordSysBindings("/Square");
    std::printf("  %zu coordinate systems bound to /Square", bindings.size());
    for (const usd::CoordSysBinding& b : bindings) {
        const render::Vec3 t = b.toWorld.translation();
        std::printf("; '%s' at (%.1f, %.1f, %.1f)", b.name.c_str(), t.x, t.y, t.z);
    }
    std::printf("\n");
    REQUIRE(bindings.size() == 1);
    CHECK(bindings[0].name == "paint");
    const render::Vec3 t = bindings[0].toWorld.translation();
    CHECK(t.x == 1.0);
    CHECK(t.y == 2.0);
    CHECK(t.z == 3.0);
}

// MaterialX's transforms between spaces, on a mesh that is not at the origin,
// and to a coordinate system bound to it (M8.2). Each is checked against a
// graph that reaches the same value without the transform node: position in
// world space against object position carried object -> world, and so on;
// both are emitted, offset to keep them positive, and the frames compared.
// Before, genslang's transforms read world matrices nothing set -- identity
// -- and knew no coordinate system at all.
TEST_CASE("MaterialX transforms between object, world and a bound coordinate system match the spaces they name",
          "[usd][gpu][mesh][materials][coordSys]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    // A square tilted in object space (its normal is not along an axis), under
    // a rotation, a non-uniform scale and a translation; a frame translated
    // and scaled, bound as "paint".
    const auto stage = [&](const std::string& name, const std::string& graph) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Xform \"Frame\"\n{\n"
               "    double3 xformOp:translate = (1, 2, 3)\n"
               "    double3 xformOp:scale = (2, 2, 2)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n}\n"
               "def Mesh \"Square\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\", \"CoordSysAPI:paint\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -0.5, -0.2), (0.5, -0.5, 0.2), (0.5, 0.5, 0.2), (-0.5, 0.5, -0.2)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    double3 xformOp:translate = (0.2, -0.1, -5)\n"
               "    double3 xformOp:rotateXYZ = (10, 25, 5)\n"
               "    double3 xformOp:scale = (2.5, 1.5, 1)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\", \"xformOp:scale\"]\n"
               "    rel coordSys:paint:binding = </Frame>\n"
               "    rel material:binding = </Materials/Mat>\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface_unlit\"\n"
               "            color3f inputs:emission_color.connect = </Materials/Mat/Out.outputs:out>\n"
               "            token outputs:out\n        }\n"
            << graph
            << "        def Shader \"Out\"\n        {\n"
               "            uniform token info:id = \"ND_convert_vector3_color3\"\n"
               "            vector3f inputs:in.connect = </Materials/Mat/Offset.outputs:out>\n"
               "            color3f outputs:out\n        }\n"
               "    }\n}\n";
        return path;
    };
    // Graph pieces: a node named N of type ND_... with inputs.
    const auto node = [](const std::string& name, const std::string& id, const std::string& inputs) {
        return "        def Shader \"" + name + "\"\n        {\n            uniform token info:id = \"" + id + "\"\n" +
               inputs + "            vector3f outputs:out\n        }\n";
    };
    const auto offset = [&](const std::string& from, float by) {
        return node("Offset", "ND_add_vector3FA",
                    "            vector3f inputs:in1.connect = </Materials/Mat/" + from + ".outputs:out>\n"
                    "            float inputs:in2 = " + std::to_string(by) + "\n");
    };
    const auto position = [&](const char* name, const char* space) {
        return node(name, "ND_position_vector3", std::string("            string inputs:space = \"") + space + "\"\n");
    };
    const auto transform = [&](const char* id, const char* from, const char* fromSpace, const char* toSpace) {
        return node("Xf", id,
                    std::string("            vector3f inputs:in.connect = </Materials/Mat/") + from + ".outputs:out>\n" +
                        "            string inputs:fromspace = \"" + fromSpace + "\"\n" +
                        "            string inputs:tospace = \"" + toSpace + "\"\n");
    };
    struct Case {
        const char* what;
        std::string transformed;
        std::string direct;
    };
    const std::vector<Case> cases{
        {"object point to world", position("P", "object") + transform("ND_transformpoint_vector3", "P", "object", "world") + offset("Xf", 10.0F),
         position("P", "world") + offset("P", 10.0F)},
        {"world point to object", position("P", "world") + transform("ND_transformpoint_vector3", "P", "world", "object") + offset("Xf", 10.0F),
         position("P", "object") + offset("P", 10.0F)},
        {"world point to the bound 'paint'",
         position("P", "world") + transform("ND_transformpoint_vector3", "P", "world", "paint") + offset("Xf", 10.0F),
         position("P", "world") +
             node("Moved", "ND_subtract_vector3", "            vector3f inputs:in1.connect = </Materials/Mat/P.outputs:out>\n"
                                                  "            vector3f inputs:in2 = (1, 2, 3)\n") +
             node("Scaled", "ND_divide_vector3FA", "            vector3f inputs:in1.connect = </Materials/Mat/Moved.outputs:out>\n"
                                                   "            float inputs:in2 = 2\n") +
             offset("Scaled", 10.0F)},
        {"object normal to world",
         node("N", "ND_normal_vector3", "            string inputs:space = \"object\"\n") +
             transform("ND_transformnormal_vector3", "N", "object", "world") + offset("Xf", 2.0F),
         node("N", "ND_normal_vector3", "            string inputs:space = \"world\"\n") + offset("N", 2.0F)},
    };
    const uint32_t w = 160, h = 120;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    int k = 0;
    for (const Case& c : cases) {
        gpu::Buffer a = frame(stage("spaces_" + std::to_string(k) + "_transformed.usda", c.transformed));
        gpu::Buffer b = frame(stage("spaces_" + std::to_string(k) + "_direct.usda", c.direct));
        auto diff = render::compareHdr(*gpu->library, a, b, w, h);
        REQUIRE(diff);
        std::vector<float> blank(size_t{w} * h * 4, 0.0F);
        auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
        REQUIRE(empty);
        auto drawn = render::compareHdr(*gpu->library, a, *empty, w, h);
        REQUIRE(drawn);
        std::printf("  %-34s: relMSE %.2e, max relative %.2e (against blank %.2e)\n", c.what, diff->relMse,
                    diff->maxRelative, drawn->relMse);
        CHECK(drawn->relMse > 0.1);
        CHECK(diff->maxRelative < 1e-4);
        ++k;
    }
    // The normals agree to the bit, which is also what a check comparing a
    // thing with itself would say: the object normal left untransformed must
    // not agree with the world normal.
    gpu::Buffer untransformed = frame(stage("spaces_normal_untransformed.usda",
        node("N", "ND_normal_vector3", "            string inputs:space = \"object\"\n") + offset("N", 2.0F)));
    gpu::Buffer world = frame(stage("spaces_normal_world.usda",
        node("N", "ND_normal_vector3", "            string inputs:space = \"world\"\n") + offset("N", 2.0F)));
    auto control = render::compareHdr(*gpu->library, untransformed, world, w, h);
    REQUIRE(control);
    std::printf("  control, object normal untransformed : max relative %.2e against the world normal\n",
                control->maxRelative);
    CHECK(control->maxRelative > 1e-2);
}

// Render settings through Hydra (M10): a UsdRenderSettings prim with its
// products and vars reaches the delegate's renderSettings bprim once it is
// the scene's active one; `renderProducts` renders each product at its own
// resolution with its vars as the layers of one EXR, `includedPurposes` as
// the render tags. The check is the plan's, literal: every layer of the
// file, read back and uploaded, is bit for bit the AOV rendered on its own
// (countDifferent 0 words), and a second settings prim that includes guides
// covers more pixels than the first, which is what proves the purposes
// reached the tags and not a default.
TEST_CASE("a render settings prim's products come out as the AOVs rendered one at a time, bit for bit",
          "[usd][gpu][mesh][render-settings]") {
    LRT_REQUIRE_GPU(gpu);
    const fs::path path = scratch("render_settings.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "    color3f[] primvars:displayColor = [(0.2, 0.7, 0.3)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Guide\"\n{\n"
               "    uniform token purpose = \"guide\"\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(1, -2, -4), (3, -2, -4), (3, 2, -4), (1, 2, -4)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Settings\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        uniform token[] includedPurposes = [\"default\"]\n"
               "        rel products = </Render/Product>\n"
               "        int2 resolution = (96, 64)\n"
               "        int lrt:pathSamples = 3\n"
               "        token lrt:technique = \"raster\"\n"
               "    }\n"
               "    def RenderSettings \"WithGuides\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        uniform token[] includedPurposes = [\"default\", \"guide\"]\n"
               "        rel products = </Render/GuideProduct>\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def RenderProduct \"Product\"\n    {\n"
               "        token productName = \"product.exr\"\n"
               "        rel orderedVars = [</Render/Vars/beauty>, </Render/Vars/depth>, </Render/Vars/primId>, </Render/Vars/Neye>]\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def RenderProduct \"GuideProduct\"\n    {\n"
               "        token productName = \"guides.exr\"\n"
               "        rel orderedVars = [</Render/Vars/primId>]\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"Ci\"\n            token dataType = \"color3f\"\n        }\n"
               "        def RenderVar \"depth\"\n        {\n            string sourceName = \"z\"\n            token dataType = \"float\"\n        }\n"
               "        def RenderVar \"primId\"\n        {\n            string sourceName = \"primId\"\n            token dataType = \"int\"\n        }\n"
               "        def RenderVar \"Neye\"\n        {\n            string sourceName = \"Neye\"\n            token dataType = \"normal3f\"\n        }\n"
               "    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto info = (*renderer)->renderSettings("/Render/Settings");
    if (!info) FAIL(info.error().toString());
    std::printf("  /Render/Settings: %s, synced %u times, %zu products, purposes:", info->active ? "active" : "inactive",
                info->syncs, info->products.size());
    for (const std::string& p : info->includedPurposes) std::printf(" %s", p.c_str());
    std::printf("; camera %s; settings:", info->camera.c_str());
    for (const auto& [k, v] : info->settings) std::printf(" %s=%s", k.c_str(), v.c_str());
    std::printf("\n");
    REQUIRE(info->active);
    REQUIRE(info->products.size() == 1);
    CHECK(info->products[0].width == 96);
    CHECK(info->products[0].height == 64);
    CHECK(info->products[0].name == "product.exr");
    REQUIRE(info->products[0].vars.size() == 4);
    CHECK(info->products[0].vars[0].name == "beauty");
    CHECK(info->products[0].vars[0].sourceName == "Ci");
    CHECK(info->includedPurposes == std::vector<std::string>{"default"});
    CHECK(info->settings["lrt:pathSamples"] == "3");
    CHECK(info->settings["lrt:technique"] == "raster");

    const fs::path directory = path.parent_path();
    auto written = (*renderer)->renderProducts("/Render/Settings", 0.0, directory);
    if (!written) FAIL(written.error().toString());
    REQUIRE(written->size() == 1);
    CHECK(written->front() == directory / "product.exr");
    auto file = io::readExrChannels(written->front());
    if (!file) FAIL(file.error().toString());
    REQUIRE(file->width == 96);
    REQUIRE(file->height == 64);
    std::printf("  product.exr: %zu channels:", file->channels.size());
    for (const io::ExrChannelData& c : file->channels) std::printf(" %s", c.name.c_str());
    std::printf("\n");
    REQUIRE(file->channels.size() == 4 + 1 + 1 + 3);
    const auto channel = [&](const std::string& name) -> const io::ExrChannelData* {
        for (const io::ExrChannelData& c : file->channels) {
            if (c.name == name) return &c;
        }
        return nullptr;
    };
    for (const char* name : {"beauty.R", "beauty.G", "beauty.B", "beauty.A", "Z", "primId", "Neye.x", "Neye.y", "Neye.z"}) {
        INFO(name);
        REQUIRE(channel(name) != nullptr);
    }
    CHECK(channel("primId")->type == io::ExrChannelType::Uint);
    CHECK(channel("Z")->type == io::ExrChannelType::Float);

    // The same AOVs rendered one at a time, through the same Hydra buffers.
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    const auto upload = [&](std::span<const uint32_t> words, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = words.size() * 4;
        desc.elementBytes = 4;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, words.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const auto differing = [&](const std::string& name, std::span<const uint32_t> expected) {
        const io::ExrChannelData* c = channel(name);
        REQUIRE(c != nullptr);
        REQUIRE(c->words.size() == pixels);
        gpu::Buffer a = upload(c->words, "exr.layer");
        gpu::Buffer b = upload(expected, "exr.alone");
        auto count = render::countDifferent(*gpu->library, a, b, static_cast<uint32_t>(pixels));
        REQUIRE(count);
        return *count;
    };
    const auto plane = [&](const std::vector<uint8_t>& bytes, size_t components, size_t c) {
        std::vector<uint32_t> words(pixels);
        for (size_t p = 0; p < pixels; ++p) {
            std::memcpy(&words[p], bytes.data() + (p * components + c) * 4, 4);
        }
        return words;
    };
    (*renderer)->requestOutputs({"primId"});
    auto alone = (*renderer)->render("/Camera", 0.0, w, h);
    if (!alone) FAIL(alone.error().toString());
    auto primId = (*renderer)->mappedOutput("primId");
    REQUIRE(primId);
    const std::vector<uint32_t> ids = plane(*primId, 1, 0);
    std::vector<uint32_t> depth(pixels);
    std::memcpy(depth.data(), alone->depth.data(), pixels * 4);
    std::vector<uint32_t> colour[4];
    for (size_t c = 0; c < 4; ++c) {
        colour[c].resize(pixels);
        for (size_t p = 0; p < pixels; ++p) std::memcpy(&colour[c][p], alone->rgba.data() + p * 4 + c, 4);
    }
    (*renderer)->requestOutputs({"Neye"});
    REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
    auto eye = (*renderer)->mappedOutput("Neye");
    REQUIRE(eye);
    const uint64_t offId = differing("primId", ids);
    const uint64_t offZ = differing("Z", depth);
    const uint64_t offR = differing("beauty.R", colour[0]);
    const uint64_t offG = differing("beauty.G", colour[1]);
    const uint64_t offB = differing("beauty.B", colour[2]);
    const uint64_t offA = differing("beauty.A", colour[3]);
    const uint64_t offNx = differing("Neye.x", plane(*eye, 3, 0));
    const uint64_t offNy = differing("Neye.y", plane(*eye, 3, 1));
    const uint64_t offNz = differing("Neye.z", plane(*eye, 3, 2));
    // Coverage of the ids plane: how many pixels the square took.
    const std::vector<uint32_t> cleared(pixels, 0xFFFFFFFFu);
    const uint64_t covered = differing("primId", cleared);
    std::printf("  layers against the AOVs alone: primId %llu words off, Z %llu, beauty %llu %llu %llu %llu, Neye %llu %llu %llu;"
                " %llu of %zu pixels covered\n",
                static_cast<unsigned long long>(offId), static_cast<unsigned long long>(offZ),
                static_cast<unsigned long long>(offR), static_cast<unsigned long long>(offG),
                static_cast<unsigned long long>(offB), static_cast<unsigned long long>(offA),
                static_cast<unsigned long long>(offNx), static_cast<unsigned long long>(offNy),
                static_cast<unsigned long long>(offNz), static_cast<unsigned long long>(covered), pixels);
    CHECK(offId == 0);
    CHECK(offZ == 0);
    CHECK(offR == 0);
    CHECK(offG == 0);
    CHECK(offB == 0);
    CHECK(offA == 0);
    CHECK(offNx == 0);
    CHECK(offNy == 0);
    CHECK(offNz == 0);
    CHECK(covered > 500);

    // The settings that include guides draw the guide square too.
    auto guides = (*renderer)->renderProducts("/Render/WithGuides", 0.0, directory);
    if (!guides) FAIL(guides.error().toString());
    REQUIRE(guides->size() == 1);
    auto guideFile = io::readExrChannels(guides->front());
    if (!guideFile) FAIL(guideFile.error().toString());
    REQUIRE(guideFile->channels.size() == 1);
    REQUIRE(guideFile->channels[0].words.size() == pixels);
    gpu::Buffer guideIds = upload(guideFile->channels[0].words, "exr.guides");
    gpu::Buffer blank = upload(cleared, "exr.cleared");
    auto guideCovered = render::countDifferent(*gpu->library, guideIds, blank, static_cast<uint32_t>(pixels));
    REQUIRE(guideCovered);
    auto second = (*renderer)->renderSettings("/Render/WithGuides");
    REQUIRE(second);
    std::printf("  with guides (%s): %llu pixels covered against %llu without\n", second->active ? "active" : "inactive",
                static_cast<unsigned long long>(*guideCovered), static_cast<unsigned long long>(covered));
    CHECK(second->active);
    CHECK(*guideCovered > covered + 200);
    // And the first is no longer the active one.
    auto first = (*renderer)->renderSettings("/Render/Settings");
    REQUIRE(first);
    CHECK(first->active);   // asking makes it active again
}

// Material binding purposes: a square bound three ways -- all-purpose to
// red, `material:binding:full` to blue, `material:binding:preview` to green.
// The delegate resolved Hydra's default purpose, "preview", so a
// production binding was never seen (the shader ball's walls are bound
// `full` alone and drew their fallback). It now resolves "full" unless a
// settings prim's `materialBindingPurposes` names another, with the
// all-purpose binding behind either.
TEST_CASE("a mesh takes the material bound for the purpose render settings name, full by default",
          "[usd][gpu][mesh][materials][purposes]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("binding_purposes.usda");
    {
        std::ofstream out(path);
        std::string square = kSquareStage;
        const std::string binding = "    rel material:binding = </Materials/Mat>\n";
        square.replace(square.find(binding), binding.size(),
                       "    rel material:binding = </Materials/Red>\n"
                       "    rel material:binding:full = </Materials/Blue>\n"
                       "    rel material:binding:preview = </Materials/Green>\n");
        out << square << "def Scope \"Materials\"\n{\n";
        const auto material = [&](const char* name, const char* colour) {
            out << "    def Material \"" << name << "\"\n    {\n"
                << "        token outputs:surface.connect = </Materials/" << name << "/Preview.outputs:surface>\n"
                << "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                << "            color3f inputs:diffuseColor = " << colour << "\n"
                << "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n    }\n";
        };
        material("Red", "(0.8, 0.1, 0.1)");
        material("Green", "(0.1, 0.8, 0.1)");
        material("Blue", "(0.1, 0.1, 0.8)");
        out << "}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Preview\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        uniform token[] materialBindingPurposes = [\"preview\", \"\"]\n"
               "        rel products = </Render/Product>\n"
               "        int2 resolution = (64, 48)\n"
               "    }\n"
               "    def RenderProduct \"Product\"\n    {\n"
               "        token productName = \"purposes.exr\"\n"
               "        rel orderedVars = [</Render/Vars/beauty>]\n"
               "        int2 resolution = (64, 48)\n"
               "    }\n"
               "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"Ci\"\n"
               "            token dataType = \"color3f\"\n        }\n"
               "    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const auto centre = [&](const char* label) {
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* c = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  %-34s: centre %.3f %.3f %.3f\n", label, double(c[0]), double(c[1]), double(c[2]));
        // Which of the three: the channel that is high.
        return c[0] > 0.5F ? 'r' : c[1] > 0.5F ? 'g' : c[2] > 0.5F ? 'b' : '?';
    };
    CHECK(centre("default (full)") == 'b');
    (*renderer)->setMaterialBindingPurposes({"preview", ""});
    CHECK(centre("preview, then all-purpose") == 'g');
    (*renderer)->setMaterialBindingPurposes({""});
    CHECK(centre("all-purpose alone") == 'r');
    (*renderer)->setMaterialBindingPurposes({});
    CHECK(centre("back to the default") == 'b');
    // From a settings prim: its products are drawn with its purposes, and
    // the renders after keep them.
    auto written = (*renderer)->renderProducts("/Render/Preview", 0.0, path.parent_path());
    if (!written) FAIL(written.error().toString());
    CHECK(centre("after /Render/Preview's products") == 'g');
}

// A render product's disableMotionBlur and disableDepthOfField (M10): read
// since the products were, applied now. One stage -- a square sliding under
// a shutter open about the frame, seen through a lens focused in front of it
// -- and three products: both switches on, which must be bit for bit the
// same stage authored with neither shutter nor lens; and each switch alone,
// which must not be, since the other effect is still drawn. The switches
// hold for their product and nothing after.
TEST_CASE("a render product's disableMotionBlur and disableDepthOfField draw it without either",
          "[usd][gpu][mesh][path][render-settings][camera][motion]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    const auto stage = [&](const char* name, bool effects) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)]\n"
               "    double3 xformOp:translate.timeSamples = {\n        0: (-1, 0, 0),\n        1: (1, 0, 0),\n    }\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (effects) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n"
                   "    float fStop = 8\n    float focusDistance = 2.5\n";
        }
        out << "}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Settings\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        rel products = [</Render/Plain>, </Render/Lens>, </Render/Blur>]\n"
               "        int2 resolution = (96, 64)\n"
               "        token lrt:technique = \"rt\"\n"
               "        int lrt:pathSamples = 16\n        int lrt:pathTotal = 16\n        int lrt:pathBounces = 0\n"
               "        int lrt:motionBuckets = 8\n"
               "    }\n";
        const auto product = [&](const char* productName, bool noBlur, bool noLens) {
            out << "    def RenderProduct \"" << productName << "\"\n    {\n"
                << "        token productName = \"switches_" << productName << ".exr\"\n"
                << "        rel orderedVars = </Render/Vars/beauty>\n"
                   "        int2 resolution = (96, 64)\n"
                << "        uniform bool disableMotionBlur = " << (noBlur ? 1 : 0) << "\n"
                << "        uniform bool disableDepthOfField = " << (noLens ? 1 : 0) << "\n    }\n";
        };
        product("Plain", true, true);
        product("Lens", true, false);
        product("Blur", false, true);
        out << "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"Ci\"\n"
               "            token dataType = \"color3f\"\n        }\n    }\n}\n";
        return path;
    };
    // The reference: neither shutter nor lens, drawn as the settings say.
    std::vector<uint32_t> reference[3];
    {
        auto renderer = usd::StageRenderer::open(stage("switches_reference.usda", false));
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(16);
        (*renderer)->setPathTotal(16);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        for (size_t c = 0; c < 3; ++c) {
            reference[c].resize(pixels);
            for (size_t p = 0; p < pixels; ++p) std::memcpy(&reference[c][p], image->rgba.data() + p * 4 + c, 4);
        }
    }
    const fs::path path = stage("switches.usda", true);
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto written = (*renderer)->renderProducts("/Render/Settings", 0.5, path.parent_path());
    if (!written) FAIL(written.error().toString());
    REQUIRE(written->size() == 3);
    const auto differing = [&](const fs::path& file) {
        auto read = io::readExrChannels(file);
        if (!read) FAIL(read.error().toString());
        uint64_t off = 0;
        const char* names[] = {"beauty.R", "beauty.G", "beauty.B"};
        for (size_t c = 0; c < 3; ++c) {
            const io::ExrChannelData* found = nullptr;
            for (const io::ExrChannelData& data : read->channels) {
                if (data.name == names[c]) found = &data;
            }
            REQUIRE(found != nullptr);
            REQUIRE(found->words.size() == pixels);
            auto a = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, found->words, "product");
            auto b = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, reference[c], "reference");
            REQUIRE(a);
            REQUIRE(b);
            auto count = render::countDifferent(*gpu->library, *a, *b, static_cast<uint32_t>(pixels));
            REQUIRE(count);
            off += *count;
        }
        return off;
    };
    const uint64_t plain = differing((*written)[0]);
    const uint64_t lens = differing((*written)[1]);
    const uint64_t blur = differing((*written)[2]);
    std::printf("  words off the plain stage's frame: both switched off %llu, lens drawn %llu, blur drawn %llu (of %zu)\n",
                static_cast<unsigned long long>(plain), static_cast<unsigned long long>(lens),
                static_cast<unsigned long long>(blur), pixels * 3);
    CHECK(plain == 0);
    CHECK(lens > 300);
    CHECK(blur > 300);
    // And after the products, the stage's own effects again.
    (*renderer)->setPathSamples(16);
    (*renderer)->setPathTotal(16);
    (*renderer)->setPathBounces(0);
    (*renderer)->setMotionBuckets(8);
    auto after = (*renderer)->render("/Camera", 0.5, w, h, "rt");
    if (!after) FAIL(after.error().toString());
    std::vector<uint32_t> red(pixels);
    for (size_t p = 0; p < pixels; ++p) std::memcpy(&red[p], after->rgba.data() + p * 4, 4);
    auto a = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, red, "after");
    auto b = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, reference[0], "reference");
    REQUIRE(a);
    REQUIRE(b);
    auto again = render::countDifferent(*gpu->library, *a, *b, static_cast<uint32_t>(pixels));
    REQUIRE(again);
    std::printf("  a render after the products: %llu red words off the plain frame\n",
                static_cast<unsigned long long>(*again));
    CHECK(*again > 100);
}

// A dome seen by the camera is in its light group -- C.*<L.'NAME'> matches
// the camera's ray meeting the dome with `.*` empty -- and the camera's
// exposure scales the groups as it scales the beauty. Before, both left
// the groups short of the beauty: the sky's pixels in no group, and every
// pixel by the exposure's factor.
TEST_CASE("a dome's background is in its light group, and the exposure in every group, raster and path traced",
          "[usd][gpu][mesh][lights][light-groups][dome]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("light_groups_dome.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
               "    string lrt:lightGroup = \"key\"\n"
               "    double3 xformOp:translate = (-2, 2, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def DomeLight \"Sky\"\n{\n"
               "    float inputs:intensity = 0.5\n    color3f inputs:color = (0.4, 0.6, 1)\n"
               "    bool inputs:shadow:enable = 0\n"
               "    string lrt:lightGroup = \"sky\"\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    float exposure = 1\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -8\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto sumKernel = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_group_check", "lightGroupSum");
    if (!sumKernel) FAIL(sumKernel.error().toString());
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    struct Counts {
        uint32_t off, emptyNonzero, covered, checked;
        float    worst;
    };
    const auto run = [&](const char* technique) {
        (*renderer)->requestOutputs({"lightGroup:key", "lightGroup:sky", "lightGroup:none"});
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        auto beauty = (*renderer)->mappedOutput("color");
        REQUIRE(beauty);
        std::vector<uint8_t> planes;
        for (const char* group : {"lightGroup:key", "lightGroup:sky", "lightGroup:none"}) {
            auto plane = (*renderer)->mappedOutput(group);
            REQUIRE(plane);
            planes.insert(planes.end(), plane->begin(), plane->end());
        }
        REQUIRE(beauty->size() == pixels * 16);
        auto b = gpu::Buffer::fromSpan<uint8_t>(*gpu->device, *beauty, "dome.beauty");
        auto p = gpu::Buffer::fromSpan<uint8_t>(*gpu->device, planes, "dome.planes");
        REQUIRE(b);
        REQUIRE(p);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "dome.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "dome.worst");
        gpu::CommandBatch batch(*gpu->device);
        sumKernel->dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) {
            c["check"]["pixels"].setData(static_cast<uint32_t>(pixels));
            c["check"]["groups"].setData(uint32_t{3});
            c["check"]["empty"].setData(uint32_t{2});
            c["check"]["tolerance"].setData(1.0e-4F);
            c["beauty"].setBinding(b->rhi());
            c["planes"].setBinding(p->rhi());
            c["counts"].setBinding(counts.rhi());
            c["worst"].setBinding(worst.rhi());
        });
        REQUIRE(batch.submit(true));
        Counts out{};
        REQUIRE(counts.read(*gpu->device, 0, 16, &out));
        REQUIRE(worst.read(*gpu->device, 0, 4, &out.worst));
        std::printf("  %-6s: %u of %u pixels covered (floor and sky); groups summed off the beauty at %u (worst %.2e "
                    "relative); the empty group nonzero at %u\n",
                    technique, out.covered, out.checked, out.off, static_cast<double>(out.worst), out.emptyNonzero);
        return out;
    };
    const Counts raster = run("raster");
    CHECK(raster.covered == pixels);
    CHECK(raster.off == 0);
    CHECK(raster.emptyNonzero == 0);
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathTotal(8);
    const Counts traced = run("rt");
    CHECK(traced.covered == pixels);
    CHECK(traced.off == 0);
    CHECK(traced.emptyNonzero == 0);
}

// Light groups (M10): a light's `lrt:lightGroup` puts its direct light,
// at every bounce, into a plane of its own beside the beauty, asked for as
// the "lightGroup:NAME" output -- or, through a render product, as the var
// with the light path expression C.*<L.'NAME'>. The plan's check: the
// groups summed are the beauty, and a group no light belongs to is exactly
// zero -- by a kernel over every pixel, for the raster's direct light and
// the path tracer alike.
TEST_CASE("light groups sum to the beauty and an empty group is zero, raster and path traced",
          "[usd][gpu][mesh][lights][light-groups]") {
    LRT_REQUIRE_GPU(gpu);
    const fs::path path = scratch("light_groups.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
               "    string lrt:lightGroup = \"key\"\n"
               "    double3 xformOp:translate = (-2, 2, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def SphereLight \"Key2\"\n{\n"
               "    float inputs:intensity = 20\n    float inputs:radius = 0.3\n"
               "    color3f inputs:color = (1, 0.6, 0.3)\n"
               "    string lrt:lightGroup = \"key\"\n"
               "    double3 xformOp:translate = (3, 2.5, -9)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def SphereLight \"Fill\"\n{\n"
               "    float inputs:intensity = 10\n    float inputs:radius = 0.5\n"
               "    color3f inputs:color = (0.3, 0.5, 1)\n"
               "    string ri:light:lightGroup = \"fill\"\n"
               "    double3 xformOp:translate = (0, 3, -4)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -18\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Traced\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        rel products = </Render/Product>\n"
               "        int2 resolution = (96, 64)\n"
               "        token lrt:technique = \"rt\"\n"
               "        int lrt:pathSamples = 4\n"
               "        int lrt:pathTotal = 16\n"
               "    }\n"
               "    def RenderProduct \"Product\"\n    {\n"
               "        token productName = \"groups.exr\"\n"
               "        rel orderedVars = [</Render/Vars/beauty>, </Render/Vars/key>, </Render/Vars/fill>, </Render/Vars/rim>]\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"color\"\n        }\n"
               "        def RenderVar \"key\"\n        {\n            string sourceName = \"C.*<L.'key'>\"\n            token sourceType = \"lpe\"\n        }\n"
               "        def RenderVar \"fill\"\n        {\n            string sourceName = \"C.*<L.'fill'>\"\n            token sourceType = \"lpe\"\n        }\n"
               "        def RenderVar \"rim\"\n        {\n            string sourceName = \"C.*<L.'rim'>\"\n            token sourceType = \"lpe\"\n        }\n"
               "    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto sumKernel = gpu::ComputeKernel::create(*gpu->library, "lrt/test/light_group_check", "lightGroupSum");
    if (!sumKernel) FAIL(sumKernel.error().toString());
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    const auto upload = [&](const void* bytes, size_t count, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = count * 16;
        desc.elementBytes = 16;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, bytes);
        REQUIRE(made);
        return std::move(*made);
    };
    // The check over the beauty and three planes: key, fill, rim (empty).
    struct Counts {
        uint32_t off, emptyNonzero, covered, checked;
        float    worst;
    };
    const auto check = [&](const std::vector<uint8_t>& beauty, const std::vector<uint8_t>& planes) {
        REQUIRE(beauty.size() == pixels * 16);
        REQUIRE(planes.size() == pixels * 16 * 3);
        gpu::Buffer b = upload(beauty.data(), pixels, "groups.beauty");
        gpu::Buffer p = upload(planes.data(), pixels * 3, "groups.planes");
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "groups.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "groups.worst");
        gpu::CommandBatch batch(*gpu->device);
        sumKernel->dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) {
            c["check"]["pixels"].setData(static_cast<uint32_t>(pixels));
            c["check"]["groups"].setData(uint32_t{3});
            c["check"]["empty"].setData(uint32_t{2});
            c["check"]["tolerance"].setData(1.0e-4F);
            c["beauty"].setBinding(b.rhi());
            c["planes"].setBinding(p.rhi());
            c["counts"].setBinding(counts.rhi());
            c["worst"].setBinding(worst.rhi());
        });
        REQUIRE(batch.submit(true));
        Counts out{};
        REQUIRE(counts.read(*gpu->device, 0, 16, &out));
        REQUIRE(worst.read(*gpu->device, 0, 4, &out.worst));
        return out;
    };
    const auto planesOf = [&](const char* technique) {
        (*renderer)->requestOutputs({"lightGroup:key", "lightGroup:fill", "lightGroup:rim"});
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        auto beauty = (*renderer)->mappedOutput("color");
        REQUIRE(beauty);
        std::vector<uint8_t> planes;
        for (const char* group : {"lightGroup:key", "lightGroup:fill", "lightGroup:rim"}) {
            auto plane = (*renderer)->mappedOutput(group);
            REQUIRE(plane);
            planes.insert(planes.end(), plane->begin(), plane->end());
        }
        {
            const std::string prefix = std::string("light_groups_") + technique;
            test::dumpPpm((prefix + "_beauty").c_str(), image->rgba.data(), w, h);
            std::vector<float> plane(pixels * 4);
            for (size_t g = 0; g < 3; ++g) {
                std::memcpy(plane.data(), planes.data() + g * pixels * 16, pixels * 16);
                test::dumpPpm((prefix + "_" + std::to_string(g)).c_str(), plane.data(), w, h);
            }
        }
        return check(*beauty, planes);
    };
    const Counts raster = planesOf("raster");
    std::printf("  raster: %u of %u pixels covered; groups summed off the beauty at %u (worst %.2e relative); the empty "
                "group nonzero at %u\n",
                raster.covered, raster.checked, raster.off, static_cast<double>(raster.worst), raster.emptyNonzero);
    CHECK(raster.covered > 1000);
    CHECK(raster.off == 0);
    CHECK(raster.emptyNonzero == 0);
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathTotal(16);
    const Counts traced = planesOf("rt");
    std::printf("  path traced: %u of %u pixels covered; groups summed off the beauty at %u (worst %.2e relative); the "
                "empty group nonzero at %u\n",
                traced.covered, traced.checked, traced.off, static_cast<double>(traced.worst), traced.emptyNonzero);
    CHECK(traced.covered > 1000);
    CHECK(traced.off == 0);
    CHECK(traced.emptyNonzero == 0);

    // And through a render product: the light path expressions name the
    // groups, and the file's layers are the same planes.
    const fs::path directory = path.parent_path();
    auto written = (*renderer)->renderProducts("/Render/Traced", 0.0, directory);
    if (!written) FAIL(written.error().toString());
    REQUIRE(written->size() == 1);
    auto file = io::readExrChannels(written->front());
    if (!file) FAIL(file.error().toString());
    std::printf("  groups.exr: %zu channels:", file->channels.size());
    for (const io::ExrChannelData& c : file->channels) std::printf(" %s", c.name.c_str());
    std::printf("\n");
    REQUIRE(file->channels.size() == 16);
    const auto channel = [&](const std::string& name) -> const io::ExrChannelData* {
        for (const io::ExrChannelData& c : file->channels) {
            if (c.name == name) return &c;
        }
        return nullptr;
    };
    std::vector<uint8_t> beauty(pixels * 16);
    std::vector<uint8_t> planes(pixels * 16 * 3);
    const auto gather = [&](std::vector<uint8_t>& into, size_t plane, const std::string& prefix) {
        for (size_t c = 0; c < 4; ++c) {
            const io::ExrChannelData* data = channel(prefix + "." + "RGBA"[c]);
            REQUIRE(data != nullptr);
            for (size_t p = 0; p < pixels; ++p) {
                std::memcpy(into.data() + (plane * pixels + p) * 16 + c * 4, &data->words[p], 4);
            }
        }
    };
    gather(beauty, 0, "beauty");
    gather(planes, 0, "key");
    gather(planes, 1, "fill");
    gather(planes, 2, "rim");
    const Counts product = check(beauty, planes);
    std::printf("  the product's layers: %u of %u pixels covered; groups summed off the beauty at %u (worst %.2e); the "
                "empty group nonzero at %u\n",
                product.covered, product.checked, product.off, static_cast<double>(product.worst),
                product.emptyNonzero);
    CHECK(product.covered > 1000);
    CHECK(product.off == 0);
    CHECK(product.emptyNonzero == 0);
}

// Shadow rays over a floor nothing occludes must change nothing: the raster
// technique's shading with the lights' shadows on is bit for bit the
// shading with them off. This is the regression that caught a Metal
// compiler problem: with a local copy of the lobe stack live across the
// shadow ray's intersector call, the shadowed kernel wrote rows of garbage
// in blocks of half a threadgroup (MaterialShading.cpp says the rest).
TEST_CASE("shadow rays over an unoccluded floor change nothing through Hydra, raster technique",
          "[usd][gpu][mesh][lights][shadows]") {
    LRT_REQUIRE_GPU(gpu);
    const auto stage = [&](bool shadows) {
        const fs::path path = scratch(shadows ? "floor_shadows_on.usda" : "floor_shadows_off.usda");
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
            << (shadows ? "" : "    bool inputs:shadow:enable = 0\n")
            << "    double3 xformOp:translate = (-2, 2, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def SphereLight \"Fill\"\n{\n"
               "    float inputs:intensity = 10\n    float inputs:radius = 0.5\n"
               "    color3f inputs:color = (0.3, 0.5, 1)\n"
            << (shadows ? "" : "    bool inputs:shadow:enable = 0\n")
            << "    double3 xformOp:translate = (0, 3, -4)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -18\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return path;
    };
    const uint32_t w = 96, h = 64;
    const auto render = [&](bool shadows) {
        auto renderer = usd::StageRenderer::open(stage(shadows));
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * 4;
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    // Three shadowed renders: the corruption was nondeterministic, and one
    // clean frame proved nothing.
    gpu::Buffer off = render(false);
    for (int k = 0; k < 3; ++k) {
        gpu::Buffer on = render(true);
        auto words = render::countDifferent(*gpu->library, on, off, w * h * 4);
        REQUIRE(words);
        std::printf("  shadows on against off, run %d: %llu of %u words differ\n", k + 1,
                    static_cast<unsigned long long>(*words), w * h * 4);
        CHECK(*words == 0);
    }
}


// A light's shadowLink collection through Hydra: an occluder between a
// sphere light and a floor, both out of the camera's view, so the only
// thing the occluder can change in the image is the shadow. With the
// collection left to include everything the floor is shadowed; with a
// membership expression naming the floor alone the occluder casts nothing,
// and the frame is bit for bit the frame without the occluder -- under
// the raster's shading and under the path tracer's direct light alike.
TEST_CASE("a UsdLux light's shadowLink collection decides what casts its shadow",
          "[usd][gpu][mesh][lights][linking][shadows]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    enum class Occluder { None, Casting, Unlinked };
    const auto stage = [&](Occluder occluder) {
        const char* names[] = {"shadow_link_none.usda", "shadow_link_casting.usda", "shadow_link_unlinked.usda"};
        const fs::path path = scratch(names[int(occluder)]);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n";
        if (occluder != Occluder::None) {
            // Above the top of the frame (it ends 0.9 degrees above the
            // horizon; this is 3.6 degrees up), below the light.
            out << "def Mesh \"Occluder\"\n{\n"
                   "    int[] faceVertexCounts = [4]\n"
                   "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-1, 2, -7), (1, 2, -7), (1, 2, -9), (-1, 2, -9)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n}\n";
        }
        out << "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
               "    double3 xformOp:translate = (0, 3, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        if (occluder == Occluder::Unlinked) {
            out << "    uniform token collection:shadowLink:mode = \"expression\"\n"
                   "    uniform pathExpression collection:shadowLink:membershipExpression = \"/Floor\"\n";
        }
        out << "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -18\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return path;
    };
    const uint32_t w = 96, h = 64;
    for (const char* technique : {"raster", "rt"}) {
        const auto render = [&](Occluder occluder) {
            auto renderer = usd::StageRenderer::open(stage(occluder));
            if (!renderer) FAIL(renderer.error().toString());
            (*renderer)->setPathSamples(16);
            (*renderer)->setPathBounces(0);
            auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
            if (!image) FAIL(image.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * 4;
            desc.elementBytes = 16;
            auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(made);
            return std::move(*made);
        };
        gpu::Buffer none = render(Occluder::None);
        gpu::Buffer casting = render(Occluder::Casting);
        gpu::Buffer unlinked = render(Occluder::Unlinked);
        auto shadowed = render::countDifferent(*gpu->library, casting, none, w * h * 4);
        auto ignored = render::countDifferent(*gpu->library, unlinked, none, w * h * 4);
        REQUIRE(shadowed);
        REQUIRE(ignored);
        std::printf("  %-6s: occluder in the shadow link changes %llu words, outside it %llu, of %u\n", technique,
                    static_cast<unsigned long long>(*shadowed), static_cast<unsigned long long>(*ignored), w * h * 4);
        CHECK(*shadowed > 200);
        CHECK(*ignored == 0);
    }
}

// Volumes through Hydra (M9): a UsdVolVolume whose density field is a
// UsdVolOpenVDBAsset, between the camera and a sun-lit plane, rendered by
// the rt technique. The volume's constant primvars make it absorb
// (lrt:albedo 0); the sun casts no shadows. Judged as the technique-level
// test judges it: each pixel's ratio against the stage without the volume
// on Beer-Lambert along its own ray, within five binomial deviations of the
// path count, and the pixels outside the box unchanged.
TEST_CASE("a UsdVol volume with an OpenVDB field absorbs as Beer-Lambert says through Hydra",
          "[usd][gpu][mesh][volume]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path vdb = scratch("hydra_box.vdb");
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.5F};
    REQUIRE(io::writeVdbBoxes(vdb, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    const auto stage = [&](bool volume) {
        const fs::path path = scratch(volume ? "volume_on.usda" : "volume_off.usda");
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Plane\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -4), (4, -4, -4), (4, 4, -4), (-4, 4, -4)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DistantLight \"Sun\"\n{\n"
               "    float inputs:intensity = 3\n    float inputs:angle = 0\n"
               "    bool inputs:shadow:enable = 0\n}\n";
        if (volume) {
            out << "def Volume \"Box\"\n{\n"
                   "    double3 xformOp:translate = (0, -1.6, -3.8)\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
                   "    rel field:density = </Box/density>\n"
                   "    float primvars:lrt:densityScale = 1\n"
                   "    color3f primvars:lrt:albedo = (0, 0, 0)\n"
                   "    def OpenVDBAsset \"density\"\n    {\n"
                   "        asset filePath = @" << vdb.string() << "@\n"
                   "        token fieldName = \"density\"\n    }\n}\n";
        }
        return path;
    };
    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    constexpr uint32_t kPaths = 1024;
    const auto render = [&](bool volume) {
        auto renderer = usd::StageRenderer::open(stage(volume));
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(256);
        (*renderer)->setPathBounces(0);
        (*renderer)->setPathTotal(kPaths);
        auto image = (*renderer)->render(camera, 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * 4;
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        if (volume) test::dumpPpm("volume_hydra", image->rgba.data(), w, h);
        return std::move(*made);
    };
    gpu::Buffer without = render(false);
    gpu::Buffer with = render(true);
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/volume_render_check", "volumeRenderCheck");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 5, "volume.hydra.counts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 4, "volume.hydra.sums");
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
            const float towardLight[3] = {0.0F, 0.0F, 1.0F};
            c["check"]["boxMin"].setData(boxMin, sizeof(boxMin));
            c["check"]["boxMax"].setData(boxMax, sizeof(boxMax));
            c["check"]["towardLight"].setData(towardLight, sizeof(towardLight));
            c["check"]["planeZ"].setData(-4.0F);
            c["check"]["sigma"].setData(0.5F);
            c["check"]["paths"].setData(static_cast<float>(kPaths));
            c["check"]["deviations"].setData(5.0F);
            c["check"]["shadows"].setData(uint32_t{0});
            c["with"].setBinding(with.rhi());
            c["without"].setBinding(without.rhi());
            c["counts"].setBinding(counts.rhi());
            c["zBits"].setBinding(sums.rhi());
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t n[5] = {};
    float s[4] = {};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
    REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
    const double meanSquare = n[0] > 0 ? s[0] / n[0] : 0.0;
    std::printf("  through Hydra: %u pixels through the box, %u beyond 5 binomial deviations, mean z^2 %.3f, mean "
                "ratio %.4f against %.4f; %u outside, %u changed; %u uncovered\n",
                n[0], n[1], meanSquare, n[0] ? s[1] / n[0] : 0.0F, n[0] ? s[2] / n[0] : 0.0F, n[2], n[3], n[4]);
    CHECK(n[0] > 3000);
    CHECK(n[1] == 0);
    CHECK(meanSquare > 0.7);
    CHECK(meanSquare < 1.3);
    CHECK(n[2] > 3000);
    CHECK(n[3] == 0);
}


// A frame of volumes alone (M9): no mesh, a medium that scatters everything
// and absorbs nothing, under a dome of radiance 1. The traced technique
// takes the mesh layer for the volume even with no mesh, every sample walks
// its camera ray, and a pixel the medium touched reads the dome's radiance
// over its opacity -- judged as the technique-level furnace is, within five
// standard errors measured from the pixels' spread.
TEST_CASE("a frame of volumes alone is path traced through Hydra, and an albedo-one medium reads the dome",
          "[usd][gpu][volume][furnace]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path vdb = scratch("hydra_furnace.vdb");
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.5F};
    REQUIRE(io::writeVdbBoxes(vdb, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    const fs::path path = scratch("volume_alone.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Volume \"Cloud\"\n{\n"
               "    double3 xformOp:translate = (-1.6, -1.6, -5)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    rel field:density = </Cloud/density>\n"
               "    float primvars:lrt:densityScale = 0.2\n"
               "    color3f primvars:lrt:albedo = (1, 1, 1)\n"
               "    def OpenVDBAsset \"density\"\n    {\n"
               "        asset filePath = @" << vdb.string() << "@\n"
               "        token fieldName = \"density\"\n    }\n}\n";
    }
    const uint32_t w = 81;
    const uint32_t h = 61;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setPathSamples(64);
    (*renderer)->setPathBounces(32);
    (*renderer)->setPathTotal(256);
    auto image = (*renderer)->render(camera, 0.0, w, h, "rt");
    if (!image) FAIL(image.error().toString());
    test::dumpPpm("volume_alone_hydra", image->rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * 4;
    desc.elementBytes = 16;
    auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(frame);
    auto made = gpu::ComputeKernel::create(*gpu->library, "lrt/test/volume_render_check", "volumeFurnaceCheck");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "furnace.hydra.counts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 2, "furnace.hydra.sums");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
            c["furnace"]["pixels"].setData(w * h);
            c["furnace"]["minAlpha"].setData(0.25F);
            c["with"].setBinding(frame->rhi());
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
    std::printf("  volumes alone: %u pixels with opacity over 0.25 (of %u touched): mean radiance %.5f, spread %.4f, "
                "%.2f standard errors from 1\n",
                n[0], n[1], mean, spread, (mean - 1.0) / std::max(standardError, 1e-9));
    CHECK(n[0] > 1000);
    CHECK(std::abs(mean - 1.0) < 5.0 * standardError);
}

// Authored normals and a dome, through Hydra. Under a dome of radiance 1
// with no image, a Lambert plane of albedo 0.18 reads exactly 0.18 --
// cosine sampling makes the estimate the albedo at every sample -- whether
// its normals are computed or authored, on both techniques. With authored
// normals it read 0: the view-space geometric normal came out reversed
// under the view's reflection, the backface test flipped the authored
// normal away, and only the dome, which samples about that normal, showed
// it (surface.slang says the rest).
TEST_CASE("a plane with authored normals under a dome reads its albedo through Hydra, on both techniques",
          "[usd][gpu][mesh][lights][normals]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 64, h = 48;
    std::vector<float> expected(size_t{w} * h * 4);
    for (size_t p = 0; p < size_t{w} * h; ++p) {
        expected[p * 4] = expected[p * 4 + 1] = expected[p * 4 + 2] = 0.18F;
        expected[p * 4 + 3] = 1.0F;
    }
    gpu::BufferDesc desc;
    desc.bytes = expected.size() * 4;
    desc.elementBytes = 16;
    auto reference = gpu::Buffer::create(*gpu->device, desc, expected.data());
    REQUIRE(reference);
    for (const bool authored : {false, true}) {
        const fs::path path = scratch(authored ? "dome_normals_authored.usda" : "dome_normals_computed.usda");
        {
            std::ofstream out(path);
            // The square fills the view: 4 wide at distance 1.5 against a 35mm lens.
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Mesh \"Square\"\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
                << (authored ? "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
                             : "")
                << "    uniform token subdivisionScheme = \"none\"\n}\n"
                   "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
                   "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
                   "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                   "    float2 clippingRange = (0.1, 1000)\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        for (const char* technique : {"raster", "rt"}) {
            auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
            if (!image) FAIL(image.error().toString());
            auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(frame);
            auto difference = render::compareHdr(*gpu->library, *frame, *reference, w, h);
            REQUIRE(difference);
            std::printf("  normals %s, %s: %llu pixels, worst %.2e relative to 0.18\n", authored ? "authored" : "computed",
                        technique, static_cast<unsigned long long>(difference->pixels), difference->maxRelative);
            CHECK(difference->pixels == uint64_t{w} * h);
            CHECK(difference->maxRelative < 1e-4);
        }
    }
}
