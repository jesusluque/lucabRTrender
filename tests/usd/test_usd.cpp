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
    const float* offAxis = image->rgba.data() + (size_t{h} / 2 * w + w / 2 + 40) * 4;
    const float expected = 0.8F * 0.4F * 0.4F / (2.0F * 2.0F);
    std::printf("  UsdLuxSphereLight: %u pixels, %u beyond 2%%, worst %.4f; centre %.5f (closed form %.5f), "
                "40 px off axis %.5f against %.5f at (%.3f, %.3f, %.3f)\n",
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
