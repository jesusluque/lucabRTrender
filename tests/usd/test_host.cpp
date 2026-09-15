// Copyright (c) 2026 lucabRTrender contributors.
//
// The plugin as a host drives it: UsdImagingGLEngine -- usdview's engine --
// loading HdLrtRendererPlugin by name, with its own render index, its own
// scene index chain and none of StageRenderer's help. What it draws is read
// from the engine's AOV render buffer and compared by a kernel. In the Storm
// executable of its own that does not link lrt::usd (tests/CMakeLists.txt).
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>

#include <pxr/base/gf/rect2i.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usdImaging/usdImagingGL/engine.h>

#include "lrt/core/Platform.h"
#include "lrt/render/ReferenceRenderer.h"

using namespace lrt;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

fs::path hostScratch(const char* name) {
    const fs::path dir = fs::temp_directory_path() / "lrt-host-tests";
    fs::create_directories(dir);
    return dir / name;
}

/// A gathered frame of `stage` through a UsdImagingGLEngine running the
/// plugin, path traced, as float4 rows.
std::vector<float> hostFrame(UsdImagingGLEngine& engine, const UsdStageRefPtr& stage, uint32_t w, uint32_t h) {
    UsdImagingGLRenderParams params;
    params.frame = UsdTimeCode(0.5);
    params.clearColor = GfVec4f(0.0F);
    engine.Render(stage->GetPseudoRoot(), params);
    for (int pass = 0; pass < 256 && !engine.IsConverged(); ++pass) {
        engine.Render(stage->GetPseudoRoot(), params);
    }
    HdRenderBuffer* buffer = engine.GetAovRenderBuffer(HdAovTokens->color);
    std::vector<float> rgba;
    if (buffer == nullptr || buffer->GetFormat() != HdFormatFloat32Vec4 || buffer->GetWidth() != w ||
        buffer->GetHeight() != h) {
        return rgba;
    }
    buffer->Resolve();
    rgba.resize(size_t{w} * h * 4);
    std::memcpy(rgba.data(), buffer->Map(), rgba.size() * sizeof(float));
    buffer->Unmap();
    return rgba;
}

std::unique_ptr<UsdImagingGLEngine> hostEngine(uint32_t w, uint32_t h) {
    UsdImagingGLEngine::Parameters parameters;
    parameters.rendererPluginId = TfToken("HdLrtRendererPlugin");
    auto engine = std::make_unique<UsdImagingGLEngine>(parameters);
    if (engine->GetCurrentRendererId() != TfToken("HdLrtRendererPlugin")) {
        return nullptr;
    }
    engine->SetEnablePresentation(false);
    engine->SetRenderBufferSize(GfVec2i(static_cast<int>(w), static_cast<int>(h)));
    engine->SetFraming(CameraUtilFraming(GfRect2i(GfVec2i(0, 0), static_cast<int>(w), static_cast<int>(h))));
    engine->SetCameraPath(SdfPath("/Camera"));
    engine->SetRendererAov(HdAovTokens->color);
    engine->SetRendererSetting(TfToken("lrt:technique"), VtValue(TfToken("rt")));
    engine->SetRendererSetting(TfToken("lrt:pathSamples"), VtValue(64));
    engine->SetRendererSetting(TfToken("lrt:pathTotal"), VtValue(64));
    engine->SetRendererSetting(TfToken("lrt:pathBounces"), VtValue(0));
    engine->SetRendererSetting(TfToken("lrt:motionBuckets"), VtValue(8));
    return engine;
}

}   // namespace

// A shutter authored open after a first frame, with a host driving the
// plugin: the prims were sampled about the closed shutter, and the pass is
// the only thing that sees the camera's change. Its change tracker's marks
// do not reach prims a scene index owns, so the delegate dirties them
// through a scene index of its own in the chain every host builds. The next
// gathered frame must be the one a fresh engine on the edited stage draws.
TEST_CASE("usdImagingGL drives the plugin, and a shutter opened after a frame blurs the next as authored",
          "[usd][gpu][host][motion][shutter]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
#if defined(__linux__)
    // UsdImagingGLEngine makes a HgiGL, which wants a context current.
    if (!platform::makeHeadlessGlContextCurrent()) {
        SKIP("no OpenGL context could be made for the host's Hgi (EGL on a GPU device)");
    }
#endif
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(LRT_HYDRA_PLUGIN_DIR).string());
    const fs::path path = hostScratch("host_shutter.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)]\n"
               "    double3 xformOp:translate.timeSamples = {\n        0: (-1, 0, 0),\n        1: (1, 0, 0),\n    }\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
    }
    const uint32_t w = 160, h = 120;
    const auto upload = [&](const std::vector<float>& rgba) {
        REQUIRE(rgba.size() == size_t{w} * h * 4);
        gpu::BufferDesc desc;
        desc.bytes = rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    auto engine = hostEngine(w, h);
    if (!engine) FAIL("UsdImagingGLEngine did not take HdLrtRendererPlugin");
    const std::vector<float> sharp = hostFrame(*engine, stage, w, h);
    {
        UsdGeomCamera camera(stage->GetPrimAtPath(SdfPath("/Camera")));
        REQUIRE(camera);
        camera.GetShutterOpenAttr().Set(-0.25);
        camera.GetShutterCloseAttr().Set(0.25);
    }
    const std::vector<float> edited = hostFrame(*engine, stage, w, h);
    engine.reset();

    UsdStageRefPtr authoredStage = UsdStage::Open(stage->GetRootLayer());
    auto fresh = hostEngine(w, h);
    REQUIRE(fresh);
    const std::vector<float> authored = hostFrame(*fresh, authoredStage, w, h);
    fresh.reset();

    const gpu::Buffer a = upload(edited);
    const gpu::Buffer b = upload(authored);
    const gpu::Buffer c = upload(sharp);
    auto same = render::compareHdr(*gpu->library, a, b, w, h);
    auto blurred = render::compareHdr(*gpu->library, b, c, w, h);
    REQUIRE(same);
    REQUIRE(blurred);
    std::printf("  through usdImagingGL, shutter opened after a frame: against the authored stage relMSE %.2e (max "
                "relative %.2e); the authored blur against the sharp frame %.2e\n",
                same->relMse, same->maxRelative, blurred->relMse);
    CHECK(blurred->relMse > 1e-2);
    CHECK(same->maxRelative < 1e-3);
}
