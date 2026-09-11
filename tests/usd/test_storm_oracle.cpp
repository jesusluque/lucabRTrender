// Copyright (c) 2026 lucabRTrender contributors.
//
// Storm -- OpenUSD's own GPU renderer, on Metal here -- as the oracle for what
// geometry the engine draws: the same stage through the same scene indices,
// the same camera, compared on primId, depth and Neye by a kernel. Not colour:
// Storm lights and shades by its own rules. Its own executable, so Storm's
// device and plugins live in a process of their own.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/range1f.h>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/hd/driver.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/tokens.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include "lrt/core/Platform.h"
#include "lrt/usd/StageRenderer.h"


using namespace lrt;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

struct Outputs {
    std::vector<uint8_t> primId, depth, eye;
};

GfMatrix4d viewOf(const render::Camera& camera) {
    const render::Mat4 toCamera = aofx::xform::inverseAffine(camera.cameraToWorld);
    GfMatrix4d view;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            view[c][r] = toCamera.at(r, c);
        }
    }
    return view;
}

GfMatrix4d projectionOf(const render::Camera& camera, uint32_t w, uint32_t h) {
    GfCamera lens;
    lens.SetFocalLength(static_cast<float>(camera.lens.focal));
    lens.SetHorizontalAperture(static_cast<float>(camera.lens.haperture));
    lens.SetVerticalAperture(static_cast<float>(camera.lens.haperture * h / w));
    lens.SetClippingRange(GfRange1f(static_cast<float>(camera.lens.nearZ), static_cast<float>(camera.lens.farZ)));
    return lens.GetFrustum().ComputeProjectionMatrix();
}

std::vector<uint8_t> mapped(HdRenderBuffer* buffer) {
    buffer->Resolve();
    const size_t bytes = size_t{buffer->GetWidth()} * buffer->GetHeight() * HdDataSizeOfFormat(buffer->GetFormat());
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), buffer->Map(), bytes);
    buffer->Unmap();
    return out;
}

Outputs storm(const fs::path& stagePath, const render::Camera& camera, uint32_t w, uint32_t h) {
    HgiUniquePtr hgi = Hgi::CreatePlatformDefaultHgi();
    REQUIRE(hgi);
    HdDriver driver{HgiTokens->renderDriver, VtValue(hgi.get())};
    HdPluginRenderDelegateUniqueHandle delegate =
        HdRendererPluginRegistry::GetInstance().CreateRenderDelegate(TfToken("HdStormRendererPlugin"));
    REQUIRE(delegate);
    std::unique_ptr<HdRenderIndex> index(HdRenderIndex::New(delegate.Get(), {&driver}));
    REQUIRE(index);
    UsdStageRefPtr stage = UsdStage::Open(stagePath.string());
    REQUIRE(stage);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices sceneIndices = UsdImagingCreateSceneIndices(info);
    index->InsertSceneIndex(sceneIndices.finalSceneIndex, SdfPath::AbsoluteRootPath());
    HdxTaskController controller(index.get(), SdfPath("/__stormOracle"), /*gpuEnabled=*/true);
    controller.SetEnableSelection(false);
    controller.SetRenderOutputs({HdAovTokens->depth, HdAovTokens->primId, HdAovTokens->Neye});
    HdRprimCollection collection(HdTokens->geometry, HdReprSelector(HdReprTokens->smoothHull));
    collection.SetRootPath(SdfPath::AbsoluteRootPath());
    controller.SetCollection(collection);
    sceneIndices.stageSceneIndex->SetTime(UsdTimeCode(0.0));
    sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    controller.SetFreeCameraMatrices(viewOf(camera), projectionOf(camera, w, h));
    controller.SetRenderBufferSize(GfVec2i(static_cast<int>(w), static_cast<int>(h)));
    controller.SetFraming(CameraUtilFraming(GfRect2i(GfVec2i(0), static_cast<int>(w), static_cast<int>(h))));
    HdEngine engine;
    HdTaskSharedPtrVector tasks = controller.GetRenderingTasks();
    engine.Execute(index.get(), &tasks);
    Outputs out;
    out.depth = mapped(controller.GetRenderOutput(HdAovTokens->depth));
    out.eye = mapped(controller.GetRenderOutput(HdAovTokens->Neye));
    out.primId = mapped(controller.GetRenderOutput(HdAovTokens->primId));
    return out;
}

Outputs engineOutputs(const fs::path& stagePath, const render::Camera& camera, uint32_t w, uint32_t h) {
    auto renderer = usd::StageRenderer::open(stagePath);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->requestOutputs({"primId", "Neye"});
    auto image = (*renderer)->render(camera, 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    Outputs out;
    auto primId = (*renderer)->mappedOutput("primId");
    auto depth = (*renderer)->mappedOutput("depth");
    auto eye = (*renderer)->mappedOutput("Neye");
    REQUIRE(primId);
    REQUIRE(depth);
    REQUIRE(eye);
    out.primId = std::move(*primId);
    out.depth = std::move(*depth);
    out.eye = std::move(*eye);
    return out;
}

}   // namespace

TEST_CASE("Kitchen_set's geometry matches Storm's: prim ids, depth and eye normals", "[usd][gpu][oracle][storm]") {
    LRT_REQUIRE_GPU(gpu);
    const fs::path kitchen = fs::path(std::getenv("HOME")) / "tools/assets/Kitchen_set/Kitchen_set.usd";
    if (!fs::exists(kitchen)) {
        SKIP("Pixar's Kitchen_set is not at " << kitchen.string());
    }
    // One sample a pixel, as the engine's visibility. Storm multisamples by
    // default, and on Metal an R32Sint buffer cannot be a resolve target: its
    // id buffers come back with their upper bits unwritten. OpenUSD reads its
    // environment settings as its libraries load, so ctest sets this one.
    if (platform::env("HDX_MSAA_SAMPLE_COUNT") != "1") {
        FAIL("Storm must render single-sampled: run with HDX_MSAA_SAMPLE_COUNT=1 in the environment (ctest sets it)");
    }
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(LRT_HYDRA_PLUGIN_DIR).string());
    const uint32_t w = 480;
    const uint32_t h = 270;
    render::Camera camera =
        render::Camera::lookingAt({500.0, -350.0, 350.0}, {0.0, 50.0, 60.0}, {0.0, 0.0, 1.0});
    camera.lens.focal = 20.0;
    camera.lens.nearZ = 1.0;
    camera.lens.farZ = 10000.0;
    const Outputs ours = engineOutputs(kitchen, camera, w, h);
    const Outputs theirs = storm(kitchen, camera, w, h);
    REQUIRE(ours.primId.size() == theirs.primId.size());
    REQUIRE(ours.depth.size() == theirs.depth.size());
    REQUIRE(ours.eye.size() == theirs.eye.size() * 3);   // float3 against RGBA8

    const auto upload = [&](const std::vector<uint8_t>& bytes, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = bytes.size();
        desc.elementBytes = 4;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, bytes.data());
        REQUIRE(made);
        return *made;
    };
    gpu::Buffer primA = upload(ours.primId, "ours.primId");
    gpu::Buffer primB = upload(theirs.primId, "storm.primId");
    gpu::Buffer depthA = upload(ours.depth, "ours.depth");
    gpu::Buffer depthB = upload(theirs.depth, "storm.depth");
    gpu::Buffer eyeA = upload(ours.eye, "ours.Neye");
    gpu::Buffer eyeB = upload(theirs.eye, "storm.Neye");
    auto compare = gpu::ComputeKernel::create(*gpu->library, "lrt/test/oracle_compare", "oracleCompare");
    if (!compare) FAIL(compare.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 9, "oracle.counts");
    gpu::BufferDesc one;
    one.bytes = 4;
    one.elementBytes = 4;
    auto worst = gpu::Buffer::create(*gpu->device, one);
    REQUIRE(worst);
    gpu::CommandBatch batch(*gpu->device);
    compare->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["primA"].setBinding(primA.rhi());
        cursor["primB"].setBinding(primB.rhi());
        cursor["depthA"].setBinding(depthA.rhi());
        cursor["depthB"].setBinding(depthB.rhi());
        cursor["eyeA"].setBinding(eyeA.rhi());
        cursor["eyeB"].setBinding(eyeB.rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst->rhi());
        cursor["params"]["width"].setData(w);
        cursor["params"]["height"].setData(h);
        cursor["params"]["depthEpsilon"].setData(1e-5F);
        cursor["params"]["byteTolerance"].setData(uint32_t{6});
    });
    REQUIRE(batch.submit(true));
    uint32_t c[9] = {};
    float depthWorst = 0.0F;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(depthWorst), &depthWorst));
    const uint32_t segmentation = c[1] + c[8];
    std::printf("  Kitchen_set against Storm: covered %u / %u (%u differ); %u of %u pixels one prim across 3x3 in only "
                "one image; %u interior pixels: %u depths > 1e-5 (worst %.2e), %u normals > 6/255\n",
                c[4], c[5], c[6], segmentation, c[7], c[0], c[2], double(depthWorst), c[3]);
    CHECK(c[0] > 50000);
    CHECK(uint64_t{c[6]} * 1000 <= uint64_t{w} * h);   // coverage: under 0.1% of the image
    CHECK(uint64_t{segmentation} * 1000 <= c[7]);
    CHECK(uint64_t{c[2]} * 1000 <= c[0]);
    CHECK(uint64_t{c[3]} * 1000 <= c[0]);
}
