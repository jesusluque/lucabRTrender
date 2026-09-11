// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/StageRenderer.h"

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include "RenderDelegate.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace lrt::usd {

struct StageRenderer::Impl {
    UsdStageRefPtr                         stage;
    std::unique_ptr<HdLrtRenderDelegate>   delegate;
    HdRenderIndex*                         index = nullptr;
    UsdImagingSceneIndices                 sceneIndices;
    std::unique_ptr<HdxTaskController>     controller;
    HdEngine                               engine;

    ~Impl() {
        controller.reset();
        delete index;
        delegate.reset();
    }
};

StageRenderer::StageRenderer() : impl_(std::make_unique<Impl>()) {}
StageRenderer::~StageRenderer() = default;

Result<std::unique_ptr<StageRenderer>> StageRenderer::open(const std::filesystem::path& path) {
    auto renderer = std::unique_ptr<StageRenderer>(new StageRenderer());
    Impl& impl = *renderer->impl_;
    impl.stage = UsdStage::Open(path.string());
    if (!impl.stage) {
        return Error::make(ErrorCode::IoFailure, "cannot open USD stage '{}'", path.string());
    }
    impl.delegate = std::make_unique<HdLrtRenderDelegate>();
    if (!impl.delegate->HasEngine()) {
        return Error(ErrorCode::DeviceFailure, "the render delegate has no GPU");
    }
    impl.index = HdRenderIndex::New(impl.delegate.get(), HdDriverVector());
    if (impl.index == nullptr) {
        return Error(ErrorCode::InternalError, "cannot make a Hydra render index");
    }
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = impl.stage;
    impl.sceneIndices = UsdImagingCreateSceneIndices(info);
    impl.index->InsertSceneIndex(impl.sceneIndices.finalSceneIndex, SdfPath::AbsoluteRootPath());

    impl.controller = std::make_unique<HdxTaskController>(
        impl.index, SdfPath("/__lrtTaskController"), /*gpuEnabled=*/false);
    impl.controller->SetRenderOutputs({HdAovTokens->color, HdAovTokens->depth});
    HdRprimCollection collection(HdTokens->geometry, HdReprSelector(HdReprTokens->smoothHull));
    collection.SetRootPath(SdfPath::AbsoluteRootPath());
    impl.controller->SetCollection(collection);
    return renderer;
}

std::vector<std::string> StageRenderer::cameras() const {
    std::vector<std::string> out;
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        if (prim.IsA<UsdGeomCamera>()) {
            out.push_back(prim.GetPath().GetString());
        }
    }
    return out;
}

double StageRenderer::timeCodesPerSecond() const {
    return impl_->stage->GetTimeCodesPerSecond();
}

double StageRenderer::startTimeCode() const {
    return impl_->stage->GetStartTimeCode();
}

Result<StageImage> StageRenderer::render(const std::string& camera, double time, uint32_t width,
                                         uint32_t height, const std::string& technique) {
    Impl& impl = *impl_;
    if (technique != "raster" && technique != "rt") {
        return Error::make(ErrorCode::InvalidArgument, "technique '{}': raster or rt", technique);
    }
    impl.delegate->SetRenderSetting(TfToken("lrt:technique"), VtValue(TfToken(technique)));
    // An image, not a viewport: streamed assets are loaded before it is drawn.
    impl.delegate->SetRenderSetting(TfToken("lrt:settleStreams"), VtValue(true));
    std::string cameraPath = camera;
    if (cameraPath.empty()) {
        const auto all = cameras();
        if (all.empty()) {
            return Error(ErrorCode::NotFound, "the stage has no camera");
        }
        cameraPath = all.front();
    }
    if (!impl.stage->GetPrimAtPath(SdfPath(cameraPath)).IsA<UsdGeomCamera>()) {
        return Error::make(ErrorCode::NotFound, "no camera at '{}'", cameraPath);
    }
    impl.sceneIndices.stageSceneIndex->SetTime(UsdTimeCode(time));
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();

    impl.controller->SetCameraPath(SdfPath(cameraPath));
    impl.controller->SetRenderBufferSize(GfVec2i(static_cast<int>(width), static_cast<int>(height)));
    impl.controller->SetFraming(CameraUtilFraming(
        GfRect2i(GfVec2i(0), static_cast<int>(width), static_cast<int>(height))));
    HdTaskSharedPtrVector tasks = impl.controller->GetRenderingTasks();
    impl.engine.Execute(impl.index, &tasks);

    StageImage image;
    image.width = width;
    image.height = height;
    image.rgba.resize(size_t{width} * height * 4);
    image.depth.resize(size_t{width} * height);
    HdRenderBuffer* colour = impl.controller->GetRenderOutput(HdAovTokens->color);
    HdRenderBuffer* depth = impl.controller->GetRenderOutput(HdAovTokens->depth);
    if (colour == nullptr || depth == nullptr) {
        return Error(ErrorCode::InternalError, "the task controller gave no render outputs");
    }
    colour->Resolve();
    depth->Resolve();
    if (colour->GetWidth() != width || colour->GetFormat() != HdFormatFloat32Vec4) {
        return Error::make(ErrorCode::InternalError, "unexpected colour output ({}x{}, format {})",
                           colour->GetWidth(), colour->GetHeight(), static_cast<int>(colour->GetFormat()));
    }
    // Render buffers are top row first; the engine's images bottom row first.
    const auto* c = static_cast<const float*>(colour->Map());
    const auto* d = static_cast<const float*>(depth->Map());
    for (uint32_t y = 0; y < height; ++y) {
        const size_t from = size_t{height - 1 - y} * width;
        std::memcpy(image.rgba.data() + size_t{y} * width * 4, c + from * 4, size_t{width} * 16);
        std::memcpy(image.depth.data() + size_t{y} * width, d + from, size_t{width} * 4);
    }
    colour->Unmap();
    depth->Unmap();
    return image;
}

}   // namespace lrt::usd
