// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/StageRenderer.h"

#include <cstring>

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/range1f.h>
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

Result<std::vector<uint8_t>> StageRenderer::mappedOutput(const std::string& aov) {
    HdRenderBuffer* buffer = impl_->controller->GetRenderOutput(TfToken(aov));
    if (buffer == nullptr) {
        return Error::make(ErrorCode::NotFound, "no render output '{}'", aov);
    }
    buffer->Resolve();
    const size_t bytes = size_t{buffer->GetWidth()} * buffer->GetHeight() * HdDataSizeOfFormat(buffer->GetFormat());
    std::vector<uint8_t> out(bytes);
    const auto* mapped = static_cast<const uint8_t*>(buffer->Map());
    std::memcpy(out.data(), mapped, bytes);
    buffer->Unmap();
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
    return execute(width, height);
}

Result<StageImage> StageRenderer::render(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                         const std::string& technique) {
    Impl& impl = *impl_;
    if (technique != "raster" && technique != "rt") {
        return Error::make(ErrorCode::InvalidArgument, "technique '{}': raster or rt", technique);
    }
    impl.delegate->SetRenderSetting(TfToken("lrt:technique"), VtValue(TfToken(technique)));
    impl.delegate->SetRenderSetting(TfToken("lrt:settleStreams"), VtValue(true));
    impl.sceneIndices.stageSceneIndex->SetTime(UsdTimeCode(time));
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    // A camera's matrices, as a free camera: world to camera, and the lens's
    // frustum (apertures in the image's aspect, so nothing is conformed).
    const render::Mat4 toCamera = aofx::xform::inverseAffine(camera.cameraToWorld);
    GfMatrix4d view;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            view[c][r] = toCamera.at(r, c);   // row-vector matrices are the transpose
        }
    }
    GfCamera lens;
    lens.SetFocalLength(static_cast<float>(camera.lens.focal));
    lens.SetHorizontalAperture(static_cast<float>(camera.lens.haperture));
    lens.SetVerticalAperture(static_cast<float>(camera.lens.haperture * height / width));
    lens.SetClippingRange(GfRange1f(static_cast<float>(camera.lens.nearZ), static_cast<float>(camera.lens.farZ)));
    if (camera.lens.projection == render::Lens::Projection::Orthographic) {
        lens.SetProjection(GfCamera::Orthographic);
    }
    impl.controller->SetCameraPath(SdfPath());
    impl.controller->SetFreeCameraMatrices(view, lens.GetFrustum().ComputeProjectionMatrix());
    return execute(width, height);
}

Result<StageImage> StageRenderer::execute(uint32_t width, uint32_t height) {
    Impl& impl = *impl_;
    impl.controller->SetRenderBufferSize(GfVec2i(static_cast<int>(width), static_cast<int>(height)));
    impl.controller->SetFraming(CameraUtilFraming(
        GfRect2i(GfVec2i(0), static_cast<int>(width), static_cast<int>(height))));
    HdTaskSharedPtrVector tasks = impl.controller->GetRenderingTasks();
    impl.engine.Execute(impl.index, &tasks);

    // The engine's own targets, as the render pass left them: bottom row
    // first and view z already, so the only step is the readback.
    auto* param = static_cast<HdLrtRenderParam*>(impl.delegate->GetRenderParam());
    const lrt::usd::Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    const render::RenderTargets* targets = engine != nullptr ? engine->lastTargets() : nullptr;
    if (targets == nullptr || targets->width != width || targets->height != height) {
        return Error(ErrorCode::InternalError, "the render pass drew nothing to read");
    }
    StageImage image;
    image.width = width;
    image.height = height;
    auto rgba = targets->colour.readAll<float>(impl.delegate->GetEngineDevice());
    if (!rgba) return std::move(rgba).error();
    auto depth = targets->depth.readAll<float>(impl.delegate->GetEngineDevice());
    if (!depth) return std::move(depth).error();
    image.rgba = std::move(*rgba);
    image.depth = std::move(*depth);
    image.rgba.resize(size_t{width} * height * 4);
    image.depth.resize(size_t{width} * height);
    return image;
}

}   // namespace lrt::usd
