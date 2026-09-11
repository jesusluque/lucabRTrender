// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/StageRenderer.h"

#include <algorithm>
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
#include <pxr/imaging/hd/primOriginSchema.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
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

void StageRenderer::requestOutputs(const std::vector<std::string>& aovs) {
    TfTokenVector outputs{HdAovTokens->color, HdAovTokens->depth};
    for (const std::string& aov : aovs) {
        const TfToken token(aov);
        if (std::find(outputs.begin(), outputs.end(), token) == outputs.end()) {
            outputs.push_back(token);
        }
    }
    impl_->controller->SetRenderOutputs(outputs);
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

Result<void> StageRenderer::setMeshVisibility(const std::string& route) {
    if (route != "automatic" && route != "raster" && route != "rays" && route != "bvh") {
        return Error::make(ErrorCode::InvalidArgument, "mesh visibility '{}': automatic, raster, rays or bvh", route);
    }
    impl_->delegate->SetRenderSetting(TfToken("lrt:visibility"), VtValue(TfToken(route)));
    return ok();
}

void StageRenderer::setLightSamples(uint32_t samples) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:lightSamples"),
                                      VtValue(static_cast<int>(std::max(samples, 1u))));
}

double StageRenderer::timeCodesPerSecond() const {
    return impl_->stage->GetTimeCodesPerSecond();
}

double StageRenderer::startTimeCode() const {
    return impl_->stage->GetStartTimeCode();
}

Result<void> StageRenderer::aim(const std::string& camera, double time, const std::string& technique) {
    Impl& impl = *impl_;
    if (technique != "raster" && technique != "rt") {
        return Error::make(ErrorCode::InvalidArgument, "technique '{}': raster or rt", technique);
    }
    impl.delegate->SetRenderSetting(TfToken("lrt:technique"), VtValue(TfToken(technique)));
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
    return ok();
}

Result<void> StageRenderer::aim(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                const std::string& technique) {
    Impl& impl = *impl_;
    if (technique != "raster" && technique != "rt") {
        return Error::make(ErrorCode::InvalidArgument, "technique '{}': raster or rt", technique);
    }
    impl.delegate->SetRenderSetting(TfToken("lrt:technique"), VtValue(TfToken(technique)));
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
    return ok();
}

Result<StageImage> StageRenderer::render(const std::string& camera, double time, uint32_t width,
                                         uint32_t height, const std::string& technique) {
    // An image, not a viewport: streamed assets are loaded before it is drawn.
    impl_->delegate->SetRenderSetting(TfToken("lrt:settleStreams"), VtValue(true));
    LRT_TRY(aim(camera, time, technique));
    LRT_TRY(execute(width, height));
    return readImage(width, height);
}

Result<StageImage> StageRenderer::render(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                         const std::string& technique) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:settleStreams"), VtValue(true));
    LRT_TRY(aim(camera, time, width, height, technique));
    LRT_TRY(execute(width, height));
    return readImage(width, height);
}

Result<void> StageRenderer::draw(const std::string& camera, double time, uint32_t width, uint32_t height,
                                 const std::string& technique) {
    // A viewport: streamed assets fill in over the frames that follow.
    impl_->delegate->SetRenderSetting(TfToken("lrt:settleStreams"), VtValue(false));
    LRT_TRY(aim(camera, time, technique));
    return execute(width, height);
}

Result<void> StageRenderer::draw(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                 const std::string& technique) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:settleStreams"), VtValue(false));
    LRT_TRY(aim(camera, time, width, height, technique));
    return execute(width, height);
}

Result<void> StageRenderer::execute(uint32_t width, uint32_t height) {
    Impl& impl = *impl_;
    impl.controller->SetRenderBufferSize(GfVec2i(static_cast<int>(width), static_cast<int>(height)));
    impl.controller->SetFraming(CameraUtilFraming(
        GfRect2i(GfVec2i(0), static_cast<int>(width), static_cast<int>(height))));
    HdTaskSharedPtrVector tasks = impl.controller->GetRenderingTasks();
    impl.engine.Execute(impl.index, &tasks);
    const render::RenderTargets* targets = lastTargets();
    if (targets == nullptr || targets->width != width || targets->height != height) {
        return Error(ErrorCode::InternalError, "the render pass drew nothing");
    }
    return ok();
}

const render::RenderTargets* StageRenderer::lastTargets() const {
    auto* param = static_cast<HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->lastTargets() : nullptr;
}

Result<StageImage> StageRenderer::readImage(uint32_t width, uint32_t height) {
    // The engine's own targets, as the render pass left them: bottom row
    // first and view z already, so the only step is the readback.
    const render::RenderTargets* targets = lastTargets();
    if (targets == nullptr || targets->width != width || targets->height != height) {
        return Error(ErrorCode::InternalError, "the render pass drew nothing to read");
    }
    StageImage image;
    image.width = width;
    image.height = height;
    auto rgba = targets->colour.readAll<float>(impl_->delegate->GetEngineDevice());
    if (!rgba) return std::move(rgba).error();
    auto depth = targets->depth.readAll<float>(impl_->delegate->GetEngineDevice());
    if (!depth) return std::move(depth).error();
    image.rgba = std::move(*rgba);
    image.depth = std::move(*depth);
    image.rgba.resize(size_t{width} * height * 4);
    image.depth.resize(size_t{width} * height);
    return image;
}

gpu::Device& StageRenderer::device() {
    return impl_->delegate->GetEngineDevice();
}

gpu::ShaderLibrary& StageRenderer::library() {
    return impl_->delegate->GetEngineLibrary();
}

Result<technique::DisplaySource> StageRenderer::displaySource(const std::string& aov) {
    const render::RenderTargets* targets = lastTargets();
    auto* param = static_cast<HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    if (targets == nullptr || engine == nullptr) {
        return Error(ErrorCode::InvalidArgument, "nothing drawn yet");
    }
    AovSource source;
    technique::DisplaySource shown;
    if (aov == "color") {
        source.kind = AovKind::Colour;
        shown.kind = technique::DisplaySource::Kind::Colour;
    } else if (aov == "depth") {
        source.kind = AovKind::Depth;
        shown.kind = technique::DisplaySource::Kind::Depth;
    } else if (aov == "primId" || aov == "instanceId" || aov == "elementId") {
        source.kind = aov == "primId" ? AovKind::PrimId : aov == "instanceId" ? AovKind::InstanceId : AovKind::ElementId;
        shown.kind = technique::DisplaySource::Kind::Ids;
    } else if (aov == "Neye" || aov == "normal") {
        source.kind = aov == "Neye" ? AovKind::EyeNormal : AovKind::WorldNormal;
        shown.kind = technique::DisplaySource::Kind::Vector;
    } else {
        return Error::make(ErrorCode::InvalidArgument, "no display for AOV '{}'", aov);
    }
    const AovView view = engine->aovView(*targets, source);
    shown.buffer = view.buffer;
    shown.stride = view.stride;
    shown.offset = view.offset;
    shown.width = targets->width;
    shown.height = targets->height;
    shown.bottomRowFirst = true;
    return shown;
}

Result<std::optional<scene::Bounds>> StageRenderer::bounds() {
    auto* param = static_cast<HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    if (engine == nullptr) {
        return std::optional<scene::Bounds>{};
    }
    return engine->bounds();
}

Result<std::optional<StagePick>> StageRenderer::pick(uint32_t x, uint32_t y) {
    Impl& impl = *impl_;
    const render::RenderTargets* targets = lastTargets();
    auto* param = static_cast<HdLrtRenderParam*>(impl.delegate->GetRenderParam());
    Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    if (targets == nullptr || engine == nullptr || x >= targets->width || y >= targets->height) {
        return std::optional<StagePick>{};
    }
    const AovView view = engine->aovView(*targets, {AovKind::PrimId, 0});
    if (view.buffer == nullptr) {
        return std::optional<StagePick>{};
    }
    // Two ids of one pixel: bookkeeping, not an image read back.
    const uint64_t pixel = uint64_t{targets->height - 1 - y} * targets->width + x;   // bottom row first
    uint32_t ids[2] = {0, 0};
    LRT_TRY(view.buffer->read(impl.delegate->GetEngineDevice(), pixel * view.stride * 4, sizeof(ids), ids));
    if (ids[0] == 0xFFFFFFFFu) {
        return std::optional<StagePick>{};
    }
    StagePick picked;
    const SdfPath rprim = impl.index->GetRprimPathFromPrimId(static_cast<int>(ids[0]));
    if (rprim.IsEmpty()) {
        return std::optional<StagePick>{};
    }
    picked.rprim = rprim.GetString();
    picked.instance = static_cast<int32_t>(ids[1]);
    const HdSceneIndexPrim prim = impl.index->GetTerminalSceneIndex()->GetPrim(rprim);
    const SdfPath origin = HdPrimOriginSchema::GetFromParent(prim.dataSource).GetOriginPath(HdPrimOriginSchemaTokens->scenePath);
    picked.prim = origin.IsEmpty() ? picked.rprim : origin.GetString();
    return std::optional<StagePick>(std::move(picked));
}

std::vector<StagePrim> StageRenderer::children(const std::string& path) const {
    std::vector<StagePrim> out;
    const UsdPrim parent = path.empty() || path == "/" ? impl_->stage->GetPseudoRoot()
                                                       : impl_->stage->GetPrimAtPath(SdfPath(path));
    if (!parent) {
        return out;
    }
    for (const UsdPrim& child : parent.GetChildren()) {
        StagePrim row;
        row.path = child.GetPath().GetString();
        row.name = child.GetName().GetString();
        row.type = child.GetTypeName().GetString();
        row.hasChildren = !child.GetChildren().empty();
        row.instance = child.IsInstance();
        out.push_back(std::move(row));
    }
    return out;
}

char StageRenderer::upAxis() const {
    return UsdGeomGetStageUpAxis(impl_->stage) == UsdGeomTokens->z ? 'Z' : 'Y';
}

double StageRenderer::endTimeCode() const {
    return impl_->stage->GetEndTimeCode();
}

}   // namespace lrt::usd
