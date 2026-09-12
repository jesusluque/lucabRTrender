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

#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hd/renderSettings.h>
#include <pxr/imaging/hd/dependencyForwardingSceneIndex.h>
#include <pxr/imaging/hdsi/legacyDisplayStyleOverrideSceneIndex.h>
#include <pxr/imaging/hdsi/sceneGlobalsSceneIndex.h>
#include <pxr/base/tf/stringUtils.h>

#include "lrt/io/Exr.h"
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include "RenderDelegate.h"
#include "RenderParam.h"
#include "RenderSettings.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace lrt::usd {

struct StageRenderer::Impl {
    UsdStageRefPtr                         stage;
    std::unique_ptr<HdLrtRenderDelegate>   delegate;
    HdsiLegacyDisplayStyleOverrideSceneIndexRefPtr displayStyle;
    HdsiSceneGlobalsSceneIndexRefPtr       globals;   ///< the active render settings prim, the frame
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
    // Through the filters registered for this renderer before the index sees
    // it: that is what resolves a light's collections into the categories
    // GetCategories then reports. Inserting the stage's own chain directly
    // skips every one of them.
    HdLrtRegisterSceneIndices();
    // The display style's fallback refine level, for setRefineLevel: what
    // usdview's complexity sets, the same way.
    impl.displayStyle = HdsiLegacyDisplayStyleOverrideSceneIndex::New(impl.sceneIndices.finalSceneIndex);
    // The scene's globals -- which render settings prim is active, which
    // frame -- are what a settings prim's `IsActive` and the products read.
    impl.globals = HdsiSceneGlobalsSceneIndex::New(impl.displayStyle);
    // The dependencies the scene indices declare (a settings prim's
    // `active` on the globals) become dirty notices only through a
    // forwarding scene index at the end of the chain.
    const HdSceneIndexBaseRefPtr scene = HdDependencyForwardingSceneIndex::New(
        HdSceneIndexPluginRegistry::GetInstance().AppendSceneIndicesForRenderer("lucabRTrender", impl.globals));
    impl.index->InsertSceneIndex(scene, SdfPath::AbsoluteRootPath());

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

namespace {

/// A UsdRender purpose as Hydra's render tag.
TfToken tagOfPurpose(const std::string& purpose) {
    if (purpose == "default" || purpose.empty()) return HdRenderTagTokens->geometry;
    if (purpose == "render") return HdRenderTagTokens->render;
    if (purpose == "proxy") return HdRenderTagTokens->proxy;
    if (purpose == "guide") return HdRenderTagTokens->guide;
    return TfToken(purpose);
}

/// A var's source as the delegate's AOV: hd's names as they are, RenderMan's
/// "Ci" and "z" as colour and depth, and of light path expressions the one
/// that names a light group -- C.*<L.'NAME'> -- as "lightGroup:NAME".
Result<std::string> aovOfVar(const RenderVarInfo& var) {
    if (var.sourceType == "lpe") {
        const size_t open = var.sourceName.find("<L.'");
        const size_t close = open != std::string::npos ? var.sourceName.find("'>", open + 4) : std::string::npos;
        if (open != std::string::npos && close != std::string::npos && close > open + 4) {
            return "lightGroup:" + var.sourceName.substr(open + 4, close - open - 4);
        }
        return Error::make(ErrorCode::Unsupported,
                           "render var '{}': of light path expressions only a light group's (C.*<L.'NAME'>) is read",
                           var.name);
    }
    if (var.sourceType == "primvar") {
        return "primvars:" + var.sourceName;
    }
    if (var.sourceName == "Ci") return std::string("color");
    if (var.sourceName == "z") return std::string("depth");
    return var.sourceName;
}

std::string textOf(const VtValue& value) {
    if (value.IsHolding<std::string>()) return value.UncheckedGet<std::string>();
    if (value.IsHolding<TfToken>()) return value.UncheckedGet<TfToken>().GetString();
    if (value.IsHolding<bool>()) return value.UncheckedGet<bool>() ? "true" : "false";
    if (value.IsHolding<int>()) return std::to_string(value.UncheckedGet<int>());
    if (value.IsHolding<float>()) return TfStringify(value.UncheckedGet<float>());
    if (value.IsHolding<double>()) return TfStringify(value.UncheckedGet<double>());
    return TfStringify(value);
}

}   // namespace

void StageRenderer::setIncludedPurposes(const std::vector<std::string>& purposes) {
    TfTokenVector tags;
    if (purposes.empty()) {
        tags = {HdRenderTagTokens->geometry, HdRenderTagTokens->render};
    }
    for (const std::string& purpose : purposes) {
        const TfToken tag = tagOfPurpose(purpose);
        if (std::find(tags.begin(), tags.end(), tag) == tags.end()) {
            tags.push_back(tag);
        }
    }
    impl_->controller->SetRenderTags(tags);
}

Result<RenderSettingsInfo> StageRenderer::renderSettings(const std::string& path) {
    Impl& impl = *impl_;
    const SdfPath id(path);
    if (!impl.stage->GetPrimAtPath(id).IsValid()) {
        return Error::make(ErrorCode::NotFound, "no prim at '{}'", path);
    }
    // Made the scene's active settings prim, then synced -- with no tasks,
    // so only the prims themselves, not a frame -- and read from the bprim.
    impl.globals->SetActiveRenderSettingsPrimPath(id);
    impl.sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    // Made active, the prim's `active` is dirtied through its dependency on
    // the globals (the filtering scene index declares it, the dependency
    // forwarding one turns it into a notice, the emulation into a dirty
    // bit) and read again at Sync: with no tasks, only the prims sync.
    {
        HdTaskSharedPtrVector none;
        HdTaskContext context;
        impl.index->SyncAll(&none, &context);
    }
    const auto* prim = dynamic_cast<const HdLrtRenderSettings*>(impl.index->GetBprim(HdPrimTypeTokens->renderSettings, id));
    if (prim == nullptr) {
        return Error::make(ErrorCode::NotFound, "'{}' is not a render settings prim Hydra delivered", path);
    }
    RenderSettingsInfo info;
    info.path = path;
    info.active = prim->IsActive();
    info.syncs = prim->GetSyncCount();
    for (const TfToken& purpose : prim->GetIncludedPurposes()) info.includedPurposes.push_back(purpose.GetString());
    for (const TfToken& purpose : prim->GetMaterialBindingPurposes()) {
        info.materialBindingPurposes.push_back(purpose.GetString());
    }
    info.renderingColorSpace = prim->GetRenderingColorSpace().GetString();
    if (prim->GetCamera().IsHolding<SdfPath>()) {
        info.camera = prim->GetCamera().UncheckedGet<SdfPath>().GetString();
    }
    info.disableMotionBlur = prim->GetDisableMotionBlur();
    info.disableDepthOfField = prim->GetDisableDepthOfField();
    for (const auto& [key, value] : prim->GetNamespacedSettings()) {
        info.settings[key] = textOf(value);
    }
    for (const HdRenderSettings::RenderProduct& product : prim->GetRenderProducts()) {
        RenderProductInfo p;
        p.path = product.productPath.GetString();
        p.name = product.name.GetString();
        p.type = product.type.GetString();
        p.width = static_cast<uint32_t>(std::max(product.resolution[0], 0));
        p.height = static_cast<uint32_t>(std::max(product.resolution[1], 0));
        p.camera = product.cameraPath.GetString();
        p.disableMotionBlur = product.disableMotionBlur;
        p.disableDepthOfField = product.disableDepthOfField;
        for (const HdRenderSettings::RenderProduct::RenderVar& var : product.renderVars) {
            p.vars.push_back({var.varPath.GetName(), var.sourceName, var.sourceType.GetString(), var.dataType.GetString()});
        }
        info.products.push_back(std::move(p));
    }
    return info;
}

Result<std::vector<std::filesystem::path>> StageRenderer::renderProducts(const std::string& path, double time,
                                                                         const std::filesystem::path& directory) {
    Impl& impl = *impl_;
    auto info = renderSettings(path);
    if (!info) return std::move(info).error();
    if (info->products.empty()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}' names no render products", path);
    }
    // The prim's own settings for this renderer, as the delegate's.
    const auto* prim =
        static_cast<const HdRenderSettings*>(impl.index->GetBprim(HdPrimTypeTokens->renderSettings, SdfPath(path)));
    std::string technique = "raster";
    bool half = false;
    for (const auto& [key, value] : prim->GetNamespacedSettings()) {
        if (key == "lrt:technique" && !textOf(value).empty()) {
            technique = textOf(value);
        } else if (key == "lrt:exrHalf") {
            half = value.IsHolding<bool>() && value.UncheckedGet<bool>();
        } else if (key.rfind("lrt:", 0) == 0) {
            impl.delegate->SetRenderSetting(TfToken(key), value);
        }
    }
    setIncludedPurposes(info->includedPurposes);
    std::vector<std::filesystem::path> written;
    for (const RenderProductInfo& product : info->products) {
        if (product.width == 0 || product.height == 0) {
            return Error::make(ErrorCode::InvalidArgument, "render product '{}' has no resolution", product.path);
        }
        if (product.vars.empty()) {
            return Error::make(ErrorCode::InvalidArgument, "render product '{}' has no vars", product.path);
        }
        std::vector<std::string> aovs;
        for (const RenderVarInfo& var : product.vars) {
            auto aov = aovOfVar(var);
            if (!aov) return std::move(aov).error();
            aovs.push_back(*aov);
        }
        requestOutputs(aovs);
        const std::string camera = !product.camera.empty() ? product.camera : info->camera;
        auto image = render(camera, time, product.width, product.height, technique);
        if (!image) return std::move(image).error();
        // Each var's layer, from the same Hydra buffer a host maps -- depth
        // as the view z the engine's own image carries, as `lrt stage -o`.
        std::vector<std::vector<uint32_t>> planes;
        std::vector<io::ExrChannel> channels;
        const size_t pixels = size_t{product.width} * product.height;
        for (size_t k = 0; k < product.vars.size(); ++k) {
            const RenderVarInfo& var = product.vars[k];
            const std::string& aov = aovs[k];
            if (aov == "depth") {
                planes.emplace_back(pixels);
                std::memcpy(planes.back().data(), image->depth.data(), pixels * 4);
                channels.push_back({var.name == "depth" ? "Z" : var.name, io::ExrChannelType::Float, {}});
                continue;
            }
            HdRenderBuffer* buffer = impl.controller->GetRenderOutput(TfToken(aov));
            if (buffer == nullptr) {
                return Error::make(ErrorCode::NotFound, "render var '{}': no output '{}'", var.name, aov);
            }
            auto bytes = mappedOutput(aov);
            if (!bytes) return std::move(bytes).error();
            const HdFormat format = buffer->GetFormat();
            const size_t components = HdGetComponentCount(format);
            const bool integer = HdGetComponentFormat(format) == HdFormatInt32;
            if (HdDataSizeOfFormat(format) != components * 4 || bytes->size() < pixels * components * 4) {
                return Error::make(ErrorCode::Unsupported, "render var '{}': output '{}' is not 32-bit", var.name, aov);
            }
            static constexpr const char* kColour[4] = {"R", "G", "B", "A"};
            static constexpr const char* kVector[4] = {"x", "y", "z", "w"};
            for (size_t c = 0; c < components; ++c) {
                planes.emplace_back(pixels);
                for (size_t p = 0; p < pixels; ++p) {
                    std::memcpy(&planes.back()[p], bytes->data() + (p * components + c) * 4, 4);
                }
                std::string name = var.name;
                if (components == 4) {
                    name = var.name == "color" ? std::string(kColour[c]) : var.name + "." + kColour[c];
                } else if (components > 1) {
                    name = var.name + "." + kVector[c];
                }
                channels.push_back({name, integer ? io::ExrChannelType::Uint
                                          : half ? io::ExrChannelType::Half
                                                 : io::ExrChannelType::Float,
                                    {}});
            }
        }
        for (size_t k = 0; k < channels.size(); ++k) {
            channels[k].words = planes[k];
        }
        std::filesystem::path file(product.name.empty() ? product.path.substr(1) + ".exr" : product.name);
        if (file.is_relative() && !directory.empty()) {
            file = directory / file;
        }
        LRT_TRY(io::writeExrChannels(file, product.width, product.height, channels));
        written.push_back(file);
    }
    return written;
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

void StageRenderer::setChooseLights(bool choose) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:chooseLights"), VtValue(choose));
}
void StageRenderer::setPathSamples(uint32_t samples) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:pathSamples"),
                                      VtValue(static_cast<int>(std::max(samples, 1u))));
}
void StageRenderer::setPathBounces(uint32_t bounces) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:pathBounces"), VtValue(static_cast<int>(bounces)));
}
void StageRenderer::setRefineLevel(uint32_t level) {
    if (impl_->displayStyle) {
        impl_->displayStyle->SetRefineLevelFallback(level > 0 ? std::optional<int>(static_cast<int>(level))
                                                               : std::nullopt);
    }
}
void StageRenderer::setMotionBuckets(uint32_t buckets) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:motionBuckets"), VtValue(static_cast<int>(buckets)));
}
void StageRenderer::setPathAdaptive(bool adaptive) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:pathAdaptive"), VtValue(adaptive));
}
void StageRenderer::setPathError(float error) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:pathError"), VtValue(error));
}
void StageRenderer::setDenoise(bool denoise) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:denoise"), VtValue(denoise));
}
void StageRenderer::setPathTotal(uint32_t total) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:pathTotal"), VtValue(static_cast<int>(std::max(total, 1u))));
}
uint32_t StageRenderer::pathAccumulated() const {
    auto* param = static_cast<HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->pathAccumulated() : 0;
}

uint64_t StageRenderer::meshGeneration() const {
    auto* param = static_cast<HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->meshGeneration() : 0;
}

std::vector<CoordSysBinding> StageRenderer::coordSysBindings(const std::string& prim) const {
    const auto* param = static_cast<const HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->coordSysOf(SdfPath(prim)) : std::vector<CoordSysBinding>{};
}

uint64_t StageRenderer::meshPositionsRevision() const {
    auto* param = static_cast<HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine != nullptr ? engine->meshPositionsRevision() : 0;
}
bool StageRenderer::pathConverged() const {
    auto* param = static_cast<HdLrtRenderParam*>(impl_->delegate->GetRenderParam());
    const Engine* engine = param != nullptr ? param->GetEngine() : nullptr;
    return engine == nullptr || engine->pathConverged();
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
    // The camera's shutter, ahead of the first Sync, so the first frame
    // samples at it rather than the second.
    {
        const UsdGeomCamera usdCamera(impl.stage->GetPrimAtPath(SdfPath(cameraPath)));
        double open = 0.0;
        double close = 0.0;
        usdCamera.GetShutterOpenAttr().Get(&open, UsdTimeCode(time));
        usdCamera.GetShutterCloseAttr().Get(&close, UsdTimeCode(time));
        if (auto* param = static_cast<HdLrtRenderParam*>(impl.delegate->GetRenderParam()); param != nullptr) {
            param->SetShutter(open, close);
        }
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
    LRT_TRY(executeUntilGathered(width, height));
    return readImage(width, height);
}

Result<StageImage> StageRenderer::render(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                         const std::string& technique) {
    impl_->delegate->SetRenderSetting(TfToken("lrt:settleStreams"), VtValue(true));
    LRT_TRY(aim(camera, time, width, height, technique));
    LRT_TRY(executeUntilGathered(width, height));
    return readImage(width, height);
}

Result<void> StageRenderer::executeUntilGathered(uint32_t width, uint32_t height) {
    // An image is drawn until the path traced frame holds what it was asked
    // for: the pass reports itself unconverged until then, and a frame that is
    // not path traced is whole at once. The cap is against a total no pass
    // count could reach.
    for (uint32_t pass = 0; pass < 65536; ++pass) {
        LRT_TRY(execute(width, height));
        if (pathConverged()) {
            return ok();
        }
    }
    return Error(ErrorCode::InternalError, "the path traced frame did not gather its total in 65536 passes");
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
