// Copyright (c) 2026 lucabRTrender contributors.
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hdsi/lightLinkingSceneIndex.h>

#include "RenderDelegate.h"

#include "Light.h"
#include "Material.h"

#include <pxr/base/tf/staticTokens.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/tokens.h>

#include "Instancer.h"
#include "Mesh.h"
#include "ParticleField.h"
#include "Points.h"
#include "RenderBuffer.h"
#include "RenderPass.h"
#include "lrt/core/Log.h"

PXR_NAMESPACE_OPEN_SCOPE

/// Which prims the light linking scene index is about. Its defaults are not
/// ours to assume: without saying so it sees the collections and marks
/// nobody, which is what the probe showed -- a good membership expression and
/// no categories at all.
static HdContainerDataSourceHandle _lightLinkingArgs() {
    VtArray<TfToken> lights;
    for (const TfToken& type : HdLightPrimTypeTokens()) {
        lights.push_back(type);
    }
    const VtArray<TfToken> geometry{HdPrimTypeTokens->mesh, HdPrimTypeTokens->basisCurves,
                                    HdPrimTypeTokens->points, HdPrimTypeTokens->volume,
                                    HdPrimTypeTokens->instancer};
    return HdRetainedContainerDataSource::New(
        HdsiLightLinkingSceneIndexTokens->lightPrimTypes,
        HdRetainedTypedSampledDataSource<VtArray<TfToken>>::New(lights),
        HdsiLightLinkingSceneIndexTokens->geometryPrimTypes,
        HdRetainedTypedSampledDataSource<VtArray<TfToken>>::New(geometry));
}

void HdLrtRegisterSceneIndices() {
    // Once, and from anywhere: a host that makes the delegate itself never
    // goes through plug's discovery, so a registry function alone would not
    // run at all -- measured, by tracing it and seeing nothing.
    static const bool once = [] {
        HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
            "lucabRTrender",
            [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
               const HdContainerDataSourceHandle& inputArgs) -> HdSceneIndexBaseRefPtr {
                return HdsiLightLinkingSceneIndex::New(inputScene, inputArgs);
            },
            _lightLinkingArgs(), 0, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
        return true;
    }();
    (void)once;
}


HdLrtRenderDelegate::HdLrtRenderDelegate() { _Setup(); }

HdLrtRenderDelegate::HdLrtRenderDelegate(HdRenderSettingsMap const& settings)
    : HdRenderDelegate(settings) {
    _Setup();
}

HdLrtRenderDelegate::~HdLrtRenderDelegate() {
    _param.reset();
    _engine.reset();
}

void HdLrtRenderDelegate::_Setup() {
    std::string why;
    _engine = lrt::usd::Engine::create(why);
    if (_engine == nullptr) {
        lrt::log::error("hdLrt: no engine: {}", why);
    }
    _param = std::make_unique<HdLrtRenderParam>(_engine.get());
    _registry = std::make_shared<HdResourceRegistry>();
}

TfTokenVector const& HdLrtRenderDelegate::GetSupportedRprimTypes() const {
    static const TfTokenVector types{HdPrimTypeTokens->mesh, HdPrimTypeTokens->particleField,
                                     HdPrimTypeTokens->points};
    return types;
}

TfTokenVector const& HdLrtRenderDelegate::GetSupportedSprimTypes() const {
    static const TfTokenVector types{HdPrimTypeTokens->camera,      HdPrimTypeTokens->material,
                                     HdPrimTypeTokens->sphereLight, HdPrimTypeTokens->diskLight,
                                     HdPrimTypeTokens->rectLight,   HdPrimTypeTokens->distantLight,
                                     HdPrimTypeTokens->domeLight};
    return types;
}

TfTokenVector HdLrtRenderDelegate::GetMaterialRenderContexts() const {
    // MaterialX networks first, then UsdPreviewSurface ones, both through hdMtlx.
    return {TfToken("mtlx"), TfToken()};
}

TfTokenVector const& HdLrtRenderDelegate::GetSupportedBprimTypes() const {
    static const TfTokenVector types{HdPrimTypeTokens->renderBuffer};
    return types;
}

HdRenderPassSharedPtr HdLrtRenderDelegate::CreateRenderPass(HdRenderIndex* index,
                                                            HdRprimCollection const& collection) {
    return std::make_shared<HdLrtRenderPass>(index, collection, _engine.get(), this);
}

TF_DEFINE_PRIVATE_TOKENS(_lrtSettings, ((technique, "lrt:technique"))((settleStreams, "lrt:settleStreams"))
                                           ((visibility, "lrt:visibility"))((lightSamples, "lrt:lightSamples"))((chooseLights, "lrt:chooseLights"))
                                           ((pathSamples, "lrt:pathSamples"))((pathBounces, "lrt:pathBounces"))((pathTotal, "lrt:pathTotal"))((denoise, "lrt:denoise"))
                                           (raster)(rt)(automatic)(rays)(bvh));

HdRenderSettingDescriptorList HdLrtRenderDelegate::GetRenderSettingDescriptors() const {
    HdRenderSettingDescriptor technique;
    technique.name = "Technique (raster | rt)";
    technique.key = _lrtSettings->technique;
    technique.defaultValue = VtValue(_lrtSettings->raster);
    HdRenderSettingDescriptor settle;
    settle.name = "Wait for streamed assets before drawing";
    settle.key = _lrtSettings->settleStreams;
    settle.defaultValue = VtValue(false);
    HdRenderSettingDescriptor visibility;
    visibility.name = "Mesh visibility (automatic | raster | rays | bvh)";
    visibility.key = _lrtSettings->visibility;
    visibility.defaultValue = VtValue(_lrtSettings->automatic);
    HdRenderSettingDescriptor samples;
    samples.name = "Samples per light";
    samples.key = _lrtSettings->lightSamples;
    samples.defaultValue = VtValue(1);
    HdRenderSettingDescriptor choose;
    choose.name = "One light per sample, chosen by power";
    choose.key = _lrtSettings->chooseLights;
    choose.defaultValue = VtValue(false);
    HdRenderSettingDescriptor paths;
    paths.name = "Paths per pixel (rt)";
    paths.key = _lrtSettings->pathSamples;
    paths.defaultValue = VtValue(1);
    HdRenderSettingDescriptor bounces;
    bounces.name = "Bounces after the first hit (rt)";
    bounces.key = _lrtSettings->pathBounces;
    bounces.defaultValue = VtValue(1);
    HdRenderSettingDescriptor total;
    total.name = "Paths per pixel to converge to (rt)";
    total.key = _lrtSettings->pathTotal;
    total.defaultValue = VtValue(1);
    HdRenderSettingDescriptor denoise;
    denoise.name = "Denoise the path traced frame once gathered (rt)";
    denoise.key = _lrtSettings->denoise;
    denoise.defaultValue = VtValue(false);
    return {technique, settle, visibility, samples, choose, paths, bounces, total, denoise};
}

lrt::usd::MeshVisibility HdLrtRenderDelegate::GetMeshVisibility() const {
    const VtValue value = GetRenderSetting(_lrtSettings->visibility);
    std::string name;
    if (value.IsHolding<TfToken>()) {
        name = value.UncheckedGet<TfToken>().GetString();
    } else if (value.IsHolding<std::string>()) {
        name = value.UncheckedGet<std::string>();
    }
    if (name == _lrtSettings->raster.GetString()) return lrt::usd::MeshVisibility::Raster;
    if (name == _lrtSettings->rays.GetString()) return lrt::usd::MeshVisibility::Rays;
    if (name == _lrtSettings->bvh.GetString()) return lrt::usd::MeshVisibility::Bvh;
    return lrt::usd::MeshVisibility::Automatic;
}

bool HdLrtRenderDelegate::GetChooseLights() const {
    const VtValue value = GetRenderSetting(_lrtSettings->chooseLights);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

uint32_t HdLrtRenderDelegate::GetLightSamples() const {
    const VtValue value = GetRenderSetting(_lrtSettings->lightSamples);
    if (value.IsHolding<int>()) {
        return static_cast<uint32_t>(std::max(value.UncheckedGet<int>(), 1));
    }
    if (value.IsHolding<unsigned int>()) {
        return std::max(value.UncheckedGet<unsigned int>(), 1u);
    }
    return 1;
}

namespace {
/// An int render setting, however the host spelled its type.
uint32_t _UintSetting(const VtValue& value, uint32_t fallback, uint32_t least) {
    if (value.IsHolding<int>()) {
        return std::max(static_cast<uint32_t>(std::max(value.UncheckedGet<int>(), 0)), least);
    }
    if (value.IsHolding<unsigned int>()) {
        return std::max(value.UncheckedGet<unsigned int>(), least);
    }
    return fallback;
}
}   // namespace

uint32_t HdLrtRenderDelegate::GetPathSamples() const {
    return _UintSetting(GetRenderSetting(_lrtSettings->pathSamples), 1, 1);
}

uint32_t HdLrtRenderDelegate::GetPathBounces() const {
    return _UintSetting(GetRenderSetting(_lrtSettings->pathBounces), 1, 0);
}

bool HdLrtRenderDelegate::GetDenoise() const {
    const VtValue value = GetRenderSetting(_lrtSettings->denoise);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

uint32_t HdLrtRenderDelegate::GetPathTotal() const {
    return _UintSetting(GetRenderSetting(_lrtSettings->pathTotal), 1, 1);
}

bool HdLrtRenderDelegate::GetSettleStreams() const {
    const VtValue value = GetRenderSetting(_lrtSettings->settleStreams);
    return value.IsHolding<bool>() && value.UncheckedGet<bool>();
}

lrt::usd::Technique HdLrtRenderDelegate::GetTechnique() const {
    const VtValue value = GetRenderSetting(_lrtSettings->technique);
    std::string name;
    if (value.IsHolding<TfToken>()) {
        name = value.UncheckedGet<TfToken>().GetString();
    } else if (value.IsHolding<std::string>()) {
        name = value.UncheckedGet<std::string>();
    }
    return name == _lrtSettings->rt.GetString() ? lrt::usd::Technique::RayTraced
                                                : lrt::usd::Technique::Raster;
}

HdRprim* HdLrtRenderDelegate::CreateRprim(TfToken const& typeId, SdfPath const& id) {
    if (typeId == HdPrimTypeTokens->particleField) {
        return new HdLrtParticleField(id);
    }
    if (typeId == HdPrimTypeTokens->points) {
        return new HdLrtPoints(id);
    }
    if (typeId == HdPrimTypeTokens->mesh) {
        return new HdLrtMesh(id);
    }
    return nullptr;
}

void HdLrtRenderDelegate::DestroyRprim(HdRprim* rprim) { delete rprim; }

HdInstancer* HdLrtRenderDelegate::CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) {
    return new HdLrtInstancer(delegate, id);
}

void HdLrtRenderDelegate::DestroyInstancer(HdInstancer* instancer) { delete instancer; }

HdSprim* HdLrtRenderDelegate::CreateSprim(TfToken const& typeId, SdfPath const& id) {
    if (typeId == HdPrimTypeTokens->material) {
        return new HdLrtMaterial(id);
    }
    if (HdPrimTypeIsLight(typeId)) {
        return new HdLrtLight(typeId, id);
    }
    return typeId == HdPrimTypeTokens->camera ? new HdCamera(id) : nullptr;
}

HdSprim* HdLrtRenderDelegate::CreateFallbackSprim(TfToken const& typeId) {
    if (typeId == HdPrimTypeTokens->material) {
        return new HdLrtMaterial(SdfPath::EmptyPath());
    }
    if (HdPrimTypeIsLight(typeId)) {
        return new HdLrtLight(typeId, SdfPath::EmptyPath());
    }
    return typeId == HdPrimTypeTokens->camera ? new HdCamera(SdfPath::EmptyPath()) : nullptr;
}

void HdLrtRenderDelegate::DestroySprim(HdSprim* sprim) { delete sprim; }

HdBprim* HdLrtRenderDelegate::CreateBprim(TfToken const& typeId, SdfPath const& id) {
    return typeId == HdPrimTypeTokens->renderBuffer ? new HdLrtRenderBuffer(id, _engine.get()) : nullptr;
}

HdBprim* HdLrtRenderDelegate::CreateFallbackBprim(TfToken const& typeId) {
    return typeId == HdPrimTypeTokens->renderBuffer ? new HdLrtRenderBuffer(SdfPath::EmptyPath(), _engine.get())
                                                    : nullptr;
}

void HdLrtRenderDelegate::DestroyBprim(HdBprim* bprim) { delete bprim; }

HdAovDescriptor HdLrtRenderDelegate::GetDefaultAovDescriptor(TfToken const& name) const {
    if (name == HdAovTokens->color) {
        return HdAovDescriptor(HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.0F)));
    }
    if (name == HdAovTokens->depth) {
        return HdAovDescriptor(HdFormatFloat32, false, VtValue(1.0F));
    }
    if (name == HdAovTokens->primId || name == HdAovTokens->instanceId || name == HdAovTokens->elementId) {
        return HdAovDescriptor(HdFormatInt32, false, VtValue(-1));
    }
    if (name == HdAovTokens->Neye || name == HdAovTokens->normal ||
        name.GetString().rfind("primvars:", 0) == 0) {
        return HdAovDescriptor(HdFormatFloat32Vec3, false, VtValue(GfVec3f(0.0F)));
    }
    return HdAovDescriptor();
}

PXR_NAMESPACE_CLOSE_SCOPE
