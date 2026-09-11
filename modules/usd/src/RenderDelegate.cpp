// Copyright (c) 2026 lucabRTrender contributors.
#include "RenderDelegate.h"

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
    static const TfTokenVector types{HdPrimTypeTokens->camera};
    return types;
}

TfTokenVector const& HdLrtRenderDelegate::GetSupportedBprimTypes() const {
    static const TfTokenVector types{HdPrimTypeTokens->renderBuffer};
    return types;
}

HdRenderPassSharedPtr HdLrtRenderDelegate::CreateRenderPass(HdRenderIndex* index,
                                                            HdRprimCollection const& collection) {
    return std::make_shared<HdLrtRenderPass>(index, collection, _engine.get(), this);
}

TF_DEFINE_PRIVATE_TOKENS(_lrtSettings, ((technique, "lrt:technique"))((settleStreams, "lrt:settleStreams"))(raster)(rt));

HdRenderSettingDescriptorList HdLrtRenderDelegate::GetRenderSettingDescriptors() const {
    HdRenderSettingDescriptor technique;
    technique.name = "Technique (raster | rt)";
    technique.key = _lrtSettings->technique;
    technique.defaultValue = VtValue(_lrtSettings->raster);
    HdRenderSettingDescriptor settle;
    settle.name = "Wait for streamed assets before drawing";
    settle.key = _lrtSettings->settleStreams;
    settle.defaultValue = VtValue(false);
    return {technique, settle};
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
    return typeId == HdPrimTypeTokens->camera ? new HdCamera(id) : nullptr;
}

HdSprim* HdLrtRenderDelegate::CreateFallbackSprim(TfToken const& typeId) {
    return typeId == HdPrimTypeTokens->camera ? new HdCamera(SdfPath::EmptyPath()) : nullptr;
}

void HdLrtRenderDelegate::DestroySprim(HdSprim* sprim) { delete sprim; }

HdBprim* HdLrtRenderDelegate::CreateBprim(TfToken const& typeId, SdfPath const& id) {
    return typeId == HdPrimTypeTokens->renderBuffer ? new HdLrtRenderBuffer(id) : nullptr;
}

HdBprim* HdLrtRenderDelegate::CreateFallbackBprim(TfToken const& typeId) {
    return typeId == HdPrimTypeTokens->renderBuffer ? new HdLrtRenderBuffer(SdfPath::EmptyPath())
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
    return HdAovDescriptor();
}

PXR_NAMESPACE_CLOSE_SCOPE
