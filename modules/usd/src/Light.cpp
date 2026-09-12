// Copyright (c) 2026 lucabRTrender contributors.
#include "Light.h"

#include <cmath>

#include <pxr/imaging/hd/sceneDelegate.h>

#include "RenderDelegate.h"
#include "lrt/core/Log.h"
#include "lrt/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

float floatOf(HdSceneDelegate* delegate, const SdfPath& id, const TfToken& name, float fallback) {
    const VtValue value = delegate->GetLightParamValue(id, name);
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<int>()) return static_cast<float>(value.UncheckedGet<int>());
    return fallback;
}

bool boolOf(HdSceneDelegate* delegate, const SdfPath& id, const TfToken& name, bool fallback) {
    const VtValue value = delegate->GetLightParamValue(id, name);
    if (value.IsHolding<bool>()) return value.UncheckedGet<bool>();
    if (value.IsHolding<int>()) return value.UncheckedGet<int>() != 0;
    return fallback;
}

/// The light's kind, and whether the engine draws it at all.
bool kindOf(const TfToken& type, lrt::light::LightKind& kind) {
    if (type == HdPrimTypeTokens->sphereLight) kind = lrt::light::LightKind::Sphere;
    else if (type == HdPrimTypeTokens->diskLight) kind = lrt::light::LightKind::Disk;
    else if (type == HdPrimTypeTokens->rectLight) kind = lrt::light::LightKind::Rect;
    else if (type == HdPrimTypeTokens->distantLight) kind = lrt::light::LightKind::Distant;
    else if (type == HdPrimTypeTokens->domeLight) kind = lrt::light::LightKind::Dome;
    else if (type == HdPrimTypeTokens->cylinderLight) kind = lrt::light::LightKind::Cylinder;
    else return false;
    return true;
}

}   // namespace

void HdLrtLight::Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr || *dirtyBits == Clean) {
        *dirtyBits = Clean;
        return;
    }
    const SdfPath& id = GetId();
    lrt::light::Light lamp;
    if (!kindOf(_type, lamp.kind)) {
        lrt::log::warn("hdLrt: light {}: {} is not a light this engine draws yet", id.GetString(),
                       _type.GetString());
        *dirtyBits = Clean;
        return;
    }
    lamp.lightToWorld = lrt::usd::fromUsd(sceneDelegate->GetTransform(id));
    const VtValue colour = sceneDelegate->GetLightParamValue(id, HdLightTokens->color);
    if (colour.IsHolding<GfVec3f>()) {
        const GfVec3f& c = colour.UncheckedGet<GfVec3f>();
        lamp.colour[0] = c[0];
        lamp.colour[1] = c[1];
        lamp.colour[2] = c[2];
    }
    lamp.intensity = floatOf(sceneDelegate, id, HdLightTokens->intensity, 1.0F);
    lamp.exposure = floatOf(sceneDelegate, id, HdLightTokens->exposure, 0.0F);
    lamp.radius = floatOf(sceneDelegate, id, HdLightTokens->radius, 0.5F);
    lamp.width = floatOf(sceneDelegate, id, HdLightTokens->width, 1.0F);
    lamp.height = floatOf(sceneDelegate, id, HdLightTokens->height, 1.0F);
    lamp.length = floatOf(sceneDelegate, id, HdLightTokens->length, 1.0F);
    // USD authors angles in degrees.
    const float degrees = 3.14159265358979F / 180.0F;
    lamp.angle = floatOf(sceneDelegate, id, HdLightTokens->angle, 0.53F) * degrees;
    lamp.temperature = floatOf(sceneDelegate, id, HdLightTokens->colorTemperature, 6500.0F);
    lamp.enableTemperature = boolOf(sceneDelegate, id, HdLightTokens->enableColorTemperature, false);
    lamp.normalize = boolOf(sceneDelegate, id, HdLightTokens->normalize, false);
    lamp.shadow = boolOf(sceneDelegate, id, HdLightTokens->shadowEnable, true);
    lamp.coneAngle = floatOf(sceneDelegate, id, HdLightTokens->shapingConeAngle, 0.0F) * degrees;
    lamp.coneSoftness = floatOf(sceneDelegate, id, HdLightTokens->shapingConeSoftness, 0.0F);
    const auto tokenOf = [&](const TfToken& name) -> std::string {
        const VtValue value = sceneDelegate->GetLightParamValue(id, name);
        if (value.IsHolding<TfToken>()) return value.UncheckedGet<TfToken>().GetString();
        if (value.IsHolding<std::string>()) return value.UncheckedGet<std::string>();
        return {};
    };
    lamp.lightLink = tokenOf(HdTokens->lightLink);
    lamp.shadowLink = tokenOf(HdTokens->shadowLink);
    if (lamp.kind == lrt::light::LightKind::Dome) {
        const VtValue file = sceneDelegate->GetLightParamValue(id, HdLightTokens->textureFile);
        if (file.IsHolding<SdfAssetPath>()) {
            const SdfAssetPath& asset = file.UncheckedGet<SdfAssetPath>();
            lamp.texture = !asset.GetResolvedPath().empty() ? asset.GetResolvedPath() : asset.GetAssetPath();
        }
    }
    engine->setLight(id, lamp);
    *dirtyBits = Clean;
}

void HdLrtLight::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->removeLight(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
