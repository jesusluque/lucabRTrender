// Copyright (c) 2026 lucabRTrender contributors.
#include "Light.h"

#include <pxr/imaging/hd/instancer.h>

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
    // A light under an instancer, as a mesh under one: the chain of
    // instancers above it, innermost first, with the elements each level
    // takes of the one above.
    _UpdateInstancer(sceneDelegate, dirtyBits);
    HdInstancer::_SyncInstancerAndParents(sceneDelegate->GetRenderIndex(), GetInstancerId());
    std::vector<lrt::usd::InstancerLink> instancing;
    {
        SdfPath child = GetId();
        SdfPath instancer = GetInstancerId();
        while (!instancer.IsEmpty()) {
            instancing.push_back({instancer, sceneDelegate->GetInstanceIndices(instancer, child)});
            HdInstancer* level = sceneDelegate->GetRenderIndex().GetInstancer(instancer);
            child = instancer;
            instancer = level != nullptr ? level->GetParentId() : SdfPath();
        }
    }
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
    // Under a shutter, its transform at the samples about it, as a mesh's.
    {
        const auto* param = static_cast<HdLrtRenderParam*>(renderParam);
        const float open = static_cast<float>(param->GetShutterOpen());
        const float close = static_cast<float>(param->GetShutterClose());
        if (close > open) {
            float times[2] = {0.0F, 0.0F};
            GfMatrix4d values[2];
            const size_t n = sceneDelegate->SampleTransform(id, open, close, 2, times, values);
            if (n >= 2 && values[0] != values[n - 1]) {
                lamp.moves = true;
                lamp.lightToWorldStart = lrt::usd::fromUsd(values[0]);
                lamp.lightToWorldEnd = lrt::usd::fromUsd(values[n - 1]);
                lamp.timeStart = times[0];
                lamp.timeEnd = times[n - 1];
            }
        }
    }
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
    // UsdLux's IES shaping: the file, resolved as the dome's image is; the
    // engine reads it once per path.
    {
        const VtValue ies = sceneDelegate->GetLightParamValue(id, HdLightTokens->shapingIesFile);
        if (ies.IsHolding<SdfAssetPath>()) {
            const SdfAssetPath& asset = ies.UncheckedGet<SdfAssetPath>();
            lamp.iesFile = !asset.GetResolvedPath().empty() ? asset.GetResolvedPath() : asset.GetAssetPath();
        }
        lamp.iesAngleScale = floatOf(sceneDelegate, id, HdLightTokens->shapingIesAngleScale, 0.0F);
        lamp.iesNormalize = boolOf(sceneDelegate, id, HdLightTokens->shapingIesNormalize, false);
    }
    const auto tokenOf = [&](const TfToken& name) -> std::string {
        const VtValue value = sceneDelegate->GetLightParamValue(id, name);
        if (value.IsHolding<TfToken>()) return value.UncheckedGet<TfToken>().GetString();
        if (value.IsHolding<std::string>()) return value.UncheckedGet<std::string>();
        return {};
    };
    lamp.lightLink = tokenOf(HdTokens->lightLink);
    lamp.shadowLink = tokenOf(HdTokens->shadowLink);
    // Its light group: ours, or RenderMan's, as authored on the prim.
    lamp.group = tokenOf(TfToken("lrt:lightGroup"));
    if (lamp.group.empty()) {
        lamp.group = tokenOf(TfToken("ri:light:lightGroup"));
    }
    if (lamp.kind == lrt::light::LightKind::Dome) {
        const VtValue file = sceneDelegate->GetLightParamValue(id, HdLightTokens->textureFile);
        if (file.IsHolding<SdfAssetPath>()) {
            const SdfAssetPath& asset = file.UncheckedGet<SdfAssetPath>();
            lamp.texture = !asset.GetResolvedPath().empty() ? asset.GetResolvedPath() : asset.GetAssetPath();
        }
    }
    engine->setLight(id, lamp, std::move(instancing));
    *dirtyBits = Clean;
}

void HdLrtLight::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->removeLight(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
