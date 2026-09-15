// Copyright (c) 2026 lucabRTrender contributors.
#include "Volume.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/sdf/assetPath.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"
#include "lrt/usd/PrimData.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

TF_DEFINE_PRIVATE_TOKENS(_lrtVolumeTokens,
    (density)
    ((densityScale, "lrt:densityScale"))
    ((albedo, "lrt:albedo"))
    ((anisotropy, "lrt:anisotropy"))
);

float floatOf(VtValue const& value, float fallback) {
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty())
        return value.UncheckedGet<VtFloatArray>()[0];
    return fallback;
}

}   // namespace

HdDirtyBits HdLrtVolume::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility | HdChangeTracker::DirtyVolumeField;
}

void HdLrtVolume::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                       TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }
    lrt::usd::VolumeArrays arrays;
    // The medium is the field named density, else the first field.
    for (const HdVolumeFieldDescriptor& field : delegate->GetVolumeFieldDescriptors(id)) {
        if (arrays.field.IsEmpty() || field.fieldName == _lrtVolumeTokens->density) {
            arrays.field = field.fieldId;
            arrays.fieldName = field.fieldName.GetString();
        }
    }
    arrays.objectToWorld = lrt::usd::fromUsd(delegate->GetTransform(id));
    arrays.densityScale = floatOf(delegate->Get(id, _lrtVolumeTokens->densityScale), 1.0F);
    const VtValue albedo = delegate->Get(id, _lrtVolumeTokens->albedo);
    if (albedo.IsHolding<GfVec3f>()) {
        const GfVec3f& a = albedo.UncheckedGet<GfVec3f>();
        arrays.albedo = {a[0], a[1], a[2]};
    } else if (albedo.IsHolding<VtVec3fArray>() && !albedo.UncheckedGet<VtVec3fArray>().empty()) {
        const GfVec3f& a = albedo.UncheckedGet<VtVec3fArray>()[0];
        arrays.albedo = {a[0], a[1], a[2]};
    }
    arrays.g = floatOf(delegate->Get(id, _lrtVolumeTokens->anisotropy), 0.0F);
    _UpdateVisibility(delegate, dirtyBits);
    arrays.visible = IsVisible();
    engine->setVolume(id, std::move(arrays));
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtVolume::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->removeVolume(GetId());
    }
}

void HdLrtVolumeField::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine != nullptr && (*dirtyBits & DirtyParams) != 0) {
        const SdfPath& id = GetId();
        lrt::usd::VolumeFieldAsset asset;
        const VtValue file = delegate->Get(id, HdFieldTokens->filePath);
        if (file.IsHolding<SdfAssetPath>()) {
            const SdfAssetPath& path = file.UncheckedGet<SdfAssetPath>();
            asset.path = !path.GetResolvedPath().empty() ? path.GetResolvedPath() : path.GetAssetPath();
        }
        const VtValue name = delegate->Get(id, HdFieldTokens->fieldName);
        if (name.IsHolding<TfToken>()) {
            asset.gridName = name.UncheckedGet<TfToken>().GetString();
        } else if (name.IsHolding<std::string>()) {
            asset.gridName = name.UncheckedGet<std::string>();
        }
        engine->setVolumeField(id, std::move(asset));
    }
    *dirtyBits = Clean;
}

void HdLrtVolumeField::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->removeVolumeField(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
