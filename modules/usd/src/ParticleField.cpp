// Copyright (c) 2026 lucabRTrender contributors.
#include "ParticleField.h"

#include <algorithm>
#include <array>

#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/gf/quath.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdVol/tokens.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"
#include "lrt/usd/PrimData.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// LrtSplatEditAPI's constant primvars (modules/usd/schemas). Constant primvars
// are inherited down the namespace, so an edit authored on an Xform stands
// over every cloud below it -- openFXplayer's Edit node over its subtree.
TF_DEFINE_PRIVATE_TOKENS(_editTokens,
    ((active, "lrt:edit:active"))
    ((shape, "lrt:edit:shape"))
    ((mode, "lrt:edit:mode"))
    ((centre, "lrt:edit:centre"))
    ((size, "lrt:edit:size"))
    ((tint, "lrt:edit:tint"))
    ((saturation, "lrt:edit:saturation"))
    ((brightness, "lrt:edit:brightness"))
    ((opacity, "lrt:edit:opacity"))
    ((minOpacity, "lrt:edit:minOpacity"))
    ((maxScale, "lrt:edit:maxScale"))
    ((invert, "lrt:edit:invert"))
    (sphere)(keep)(remove)
);

// LrtStreamedAssetAPI's.
TF_DEFINE_PRIVATE_TOKENS(_assetTokens,
    ((asset, "lrt:asset"))
    ((threshold, "lrt:lod:threshold"))
    ((budget, "lrt:stream:budget"))
);

float floatOf(VtValue const& value, float fallback) {
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty())
        return value.UncheckedGet<VtFloatArray>()[0];
    return fallback;
}

bool boolOf(VtValue const& value, bool fallback) {
    if (value.IsHolding<bool>()) return value.UncheckedGet<bool>();
    if (value.IsHolding<VtBoolArray>() && !value.UncheckedGet<VtBoolArray>().empty())
        return value.UncheckedGet<VtBoolArray>()[0];
    return fallback;
}

std::array<float, 3> vec3Of(VtValue const& value, std::array<float, 3> fallback) {
    if (value.IsHolding<GfVec3f>()) {
        const GfVec3f v = value.UncheckedGet<GfVec3f>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec3d>()) {
        const GfVec3d v = value.UncheckedGet<GfVec3d>();
        return {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
    }
    if (value.IsHolding<VtVec3fArray>() && !value.UncheckedGet<VtVec3fArray>().empty()) {
        const GfVec3f v = value.UncheckedGet<VtVec3fArray>()[0];
        return {v[0], v[1], v[2]};
    }
    return fallback;
}

TfToken tokenOf(VtValue const& value) {
    if (value.IsHolding<TfToken>()) return value.UncheckedGet<TfToken>();
    if (value.IsHolding<VtTokenArray>() && !value.UncheckedGet<VtTokenArray>().empty())
        return value.UncheckedGet<VtTokenArray>()[0];
    if (value.IsHolding<std::string>()) return TfToken(value.UncheckedGet<std::string>());
    return TfToken();
}

lrt::render::SplatEdit editOf(HdSceneDelegate* delegate, SdfPath const& id) {
    using Edit = lrt::render::SplatEdit;
    Edit edit;
    edit.active = boolOf(delegate->Get(id, _editTokens->active), false);
    if (!edit.active) {
        return edit;
    }
    edit.shape = tokenOf(delegate->Get(id, _editTokens->shape)) == _editTokens->sphere ? Edit::Shape::Sphere
                                                                                       : Edit::Shape::Box;
    const TfToken mode = tokenOf(delegate->Get(id, _editTokens->mode));
    edit.mode = mode == _editTokens->keep ? Edit::Mode::Keep
              : mode == _editTokens->remove ? Edit::Mode::Remove
                                            : Edit::Mode::Grade;
    edit.centre = vec3Of(delegate->Get(id, _editTokens->centre), edit.centre);
    edit.size = vec3Of(delegate->Get(id, _editTokens->size), edit.size);
    edit.tint = vec3Of(delegate->Get(id, _editTokens->tint), edit.tint);
    edit.saturation = floatOf(delegate->Get(id, _editTokens->saturation), edit.saturation);
    edit.brightness = floatOf(delegate->Get(id, _editTokens->brightness), edit.brightness);
    edit.opacity = floatOf(delegate->Get(id, _editTokens->opacity), edit.opacity);
    edit.minOpacity = floatOf(delegate->Get(id, _editTokens->minOpacity), edit.minOpacity);
    edit.maxScale = floatOf(delegate->Get(id, _editTokens->maxScale), edit.maxScale);
    edit.invert = boolOf(delegate->Get(id, _editTokens->invert), false);
    return edit;
}

/// LrtSplatLightingAPI: whether this cloud is relit rather than shown as it
/// was baked.
bool relightOf(HdSceneDelegate* delegate, SdfPath const& id) {
    static const TfToken kRelight("lrt:splat:relight");
    return boolOf(delegate->Get(id, kRelight), false);
}

/// Whether its colours are light already: what a conversion bakes into them
/// (`lrt mesh2splat`), so that relighting adds the polish and nothing else.
bool litBodyOf(HdSceneDelegate* delegate, SdfPath const& id) {
    static const TfToken kLit("lrt:splat:litBody");
    return boolOf(delegate->Get(id, kLit), false);
}

lrt::usd::StreamedAsset assetOf(HdSceneDelegate* delegate, SdfPath const& id) {
    lrt::usd::StreamedAsset asset;
    VtValue value = delegate->Get(id, _assetTokens->asset);
    SdfAssetPath path;
    if (value.IsHolding<SdfAssetPath>()) {
        path = value.UncheckedGet<SdfAssetPath>();
    } else if (value.IsHolding<VtArray<SdfAssetPath>>() && !value.UncheckedGet<VtArray<SdfAssetPath>>().empty()) {
        path = value.UncheckedGet<VtArray<SdfAssetPath>>()[0];
    }
    asset.path = !path.GetResolvedPath().empty() ? path.GetResolvedPath() : path.GetAssetPath();
    if (asset.path.empty()) {
        return asset;
    }
    asset.threshold = std::max(floatOf(delegate->Get(id, _assetTokens->threshold), asset.threshold), 0.0F);
    const VtValue budget = delegate->Get(id, _assetTokens->budget);
    if (budget.IsHolding<int64_t>()) {
        asset.budget = static_cast<uint64_t>(std::max<int64_t>(budget.UncheckedGet<int64_t>(), 0));
    } else if (budget.IsHolding<int>()) {
        asset.budget = static_cast<uint64_t>(std::max(budget.UncheckedGet<int>(), 0));
    } else if (budget.IsHolding<VtInt64Array>() && !budget.UncheckedGet<VtInt64Array>().empty()) {
        asset.budget = static_cast<uint64_t>(std::max<int64_t>(budget.UncheckedGet<VtInt64Array>()[0], 0));
    }
    return asset;
}


}   // namespace

HdDirtyBits HdLrtParticleField::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyTransform |
           HdChangeTracker::DirtyVisibility;
}

void HdLrtParticleField::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam,
                              HdDirtyBits* dirtyBits, TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }

    std::optional<lrt::usd::ParticleFieldArrays> raw;
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyPrimvar)) != 0) {
        // The float attribute, or its half twin: kept as they come, halves
        // turned into floats on the device.
        const auto either = [&](TfToken const& full, TfToken const& half) {
            VtValue value = delegate->Get(id, full);
            return lrt::usd::streamOf(value).empty() ? delegate->Get(id, half) : value;
        };
        lrt::usd::ParticleFieldArrays arrays;
        arrays.positions = either(UsdVolTokens->positions, UsdVolTokens->positionsh);
        arrays.orientations = either(UsdVolTokens->orientations, UsdVolTokens->orientationsh);
        arrays.scales = either(UsdVolTokens->scales, UsdVolTokens->scalesh);
        arrays.opacities = either(UsdVolTokens->opacities, UsdVolTokens->opacitiesh);
        VtValue degree = delegate->Get(id, UsdVolTokens->radianceSphericalHarmonicsDegree);
        arrays.shDegree = degree.IsHolding<int>() ? degree.UncheckedGet<int>() : 0;
        arrays.shCoefficients = either(UsdVolTokens->radianceSphericalHarmonicsCoefficients,
                                       UsdVolTokens->radianceSphericalHarmonicsCoefficientsh);
        // LrtSplatLightingAPI's other two: what a relit gaussian reflects
        // with, which a cloud converted from a mesh knows and a capture does
        // not. Primvars, so they arrive without the namespace.
        static const TfToken kMetallic("lrt:splat:metallic");
        static const TfToken kRoughness("lrt:splat:roughness");
        static const TfToken kTransmission("lrt:splat:transmission");
        arrays.metallic = delegate->Get(id, kMetallic);
        arrays.roughness = delegate->Get(id, kRoughness);
        arrays.transmission = delegate->Get(id, kTransmission);
        raw = std::move(arrays);
    }
    std::optional<lrt::render::SplatEdit> edit;
    std::optional<lrt::usd::StreamedAsset> asset;
    if ((*dirtyBits & HdChangeTracker::DirtyPrimvar) != 0) {
        edit = editOf(delegate, id);
        asset = assetOf(delegate, id);
    }

    lrt::render::Mat4 transform;
    const bool transformDirty = HdChangeTracker::IsTransformDirty(*dirtyBits, id);
    if (transformDirty) {
        transform = lrt::usd::fromUsd(delegate->GetTransform(id));
    }
    std::optional<bool> visible;
    if (HdChangeTracker::IsVisibilityDirty(*dirtyBits, id)) {
        _UpdateVisibility(delegate, dirtyBits);
        visible = IsVisible();
    }
    engine->setSplats(id, std::move(raw), transformDirty ? &transform : nullptr, visible, edit, std::move(asset),
                      relightOf(delegate, id),
                      [&] {
                          const VtArray<TfToken> cats = delegate->GetCategories(id);
                          return std::vector<TfToken>(cats.begin(), cats.end());
                      }(),
                      litBodyOf(delegate, id));
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtParticleField::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
