// Copyright (c) 2026 lucabRTrender contributors.
#include "ParticleField.h"

#include <array>

#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/gf/quath.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
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

VtVec3fArray vec3fOf(VtValue const& value) {
    if (value.IsHolding<VtVec3fArray>()) {
        return value.UncheckedGet<VtVec3fArray>();
    }
    if (value.IsHolding<VtVec3hArray>()) {
        const auto& h = value.UncheckedGet<VtVec3hArray>();
        return VtVec3fArray(h.begin(), h.end());
    }
    return {};
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

    std::optional<lrt::io::RawSplats> raw;
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyPrimvar)) != 0) {
        lrt::usd::ParticleFieldArrays arrays;
        arrays.positions = vec3fOf(delegate->Get(id, UsdVolTokens->positions));
        if (arrays.positions.empty()) {
            arrays.positions = vec3fOf(delegate->Get(id, UsdVolTokens->positionsh));
        }
        VtValue orientations = delegate->Get(id, UsdVolTokens->orientations);
        if (orientations.IsHolding<VtQuatfArray>()) {
            arrays.orientations = orientations.UncheckedGet<VtQuatfArray>();
        } else if (VtValue h = delegate->Get(id, UsdVolTokens->orientationsh); h.IsHolding<VtQuathArray>()) {
            const auto& q = h.UncheckedGet<VtQuathArray>();
            arrays.orientations = VtQuatfArray(q.begin(), q.end());
        }
        arrays.scales = vec3fOf(delegate->Get(id, UsdVolTokens->scales));
        if (arrays.scales.empty()) {
            arrays.scales = vec3fOf(delegate->Get(id, UsdVolTokens->scalesh));
        }
        VtValue opacities = delegate->Get(id, UsdVolTokens->opacities);
        if (opacities.IsHolding<VtFloatArray>()) {
            arrays.opacities = opacities.UncheckedGet<VtFloatArray>();
        } else if (VtValue h = delegate->Get(id, UsdVolTokens->opacitiesh); h.IsHolding<VtHalfArray>()) {
            const auto& o = h.UncheckedGet<VtHalfArray>();
            arrays.opacities = VtFloatArray(o.begin(), o.end());
        }
        VtValue degree = delegate->Get(id, UsdVolTokens->radianceSphericalHarmonicsDegree);
        arrays.shDegree = degree.IsHolding<int>() ? degree.UncheckedGet<int>() : 0;
        arrays.shCoefficients = vec3fOf(delegate->Get(id, UsdVolTokens->radianceSphericalHarmonicsCoefficients));
        if (arrays.shCoefficients.empty()) {
            arrays.shCoefficients =
                vec3fOf(delegate->Get(id, UsdVolTokens->radianceSphericalHarmonicsCoefficientsh));
        }
        raw = lrt::usd::rawSplatsFrom(arrays, id.GetString());
    }
    std::optional<lrt::render::SplatEdit> edit;
    if ((*dirtyBits & HdChangeTracker::DirtyPrimvar) != 0) {
        edit = editOf(delegate, id);
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
    engine->setSplats(id, std::move(raw), transformDirty ? &transform : nullptr, visible, edit);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtParticleField::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
