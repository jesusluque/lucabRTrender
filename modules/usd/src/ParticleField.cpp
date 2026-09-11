// Copyright (c) 2026 lucabRTrender contributors.
#include "ParticleField.h"

#include <pxr/base/gf/vec3h.h>
#include <pxr/base/gf/quath.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/usdVol/tokens.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"
#include "lrt/usd/PrimData.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

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
    engine->setSplats(id, std::move(raw), transformDirty ? &transform : nullptr, visible);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtParticleField::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
