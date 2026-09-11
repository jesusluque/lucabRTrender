// Copyright (c) 2026 lucabRTrender contributors.
#include "Points.h"

#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"
#include "lrt/usd/PrimData.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

TF_DEFINE_PRIVATE_TOKENS(_lrtTokens,
    ((sizeInPixels, "lrt:sizeInPixels"))
    ((edl, "lrt:edl"))
    ((surfaceOffset, "lrt:surfaceOffset"))
);

float firstFloat(VtValue const& value, float fallback) {
    if (value.IsHolding<float>()) return value.UncheckedGet<float>();
    if (value.IsHolding<double>()) return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty())
        return value.UncheckedGet<VtFloatArray>()[0];
    return fallback;
}

}   // namespace

TfTokenVector const& HdLrtPoints::GetBuiltinPrimvarNames() const {
    static const TfTokenVector names{HdTokens->points, HdTokens->widths, HdTokens->displayColor};
    return names;
}

HdDirtyBits HdLrtPoints::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyWidths | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility;
}

void HdLrtPoints::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam,
                       HdDirtyBits* dirtyBits, TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }
    std::optional<lrt::usd::PointsArrays> raw;
    std::optional<lrt::render::PointStyle> style;
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyPrimvar |
                       HdChangeTracker::DirtyWidths)) != 0) {
        lrt::usd::PointsArrays arrays;
        arrays.positions = delegate->Get(id, HdTokens->points);
        arrays.colours = delegate->Get(id, HdTokens->displayColor);
        raw = std::move(arrays);

        lrt::render::PointStyle s;
        s.size = firstFloat(delegate->Get(id, HdTokens->widths), 0.01F);
        const float pixels = firstFloat(delegate->Get(id, _lrtTokens->sizeInPixels), 0.0F);
        if (pixels > 0.0F) {
            s.sizeMode = lrt::render::PointStyle::Size::Pixels;
            s.size = pixels;
        }
        s.edlStrength = firstFloat(delegate->Get(id, _lrtTokens->edl), 0.0F);
        s.surfaceDepthOffset = firstFloat(delegate->Get(id, _lrtTokens->surfaceOffset), 0.0F);
        style = s;
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
    engine->setPoints(id, std::move(raw), transformDirty ? &transform : nullptr, visible, style);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtPoints::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
