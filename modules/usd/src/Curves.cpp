// Copyright (c) 2026 lucabRTrender contributors.
#include "Curves.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/basisCurvesTopology.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

TfTokenVector const& HdLrtBasisCurves::GetBuiltinPrimvarNames() const {
    static const TfTokenVector names{HdTokens->points, HdTokens->widths, HdTokens->displayColor,
                                     HdTokens->displayOpacity};
    return names;
}

HdDirtyBits HdLrtBasisCurves::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyWidths | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility | HdChangeTracker::DirtyRenderTag |
           HdChangeTracker::DirtyMaterialId;
}

void HdLrtBasisCurves::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                            TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }
    std::optional<lrt::usd::CurveArrays> arrays;
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyWidths |
                       HdChangeTracker::DirtyPrimvar)) != 0) {
        lrt::usd::CurveArrays a;
        const HdBasisCurvesTopology topology = delegate->GetBasisCurvesTopology(id);
        a.curveVertexCounts = topology.GetCurveVertexCounts();
        a.curveIndices = topology.GetCurveIndices();
        a.type = topology.GetCurveType();
        a.basis = topology.GetCurveBasis();
        a.wrap = topology.GetCurveWrap();
        a.points = delegate->Get(id, HdTokens->points);
        a.widths = delegate->Get(id, HdTokens->widths);
        for (const HdInterpolation interpolation : {HdInterpolationConstant, HdInterpolationUniform,
                                                    HdInterpolationVarying, HdInterpolationVertex}) {
            for (const HdPrimvarDescriptor& descriptor : GetPrimvarDescriptors(delegate, interpolation)) {
                if (descriptor.name == HdTokens->widths) {
                    a.widthsInterpolation = static_cast<uint32_t>(interpolation);
                }
                if (descriptor.name == HdTokens->points || descriptor.name == HdTokens->widths) {
                    continue;
                }
                lrt::usd::PrimvarArrays primvar;
                primvar.name = descriptor.name.GetString();
                primvar.interpolation = static_cast<uint32_t>(interpolation);
                primvar.values = GetPrimvar(delegate, descriptor.name);
                uint32_t components = 0;
                if (!lrt::usd::primvarStreamOf(primvar.values, &components).empty()) {
                    a.primvars.push_back(std::move(primvar));
                }
            }
        }
        a.topologyChanged = (*dirtyBits & HdChangeTracker::DirtyTopology) != 0;
        arrays = std::move(a);
    }
    std::optional<lrt::usd::MeshLook> look;
    if (*dirtyBits & HdChangeTracker::DirtyMaterialId) {
        SetMaterialId(delegate->GetMaterialId(id));
    }
    if ((*dirtyBits & (HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyMaterialId)) != 0) {
        lrt::usd::MeshLook l;
        l.material = GetMaterialId();
        const VtValue colour = delegate->Get(id, HdTokens->displayColor);
        if (colour.IsHolding<VtVec3fArray>() && !colour.UncheckedGet<VtVec3fArray>().empty()) {
            const GfVec3f c = colour.UncheckedGet<VtVec3fArray>()[0];
            l.displayColor = {c[0], c[1], c[2]};
        } else if (colour.IsHolding<GfVec3f>()) {
            const GfVec3f c = colour.UncheckedGet<GfVec3f>();
            l.displayColor = {c[0], c[1], c[2]};
        }
        l.doubleSided = true;   // a tube has no back to cull
        const VtArray<TfToken> categories = delegate->GetCategories(id);
        l.categories.assign(categories.begin(), categories.end());
        look = l;
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
    engine->setCurves(id, GetPrimId(), GetRenderTag(), std::move(arrays), transformDirty ? &transform : nullptr,
                      visible, look);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtBasisCurves::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
