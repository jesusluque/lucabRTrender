// Copyright (c) 2026 lucabRTrender contributors.
#include "Mesh.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/meshTopology.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/tokens.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// A constant primvar's value (or the first of an array), as float3.
bool firstVec3(VtValue const& value, GfVec3f* out) {
    if (value.IsHolding<GfVec3f>()) {
        *out = value.UncheckedGet<GfVec3f>();
        return true;
    }
    if (value.IsHolding<VtVec3fArray>() && !value.UncheckedGet<VtVec3fArray>().empty()) {
        *out = value.UncheckedGet<VtVec3fArray>()[0];
        return true;
    }
    return false;
}

bool firstFloat(VtValue const& value, float* out) {
    if (value.IsHolding<float>()) {
        *out = value.UncheckedGet<float>();
        return true;
    }
    if (value.IsHolding<VtFloatArray>() && !value.UncheckedGet<VtFloatArray>().empty()) {
        *out = value.UncheckedGet<VtFloatArray>()[0];
        return true;
    }
    return false;
}

}   // namespace

TfTokenVector const& HdLrtMesh::GetBuiltinPrimvarNames() const {
    static const TfTokenVector names{HdTokens->points, HdTokens->normals, HdTokens->displayColor,
                                     HdTokens->displayOpacity};
    return names;
}

HdDirtyBits HdLrtMesh::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::Clean | HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility |
           HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyDoubleSided |
           HdChangeTracker::DirtyDisplayStyle | HdChangeTracker::DirtyRenderTag |
           HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex;
}

void HdLrtMesh::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                     TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }

    std::optional<lrt::usd::MeshArrays> arrays;
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTopology |
                       HdChangeTracker::DirtyDisplayStyle | HdChangeTracker::DirtyNormals)) != 0) {
        lrt::usd::MeshArrays a;
        const HdMeshTopology topology = GetMeshTopology(delegate);
        a.faceVertexCounts = topology.GetFaceVertexCounts();
        a.faceVertexIndices = topology.GetFaceVertexIndices();
        a.holeIndices = topology.GetHoleIndices();
        a.leftHanded = topology.GetOrientation() != HdTokens->rightHanded;
        a.points = delegate->Get(id, HdTokens->points);
        // Smooth normals where Storm computes them: a subdivision scheme that
        // is not none or bilinear, and no flat shading asked for.
        const HdDisplayStyle style = GetDisplayStyle(delegate);
        a.smoothNormals = !style.flatShadingEnabled && topology.GetScheme() != PxOsdOpenSubdivTokens->none &&
                          topology.GetScheme() != PxOsdOpenSubdivTokens->bilinear;
        arrays = std::move(a);
    }
    std::optional<lrt::usd::MeshLook> look;
    if ((*dirtyBits & (HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyDoubleSided)) != 0) {
        lrt::usd::MeshLook l;
        GfVec3f colour;
        if (firstVec3(delegate->Get(id, HdTokens->displayColor), &colour)) {
            l.displayColor = {colour[0], colour[1], colour[2]};
        }
        float opacity = 1.0F;
        if (firstFloat(delegate->Get(id, HdTokens->displayOpacity), &opacity)) {
            l.displayOpacity = opacity;
        }
        l.doubleSided = IsDoubleSided(delegate);
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
    engine->setMesh(id, GetPrimId(), GetRenderTag(), std::move(arrays), transformDirty ? &transform : nullptr,
                    visible, look);
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtMesh::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
