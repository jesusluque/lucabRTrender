// Copyright (c) 2026 lucabRTrender contributors.
#include "Mesh.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/renderIndex.h>
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
           HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex | HdChangeTracker::DirtyMaterialId;
}

void HdLrtMesh::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                     TfToken const&) {
    SdfPath const& id = GetId();
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
        return;
    }
    _UpdateInstancer(delegate, dirtyBits);
    HdInstancer::_SyncInstancerAndParents(delegate->GetRenderIndex(), GetInstancerId());
    std::optional<std::vector<lrt::usd::InstancerLink>> instancing;
    if ((*dirtyBits & (HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex)) != 0) {
        // The chain of instancers above this prototype, innermost first, with
        // the elements each level takes of the one above.
        std::vector<lrt::usd::InstancerLink> chain;
        SdfPath child = id;
        SdfPath instancer = GetInstancerId();
        while (!instancer.IsEmpty()) {
            chain.push_back({instancer, delegate->GetInstanceIndices(instancer, child)});
            HdInstancer* level = delegate->GetRenderIndex().GetInstancer(instancer);
            child = instancer;
            instancer = level != nullptr ? level->GetParentId() : SdfPath();
        }
        instancing = std::move(chain);
    }

    std::optional<lrt::usd::MeshArrays> arrays;
    if ((*dirtyBits & (HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTopology |
                       HdChangeTracker::DirtyDisplayStyle | HdChangeTracker::DirtyNormals |
                       HdChangeTracker::DirtyPrimvar)) != 0) {
        lrt::usd::MeshArrays a;
        const HdMeshTopology topology = GetMeshTopology(delegate);
        a.faceVertexCounts = topology.GetFaceVertexCounts();
        a.faceVertexIndices = topology.GetFaceVertexIndices();
        a.holeIndices = topology.GetHoleIndices();
        a.leftHanded = topology.GetOrientation() != HdTokens->rightHanded;
        a.points = delegate->Get(id, HdTokens->points);
        // Smooth normals where Storm computes them: a subdivision scheme that
        // is not none or bilinear, and no flat shading asked for.
        // Every numeric primvar, as it is (indices resolved on the device).
        for (const HdInterpolation interpolation : {HdInterpolationConstant, HdInterpolationUniform,
                                                    HdInterpolationVarying, HdInterpolationVertex,
                                                    HdInterpolationFaceVarying}) {
            for (const HdPrimvarDescriptor& descriptor : GetPrimvarDescriptors(delegate, interpolation)) {
                if (descriptor.name == HdTokens->points) {
                    continue;
                }
                lrt::usd::PrimvarArrays primvar;
                primvar.name = descriptor.name.GetString();
                primvar.interpolation = static_cast<uint32_t>(interpolation);
                primvar.values = descriptor.indexed
                                     ? delegate->GetIndexedPrimvar(id, descriptor.name, &primvar.indices)
                                     : GetPrimvar(delegate, descriptor.name);
                uint32_t components = 0;
                if (!lrt::usd::primvarStreamOf(primvar.values, &components).empty()) {
                    a.primvars.push_back(std::move(primvar));
                }
            }
        }
        const HdDisplayStyle style = GetDisplayStyle(delegate);
        a.smoothNormals = !style.flatShadingEnabled && topology.GetScheme() != PxOsdOpenSubdivTokens->none &&
                          topology.GetScheme() != PxOsdOpenSubdivTokens->bilinear;
        arrays = std::move(a);
    }
    std::optional<lrt::usd::MeshLook> look;
    if (*dirtyBits & HdChangeTracker::DirtyMaterialId) {
        SetMaterialId(delegate->GetMaterialId(id));
    }
    if ((*dirtyBits & (HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyDoubleSided |
                       HdChangeTracker::DirtyMaterialId)) != 0) {
        lrt::usd::MeshLook l;
        l.material = GetMaterialId();
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
                    visible, look, std::move(instancing));
    *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
}

void HdLrtMesh::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->remove(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
