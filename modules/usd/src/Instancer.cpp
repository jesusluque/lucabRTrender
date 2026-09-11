// Copyright (c) 2026 lucabRTrender contributors.
#include "Instancer.h"

#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdLrtInstancer::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
    _UpdateInstancer(delegate, dirtyBits);
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        return;
    }
    SdfPath const& id = GetId();
    if (HdChangeTracker::IsAnyPrimvarDirty(*dirtyBits, id) || HdChangeTracker::IsTransformDirty(*dirtyBits, id) ||
        (*dirtyBits & HdChangeTracker::DirtyInstancer) != 0) {
        lrt::usd::InstancerArrays a;
        a.translations = delegate->Get(id, HdInstancerTokens->instanceTranslations);
        a.rotations = delegate->Get(id, HdInstancerTokens->instanceRotations);
        a.scales = delegate->Get(id, HdInstancerTokens->instanceScales);
        a.transforms = delegate->Get(id, HdInstancerTokens->instanceTransforms);
        a.instancerTransform = lrt::usd::fromUsd(delegate->GetInstancerTransform(id));
        engine->setInstancer(id, GetParentId(), std::move(a));
    }
}

void HdLrtInstancer::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->removeInstancer(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
