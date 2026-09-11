// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <pxr/imaging/hd/instancer.h>

PXR_NAMESPACE_OPEN_SCOPE

/// A Hydra instancer (PointInstancer, native instancing): its primvars handed
/// to the engine as they are, composed into instance transforms on the device.
class HdLrtInstancer final : public HdInstancer {
public:
    HdLrtInstancer(HdSceneDelegate* delegate, SdfPath const& id) : HdInstancer(delegate, id) {}

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;
    void Finalize(HdRenderParam* renderParam) override;
};

PXR_NAMESPACE_CLOSE_SCOPE
