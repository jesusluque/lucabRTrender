// Copyright (c) 2026 lucabRTrender contributors.
//
// A UsdLux light as the engine has it: the sprim reads what the scene
// delegate authored and hands it over as a light::Light, in world space and
// in USD's units.
#pragma once

#include <pxr/imaging/hd/light.h>
#include <pxr/pxr.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdLrtLight final : public HdLight {
public:
    HdLrtLight(const TfToken& type, const SdfPath& id) : HdLight(id), _type(type) {}

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;
    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
    void Finalize(HdRenderParam* renderParam) override;

private:
    TfToken _type;
};

PXR_NAMESPACE_CLOSE_SCOPE
