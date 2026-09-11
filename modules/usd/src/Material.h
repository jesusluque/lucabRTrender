// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <pxr/imaging/hd/material.h>

PXR_NAMESPACE_OPEN_SCOPE

/// A Hydra material: its network as a MaterialX document (hdMtlx), handed to
/// the engine, which compiles it (material::MaterialCompiler) at its next
/// commit. UsdPreviewSurface networks arrive under their USD node names and
/// are renamed to the MaterialX nodedefs MaterialX ships for them.
class HdLrtMaterial final : public HdMaterial {
public:
    explicit HdLrtMaterial(SdfPath const& id) : HdMaterial(id) {}

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;
    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
    void Finalize(HdRenderParam* renderParam) override;
};

PXR_NAMESPACE_CLOSE_SCOPE
