// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <pxr/imaging/hd/mesh.h>

PXR_NAMESPACE_OPEN_SCOPE

/// A UsdGeomMesh (and anything a scene index turns into one), as Hydra
/// delivers it: its arrays handed to the engine as they are, built into
/// triangles and normals on the device at commit.
class HdLrtMesh final : public HdMesh {
public:
    explicit HdLrtMesh(SdfPath const& id) : HdMesh(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
              TfToken const& reprToken) override;
    void Finalize(HdRenderParam* renderParam) override;
    TfTokenVector const& GetBuiltinPrimvarNames() const override;

protected:
    void _InitRepr(TfToken const&, HdDirtyBits*) override {}
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }
};

PXR_NAMESPACE_CLOSE_SCOPE
