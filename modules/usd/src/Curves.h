// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <pxr/imaging/hd/rprim.h>

PXR_NAMESPACE_OPEN_SCOPE

/// UsdGeomBasisCurves: its topology, points and widths go to the engine,
/// which lays a tube over every span on the device (geom::CurveBuilder) and
/// draws it as a mesh.
class HdLrtBasisCurves final : public HdRprim {
public:
    explicit HdLrtBasisCurves(SdfPath const& id) : HdRprim(id) {}

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
