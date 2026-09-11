// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <pxr/imaging/hd/rprim.h>

PXR_NAMESPACE_OPEN_SCOPE

/// A UsdVolParticleField3DGaussianSplat, as Hydra delivers it.
class HdLrtParticleField final : public HdRprim {
public:
    explicit HdLrtParticleField(SdfPath const& id) : HdRprim(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
              TfToken const& reprToken) override;
    void Finalize(HdRenderParam* renderParam) override;
    TfTokenVector const& GetBuiltinPrimvarNames() const override {
        static TfTokenVector builtins;
        return builtins;
    }

protected:
    void _InitRepr(TfToken const&, HdDirtyBits*) override {}
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }
};

PXR_NAMESPACE_CLOSE_SCOPE
