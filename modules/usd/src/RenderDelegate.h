// Copyright (c) 2026 lucabRTrender contributors.
//
// lucabRTrender as a Hydra 2.0 render delegate: USD is the scene, the engine
// is the renderer. ParticleField3DGaussianSplat and Points are drawn; cameras
// come through HdCamera; AOVs are colour and depth.
#pragma once

#include <memory>

#include <pxr/imaging/hd/renderDelegate.h>

#include "Engine.h"
#include "RenderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdLrtRenderDelegate final : public HdRenderDelegate {
public:
    HdLrtRenderDelegate();
    explicit HdLrtRenderDelegate(HdRenderSettingsMap const& settings);
    ~HdLrtRenderDelegate() override;

    TfTokenVector const& GetSupportedRprimTypes() const override;
    TfTokenVector const& GetSupportedSprimTypes() const override;
    TfTokenVector const& GetSupportedBprimTypes() const override;
    HdRenderParam* GetRenderParam() const override { return _param.get(); }
    HdResourceRegistrySharedPtr GetResourceRegistry() const override { return _registry; }

    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index,
                                           HdRprimCollection const& collection) override;
    HdInstancer* CreateInstancer(HdSceneDelegate*, SdfPath const&) override { return nullptr; }
    void DestroyInstancer(HdInstancer*) override {}

    HdRprim* CreateRprim(TfToken const& typeId, SdfPath const& rprimId) override;
    void DestroyRprim(HdRprim* rprim) override;
    HdSprim* CreateSprim(TfToken const& typeId, SdfPath const& sprimId) override;
    HdSprim* CreateFallbackSprim(TfToken const& typeId) override;
    void DestroySprim(HdSprim* sprim) override;
    HdBprim* CreateBprim(TfToken const& typeId, SdfPath const& bprimId) override;
    HdBprim* CreateFallbackBprim(TfToken const& typeId) override;
    void DestroyBprim(HdBprim* bprim) override;
    void CommitResources(HdChangeTracker*) override {}

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const& name) const override;

    /// `lrt:technique`: "raster" (default) or "rt".
    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;
    [[nodiscard]] lrt::usd::Technique GetTechnique() const;

    /// True when a device opened; a delegate without one draws nothing.
    [[nodiscard]] bool HasEngine() const { return _engine != nullptr; }

private:
    void _Setup();

    std::unique_ptr<lrt::usd::Engine>  _engine;
    std::unique_ptr<HdLrtRenderParam>  _param;
    HdResourceRegistrySharedPtr        _registry;
};

PXR_NAMESPACE_CLOSE_SCOPE
