// Copyright (c) 2026 lucabRTrender contributors.
//
// lucabRTrender as a Hydra 2.0 render delegate: USD is the scene, the engine
// is the renderer. ParticleField3DGaussianSplat and Points are drawn; cameras
// come through HdCamera; AOVs are colour and depth.
#pragma once

#include <memory>

#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>

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
    TfTokenVector        GetRenderSettingsNamespaces() const override;
    HdRenderParam* GetRenderParam() const override { return _param.get(); }

    /// Kept, so that a change the scene does not carry can be sent through it.
    void SetTerminalSceneIndex(const HdSceneIndexBaseRefPtr& terminalSceneIndex) override;
    /// Every prim's transform and primvars sampled again -- the shutter
    /// changed -- through the HdLrtResampleSceneIndex in the host's chain.
    /// False when none is there (a chain not built for this renderer).
    bool ResampleAllPrims() const;
    HdResourceRegistrySharedPtr GetResourceRegistry() const override { return _registry; }

    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index,
                                           HdRprimCollection const& collection) override;
    HdInstancer* CreateInstancer(HdSceneDelegate* delegate, SdfPath const& id) override;
    void DestroyInstancer(HdInstancer* instancer) override;

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
    TfTokenVector GetMaterialRenderContexts() const override;
    /// The binding purpose a mesh's material is resolved with, falling back
    /// to the all-purpose binding: a renderer's "full", not Storm's "preview".
    /// (StageRenderer resolves render settings' purposes before this is asked.)
    TfToken GetMaterialBindingPurpose() const override { return HdTokens->full; }
    [[nodiscard]] lrt::usd::Technique GetTechnique() const;
    /// `lrt:settleStreams`: false (default) for a viewport, which lets streamed
    /// assets fill in over frames; true for an image that must be complete.
    [[nodiscard]] bool GetSettleStreams() const;

    /// Samples per light per pixel ("lrt:lightSamples"), at least one.
    [[nodiscard]] uint32_t GetLightSamples() const;
    /// Whether to choose one light a sample ("lrt:chooseLights").
    [[nodiscard]] bool GetChooseLights() const;
    /// Paths a pixel a path traced frame gathers ("lrt:pathSamples"), at
    /// least one, and bounces after the first hit ("lrt:pathBounces").
    [[nodiscard]] uint32_t GetPathSamples() const;
    [[nodiscard]] uint32_t GetPathBounces() const;
    [[nodiscard]] uint32_t GetMotionBuckets() const;
    /// A render product's (or its settings prim's) disableMotionBlur and
    /// disableDepthOfField: "lrt:disableMotionBlur" draws one shutter slice,
    /// "lrt:disableDepthOfField" a pinhole through the camera's lens.
    [[nodiscard]] bool GetDisableMotionBlur() const;
    [[nodiscard]] bool GetDisableDepthOfField() const;
    /// Paths a pixel at which a path traced frame is finished
    /// ("lrt:pathTotal"); one, the default, never accumulates.
    [[nodiscard]] uint32_t GetPathTotal() const;
    /// Denoise a path traced frame once it is gathered ("lrt:denoise").
    [[nodiscard]] bool GetDenoise() const;
    /// Adaptive sampling ("lrt:pathAdaptive") and its relative error target
    /// ("lrt:pathError").
    [[nodiscard]] bool GetPathAdaptive() const;
    /// "lrt:pathMis": light and material sampling weighed (true, the default).
    [[nodiscard]] bool GetPathMis() const;
    [[nodiscard]] float GetPathError() const;
    /// `lrt:visibility`: "automatic" (default), "raster", "rays" or "bvh".
    [[nodiscard]] lrt::usd::MeshVisibility GetMeshVisibility() const;

    /// True when a device opened; a delegate without one draws nothing.
    [[nodiscard]] bool HasEngine() const { return _engine != nullptr; }
    /// The engine's device; only valid when HasEngine().
    [[nodiscard]] lrt::gpu::Device& GetEngineDevice() const { return _engine->device(); }
    [[nodiscard]] lrt::gpu::ShaderLibrary& GetEngineLibrary() const { return _engine->library(); }

private:
    void _Setup();

    std::unique_ptr<lrt::usd::Engine>  _engine;
    std::unique_ptr<HdLrtRenderParam>  _param;
    HdResourceRegistrySharedPtr        _registry;
    HdSceneIndexBasePtr                _terminal;   ///< weak: the render index owns it
};

/// Registers the scene indices this renderer needs (light linking), once.
/// Hosts that build the delegate themselves must call it; the plugin does.
void HdLrtRegisterSceneIndices();

PXR_NAMESPACE_CLOSE_SCOPE
