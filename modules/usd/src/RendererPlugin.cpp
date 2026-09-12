// Copyright (c) 2026 lucabRTrender contributors.
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>

#include "RenderDelegate.h"
#include "lrt/core/Log.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdLrtRendererPlugin final : public HdRendererPlugin {
public:
    HdRenderDelegate* CreateRenderDelegate() override { return new HdLrtRenderDelegate(); }
    HdRenderDelegate* CreateRenderDelegate(HdRenderSettingsMap const& settings) override {
        return new HdLrtRenderDelegate(settings);
    }
    void DeleteRenderDelegate(HdRenderDelegate* delegate) override { delete delegate; }
#if HD_API_VERSION >= 103
    bool IsSupported(HdRendererCreateArgsSchema const&, std::string*) const override { return true; }
#else
    bool IsSupported(HdRendererCreateArgs const&, std::string*) const override { return true; }
#endif
};

TF_REGISTRY_FUNCTION(TfType) {
    HdRendererPluginRegistry::Define<HdLrtRendererPlugin>();
    // Light linking is USD's to resolve: this scene index turns a light's
    // lightLink and shadowLink collections into categories on the geometry
    // they include, which is what GetCategories hands the delegate. Hosts on
    // the scene index path get it from here; the legacy scene delegate
    // resolves the same collections itself.
    HdLrtRegisterSceneIndices();
}

PXR_NAMESPACE_CLOSE_SCOPE
