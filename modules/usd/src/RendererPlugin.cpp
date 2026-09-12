// Copyright (c) 2026 lucabRTrender contributors.
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/sceneIndexPluginRegistry.h>
#include <pxr/imaging/hdsi/lightLinkingSceneIndex.h>

#include "RenderDelegate.h"

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
}

/// Light linking is USD's to resolve: this scene index turns a light's
/// collections into categories on the geometry they include, which is what
/// GetCategories then hands the delegate and what a light's lightLink and
/// shadowLink name. Without it every collection is invisible here.
TF_REGISTRY_FUNCTION(HdSceneIndexPlugin) {
    HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
        "lucabRTrender",
        [](const std::string&, const HdSceneIndexBaseRefPtr& inputScene,
           const HdContainerDataSourceHandle& inputArgs) -> HdSceneIndexBaseRefPtr {
            return HdsiLightLinkingSceneIndex::New(inputScene, inputArgs);
        },
        nullptr, 0, HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
}

PXR_NAMESPACE_CLOSE_SCOPE
