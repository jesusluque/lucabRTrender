// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <memory>

#include <pxr/imaging/hd/renderPass.h>

#include "Engine.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdLrtRenderDelegate;

class HdLrtRenderPass final : public HdRenderPass {
public:
    HdLrtRenderPass(HdRenderIndex* index, HdRprimCollection const& collection,
                    lrt::usd::Engine* engine, const HdLrtRenderDelegate* delegate)
        : HdRenderPass(index, collection), _engine(engine), _delegate(delegate) {}

    bool IsConverged() const override { return true; }

protected:
    void _Execute(HdRenderPassStateSharedPtr const& renderPassState,
                  TfTokenVector const& renderTags) override;

private:
    lrt::usd::Engine*             _engine = nullptr;
    const HdLrtRenderDelegate*    _delegate = nullptr;
    /// Shared with the fills the render buffers hold until they are mapped.
    std::shared_ptr<lrt::render::RenderTargets> _targets = std::make_shared<lrt::render::RenderTargets>();
};

PXR_NAMESPACE_CLOSE_SCOPE
