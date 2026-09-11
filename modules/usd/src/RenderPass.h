// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

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
    lrt::render::RenderTargets    _targets;
};

PXR_NAMESPACE_CLOSE_SCOPE
