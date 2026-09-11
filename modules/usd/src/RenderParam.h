// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <pxr/imaging/hd/renderDelegate.h>

#include "Engine.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdLrtRenderParam final : public HdRenderParam {
public:
    explicit HdLrtRenderParam(lrt::usd::Engine* engine) : _engine(engine) {}
    [[nodiscard]] lrt::usd::Engine* GetEngine() const { return _engine; }

private:
    lrt::usd::Engine* _engine = nullptr;
};

PXR_NAMESPACE_CLOSE_SCOPE
