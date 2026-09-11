// Copyright (c) 2026 lucabRTrender contributors.
#include "RenderPass.h"

#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderPassState.h>

#include "RenderBuffer.h"
#include "RenderDelegate.h"
#include "lrt/core/Log.h"
#include "lrt/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdLrtRenderPass::_Execute(HdRenderPassStateSharedPtr const& state, TfTokenVector const&) {
    if (_engine == nullptr) {
        return;
    }
    GfRect2i window;
    const CameraUtilFraming& framing = state->GetFraming();
    if (framing.IsValid()) {
        window = framing.dataWindow;
    } else {
        const GfVec4f vp = state->GetViewport();
        window = GfRect2i(GfVec2i(0), int(vp[2]), int(vp[3]));
    }
    const unsigned width = static_cast<unsigned>(std::max(window.GetWidth(), 1));
    const unsigned height = static_cast<unsigned>(std::max(window.GetHeight(), 1));

    if (auto uploaded = _engine->commit(); !uploaded) {
        lrt::log::error("hdLrt: {}", uploaded.error().toString());
        return;
    }
    const GfMatrix4d proj = state->GetProjectionMatrix();
    const lrt::render::Projection projection =
        lrt::usd::projectionFromHydra(state->GetWorldToViewMatrix(), proj, width, height);

    lrt::render::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    HdLrtRenderBuffer* colourBuffer = nullptr;
    HdLrtRenderBuffer* depthBuffer = nullptr;
    for (HdRenderPassAovBinding const& binding : state->GetAovBindings()) {
        auto* buffer = dynamic_cast<HdLrtRenderBuffer*>(binding.renderBuffer);
        if (buffer == nullptr) {
            continue;
        }
        if (binding.aovName == HdAovTokens->color) {
            colourBuffer = buffer;
            if (binding.clearValue.IsHolding<GfVec4f>()) {
                const GfVec4f c = binding.clearValue.UncheckedGet<GfVec4f>();
                settings.background = {c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]};
            }
        } else if (binding.aovName == HdAovTokens->depth) {
            depthBuffer = buffer;
        }
    }

    const auto technique = _delegate != nullptr ? _delegate->GetTechnique() : lrt::usd::Technique::Raster;
    const bool settle = _delegate != nullptr && _delegate->GetSettleStreams();
    if (auto drawn = _engine->render(projection, settings, _targets, technique, settle); !drawn) {
        lrt::log::error("hdLrt: {}", drawn.error().toString());
        return;
    }
    const auto fill = [&](HdLrtRenderBuffer* buffer, bool depth) {
        if (buffer == nullptr || buffer->GetWidth() != width || buffer->GetHeight() != height) {
            return;
        }
        const HdFormat component = HdGetComponentFormat(buffer->GetFormat());
        lrt::usd::AovLayout layout;
        layout.channels = static_cast<uint32_t>(HdGetComponentCount(buffer->GetFormat()));
        layout.componentBytes = static_cast<uint32_t>(HdDataSizeOfFormat(component));
        layout.componentKind = component == HdFormatUNorm8 ? 0u : component == HdFormatFloat16 ? 1u : 2u;
        if (component != HdFormatUNorm8 && component != HdFormatFloat16 && component != HdFormatFloat32) {
            lrt::log::warn("hdLrt: render buffer format {} is not filled", static_cast<int>(buffer->GetFormat()));
            return;
        }
        if (auto written = _engine->writeAov(_targets, depth, layout, proj.data(), buffer->Bytes()); !written) {
            lrt::log::error("hdLrt: {}", written.error().toString());
        }
    };
    fill(colourBuffer, false);
    fill(depthBuffer, true);
}

PXR_NAMESPACE_CLOSE_SCOPE
