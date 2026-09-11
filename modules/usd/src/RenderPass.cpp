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
    if (auto drawn = _engine->render(projection, settings, _targets, technique); !drawn) {
        lrt::log::error("hdLrt: {}", drawn.error().toString());
        return;
    }
    if (colourBuffer != nullptr) {
        if (auto colour = _targets.colour.readAll<float>(_engine->device())) {
            colourBuffer->WriteColour(colour->data(), width, height);
        }
    }
    if (depthBuffer != nullptr) {
        if (auto depth = _targets.depth.readAll<float>(_engine->device())) {
            // View z to Hydra's [0, 1] depth through the host's own projection;
            // nothing drawn is the far plane.
            for (float& z : *depth) {
                if (!(z > 0.0F)) {
                    z = 1.0F;
                    continue;
                }
                const GfVec3d clip = proj.Transform(GfVec3d(0.0, 0.0, -static_cast<double>(z)));
                z = static_cast<float>(std::clamp(clip[2] * 0.5 + 0.5, 0.0, 1.0));
            }
            depthBuffer->WriteDepth(depth->data(), width, height);
        }
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
