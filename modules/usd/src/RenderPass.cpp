// Copyright (c) 2026 lucabRTrender contributors.
#include "RenderPass.h"

#include <algorithm>
#include <array>

#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderPassState.h>

#include "RenderBuffer.h"
#include "RenderDelegate.h"
#include "lrt/core/Log.h"
#include "lrt/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdLrtRenderPass::_Execute(HdRenderPassStateSharedPtr const& state, TfTokenVector const& renderTags) {
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
    // Every AOV binding: what each reads, and what the frame must compute.
    struct Output {
        HdLrtRenderBuffer*      buffer;
        lrt::usd::AovSource     source;
    };
    std::vector<Output> outputs;
    lrt::usd::AovRequest request;
    for (HdRenderPassAovBinding const& binding : state->GetAovBindings()) {
        auto* buffer = dynamic_cast<HdLrtRenderBuffer*>(binding.renderBuffer);
        if (buffer == nullptr) {
            continue;
        }
        const std::string& name = binding.aovName.GetString();
        lrt::usd::AovSource source;
        if (binding.aovName == HdAovTokens->color) {
            source.kind = lrt::usd::AovKind::Colour;
            if (binding.clearValue.IsHolding<GfVec4f>()) {
                const GfVec4f c = binding.clearValue.UncheckedGet<GfVec4f>();
                settings.background = {c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]};
            }
        } else if (binding.aovName == HdAovTokens->depth) {
            source.kind = lrt::usd::AovKind::Depth;
        } else if (binding.aovName == HdAovTokens->primId) {
            source.kind = lrt::usd::AovKind::PrimId;
            request.ids = true;
        } else if (binding.aovName == HdAovTokens->instanceId) {
            source.kind = lrt::usd::AovKind::InstanceId;
            request.ids = true;
        } else if (binding.aovName == HdAovTokens->elementId) {
            source.kind = lrt::usd::AovKind::ElementId;
            request.ids = true;
        } else if (binding.aovName == TfToken("albedo")) {
            source.kind = lrt::usd::AovKind::Albedo;
        } else if (binding.aovName == TfToken("shadingNormal")) {
            source.kind = lrt::usd::AovKind::ShadingNormal;
        } else if (binding.aovName == HdAovTokens->Neye) {
            source.kind = lrt::usd::AovKind::EyeNormal;
            request.normals = true;
        } else if (binding.aovName == HdAovTokens->normal) {
            source.kind = lrt::usd::AovKind::WorldNormal;
            request.normals = true;
        } else if (name.rfind("primvars:", 0) == 0) {
            source.kind = lrt::usd::AovKind::Primvar;
            source.primvar = static_cast<uint32_t>(request.primvars.size());
            request.primvars.push_back(name.substr(9));
        } else {
            continue;
        }
        outputs.push_back({buffer, source});
    }

    const auto technique = _delegate != nullptr ? _delegate->GetTechnique() : lrt::usd::Technique::Raster;
    const bool settle = _delegate != nullptr && _delegate->GetSettleStreams();
    const auto visibility =
        _delegate != nullptr ? _delegate->GetMeshVisibility() : lrt::usd::MeshVisibility::Automatic;
    if (_delegate != nullptr) {
        _engine->setLightSamples(_delegate->GetLightSamples());
        _engine->setChooseLights(_delegate->GetChooseLights());
        _engine->setPathSamples(_delegate->GetPathSamples());
        _engine->setPathBounces(_delegate->GetPathBounces());
        _engine->setPathTotal(_delegate->GetPathTotal());
        _engine->setDenoise(_delegate->GetDenoise());
    }
    if (auto drawn =
            _engine->render(projection, settings, *_targets, technique, settle, &renderTags, request, visibility);
        !drawn) {
        lrt::log::error("hdLrt: {}", drawn.error().toString());
        return;
    }
    for (const Output& output : outputs) {
        HdLrtRenderBuffer* buffer = output.buffer;
        if (buffer->GetWidth() != width || buffer->GetHeight() != height) {
            continue;
        }
        const HdFormat component = HdGetComponentFormat(buffer->GetFormat());
        lrt::usd::AovLayout layout;
        layout.channels = static_cast<uint32_t>(HdGetComponentCount(buffer->GetFormat()));
        layout.componentBytes = static_cast<uint32_t>(HdDataSizeOfFormat(component));
        switch (component) {
        case HdFormatUNorm8: layout.componentKind = 0; break;
        case HdFormatFloat16: layout.componentKind = 1; break;
        case HdFormatFloat32: layout.componentKind = 2; break;
        case HdFormatInt32: layout.componentKind = 3; break;
        default:
            lrt::log::warn("hdLrt: render buffer format {} is not filled", static_cast<int>(buffer->GetFormat()));
            continue;
        }
        std::array<double, 16> hostProjection{};
        std::copy(proj.data(), proj.data() + 16, hostProjection.begin());
        buffer->SetPendingFill([engine = _engine, targets = _targets, source = output.source, layout,
                                hostProjection](std::span<uint8_t> into) {
            if (auto written = engine->writeAov(*targets, source, layout, hostProjection.data(), into); !written) {
                lrt::log::error("hdLrt: {}", written.error().toString());
            }
        });
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
