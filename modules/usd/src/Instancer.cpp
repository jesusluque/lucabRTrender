// Copyright (c) 2026 lucabRTrender contributors.
#include "Instancer.h"

#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include "RenderParam.h"
#include "lrt/usd/HydraCamera.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdLrtInstancer::Sync(HdSceneDelegate* delegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
    _UpdateInstancer(delegate, dirtyBits);
    auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine();
    if (engine == nullptr) {
        return;
    }
    SdfPath const& id = GetId();
    if (HdChangeTracker::IsAnyPrimvarDirty(*dirtyBits, id) || HdChangeTracker::IsTransformDirty(*dirtyBits, id) ||
        (*dirtyBits & HdChangeTracker::DirtyInstancer) != 0) {
        lrt::usd::InstancerArrays a;
        a.translations = delegate->Get(id, HdInstancerTokens->instanceTranslations);
        a.rotations = delegate->Get(id, HdInstancerTokens->instanceRotations);
        a.scales = delegate->Get(id, HdInstancerTokens->instanceScales);
        a.transforms = delegate->Get(id, HdInstancerTokens->instanceTransforms);
        a.instancerTransform = lrt::usd::fromUsd(delegate->GetInstancerTransform(id));
        // Under a shutter, the samples about it, as a mesh samples its
        // transform and points: each array and the instancer's own transform,
        // their times kept. Instances placed by velocities arrive here the
        // same way, resolved by hdsi.
        const auto* param = static_cast<HdLrtRenderParam*>(renderParam);
        const float open = static_cast<float>(param->GetShutterOpen());
        const float close = static_cast<float>(param->GetShutterClose());
        if (close > open) {
            lrt::usd::InstancerSample start;
            lrt::usd::InstancerSample end;
            start.translations = end.translations = a.translations;
            start.rotations = end.rotations = a.rotations;
            start.scales = end.scales = a.scales;
            start.transforms = end.transforms = a.transforms;
            start.instancerTransform = end.instancerTransform = a.instancerTransform;
            bool moves = false;
            float timeStart = 0.0F;
            float timeEnd = 0.0F;
            const auto sample = [&](const TfToken& key, VtValue& first, VtValue& last) {
                float times[2] = {0.0F, 0.0F};
                VtValue values[2];
                const size_t n = delegate->SamplePrimvar(id, key, open, close, 2, times, values);
                if (n >= 2 && values[0] != values[n - 1]) {
                    first = values[0];
                    last = values[n - 1];
                    if (!moves) {
                        timeStart = times[0];
                        timeEnd = times[n - 1];
                    }
                    moves = true;
                }
            };
            sample(HdInstancerTokens->instanceTranslations, start.translations, end.translations);
            sample(HdInstancerTokens->instanceRotations, start.rotations, end.rotations);
            sample(HdInstancerTokens->instanceScales, start.scales, end.scales);
            sample(HdInstancerTokens->instanceTransforms, start.transforms, end.transforms);
            {
                float times[2] = {0.0F, 0.0F};
                GfMatrix4d values[2];
                const size_t n = delegate->SampleInstancerTransform(id, open, close, 2, times, values);
                if (n >= 2 && values[0] != values[n - 1]) {
                    start.instancerTransform = lrt::usd::fromUsd(values[0]);
                    end.instancerTransform = lrt::usd::fromUsd(values[n - 1]);
                    if (!moves) {
                        timeStart = times[0];
                        timeEnd = times[n - 1];
                    }
                    moves = true;
                }
            }
            if (moves) {
                a.start = std::move(start);
                a.end = std::move(end);
                a.timeStart = timeStart;
                a.timeEnd = timeEnd;
            }
        }
        engine->setInstancer(id, GetParentId(), std::move(a));
    }
}

void HdLrtInstancer::Finalize(HdRenderParam* renderParam) {
    if (auto* engine = static_cast<HdLrtRenderParam*>(renderParam)->GetEngine()) {
        engine->removeInstancer(GetId());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
