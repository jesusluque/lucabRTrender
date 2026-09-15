// Copyright (c) 2026 lucabRTrender contributors.
#include "Camera.h"

#include <pxr/imaging/hd/sceneDelegate.h>

#include "RenderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

void HdLrtCamera::Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) {
    // The samples first: HdCamera's Sync cleans the bits.
    const bool transformDirty = (*dirtyBits & DirtyTransform) != 0;
    const auto* param = static_cast<HdLrtRenderParam*>(renderParam);
    if (transformDirty && param != nullptr && sceneDelegate != nullptr) {
        const float open = static_cast<float>(param->GetShutterOpen());
        const float close = static_cast<float>(param->GetShutterClose());
        _moves = false;
        if (close > open) {
            float times[2] = {0.0F, 0.0F};
            GfMatrix4d values[2];
            const size_t n = sceneDelegate->SampleTransform(GetId(), open, close, 2, times, values);
            if (n >= 2 && values[0] != values[n - 1]) {
                _moves = true;
                _start = values[0];
                _end = values[n - 1];
                _timeStart = times[0];
                _timeEnd = times[n - 1];
            }
        }
    }
    HdCamera::Sync(sceneDelegate, renderParam, dirtyBits);
}

PXR_NAMESPACE_CLOSE_SCOPE
