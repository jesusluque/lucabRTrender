// Copyright (c) 2026 lucabRTrender contributors.
//
// The renderSettings bprim: hd's HdRenderSettings holds what the scene
// index delivered (products, purposes, namespaced settings); this one
// counts its syncs, so a host can tell what arrived.
#pragma once

#include <atomic>

#include <pxr/imaging/hd/renderSettings.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdLrtRenderSettings final : public HdRenderSettings {
public:
    explicit HdLrtRenderSettings(SdfPath const& id) : HdRenderSettings(id) {}

    [[nodiscard]] uint32_t GetSyncCount() const { return _syncs.load(); }

protected:
    void _Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, const HdDirtyBits* dirtyBits) override {
        HdRenderSettings::_Sync(sceneDelegate, renderParam, dirtyBits);
        _syncs.fetch_add(1);
    }

private:
    std::atomic<uint32_t> _syncs{0};
};

PXR_NAMESPACE_CLOSE_SCOPE
