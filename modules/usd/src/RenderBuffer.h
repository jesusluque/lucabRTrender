// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <span>
#include <vector>

#include <pxr/imaging/hd/renderBuffer.h>

PXR_NAMESPACE_OPEN_SCOPE

/// An AOV in host memory, for a host that maps it. After each frame the render
/// pass leaves it a fill -- the device converts the AOV to the buffer's format
/// and reads it back (Engine::writeAov) -- which runs when the buffer is first
/// mapped: a host that never maps an output (a viewer showing it on the
/// device) never reads it back. Always converged: the engine has no
/// progressive mode. Mapped on the thread that executes the render pass, as
/// hosts that render synchronously do.
class HdLrtRenderBuffer final : public HdRenderBuffer {
public:
    explicit HdLrtRenderBuffer(SdfPath const& id) : HdRenderBuffer(id) {}

    bool Allocate(GfVec3i const& dimensions, HdFormat format, bool multiSampled) override;
    unsigned int GetWidth() const override { return _width; }
    unsigned int GetHeight() const override { return _height; }
    unsigned int GetDepth() const override { return 1; }
    HdFormat GetFormat() const override { return _format; }
    bool IsMultiSampled() const override { return false; }
    void* Map() override;
    void Unmap() override { --_mappers; }
    bool IsMapped() const override { return _mappers.load() != 0; }
    void Resolve() override {}
    bool IsConverged() const override { return true; }

    /// What the next Map runs first: the last frame's conversion into the bytes.
    void SetPendingFill(std::function<void(std::span<uint8_t>)> fill);

private:
    void _Deallocate() override { _data.clear(); }

    unsigned int         _width = 0;
    unsigned int         _height = 0;
    HdFormat             _format = HdFormatInvalid;
    std::vector<uint8_t> _data;
    std::atomic<int>     _mappers{0};
    std::mutex           _pendingLock;
    std::function<void(std::span<uint8_t>)> _pending;
};

PXR_NAMESPACE_CLOSE_SCOPE
