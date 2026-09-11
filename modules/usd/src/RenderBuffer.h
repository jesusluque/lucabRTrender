// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <atomic>
#include <span>
#include <vector>

#include <pxr/imaging/hd/renderBuffer.h>

PXR_NAMESPACE_OPEN_SCOPE

/// An AOV in host memory, for a host that maps it. The render pass fills it
/// after each frame with bytes the device already converted to its format
/// (Engine::writeAov). Always converged: the engine has no progressive mode.
class HdLrtRenderBuffer final : public HdRenderBuffer {
public:
    explicit HdLrtRenderBuffer(SdfPath const& id) : HdRenderBuffer(id) {}

    bool Allocate(GfVec3i const& dimensions, HdFormat format, bool multiSampled) override;
    unsigned int GetWidth() const override { return _width; }
    unsigned int GetHeight() const override { return _height; }
    unsigned int GetDepth() const override { return 1; }
    HdFormat GetFormat() const override { return _format; }
    bool IsMultiSampled() const override { return false; }
    void* Map() override { ++_mappers; return _data.data(); }
    void Unmap() override { --_mappers; }
    bool IsMapped() const override { return _mappers.load() != 0; }
    void Resolve() override {}
    bool IsConverged() const override { return true; }

    /// Where the converted bytes go.
    [[nodiscard]] std::span<uint8_t> Bytes() { return {_data.data(), _data.size()}; }

private:
    void _Deallocate() override { _data.clear(); }

    unsigned int         _width = 0;
    unsigned int         _height = 0;
    HdFormat             _format = HdFormatInvalid;
    std::vector<uint8_t> _data;
    std::atomic<int>     _mappers{0};
};

PXR_NAMESPACE_CLOSE_SCOPE
