// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <atomic>
#include <vector>

#include <pxr/imaging/hd/renderBuffer.h>

PXR_NAMESPACE_OPEN_SCOPE

/// An AOV in host memory. The render pass writes it after each frame; a host
/// maps and reads it. Always converged: the engine has no progressive mode.
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

    /// Writes an engine image (bottom row first) into this buffer's format,
    /// top row first as Hydra reads it. `depth`: view z converted to Hydra's
    /// [0, 1] clip depth with `depthScale`/`depthBias` by the caller.
    void WriteColour(const float* rgba, unsigned int width, unsigned int height);
    void WriteDepth(const float* depth01, unsigned int width, unsigned int height);

private:
    void _Deallocate() override { _data.clear(); }

    unsigned int         _width = 0;
    unsigned int         _height = 0;
    HdFormat             _format = HdFormatInvalid;
    std::vector<uint8_t> _data;
    std::atomic<int>     _mappers{0};
};

PXR_NAMESPACE_CLOSE_SCOPE
