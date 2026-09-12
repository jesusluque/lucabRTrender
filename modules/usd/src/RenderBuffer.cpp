// Copyright (c) 2026 lucabRTrender contributors.
#include "RenderBuffer.h"

#include <pxr/base/gf/vec3i.h>

PXR_NAMESPACE_OPEN_SCOPE

bool HdLrtRenderBuffer::IsConverged() const {
    return _engine == nullptr || _engine->pathConverged();
}

bool HdLrtRenderBuffer::Allocate(GfVec3i const& dimensions, HdFormat format, bool) {
    _Deallocate();
    if (dimensions[2] != 1) {
        return false;
    }
    _width = static_cast<unsigned>(dimensions[0]);
    _height = static_cast<unsigned>(dimensions[1]);
    _format = format;
    _data.resize(size_t{_width} * _height * HdDataSizeOfFormat(format));
    SetPendingFill({});
    return true;
}

void* HdLrtRenderBuffer::Map() {
    ++_mappers;
    std::function<void(std::span<uint8_t>)> fill;
    {
        const std::lock_guard<std::mutex> held(_pendingLock);
        fill.swap(_pending);
    }
    if (fill) {
        fill(std::span<uint8_t>(_data.data(), _data.size()));
    }
    return _data.data();
}

void HdLrtRenderBuffer::SetPendingFill(std::function<void(std::span<uint8_t>)> fill) {
    const std::lock_guard<std::mutex> held(_pendingLock);
    _pending = std::move(fill);
}

PXR_NAMESPACE_CLOSE_SCOPE
