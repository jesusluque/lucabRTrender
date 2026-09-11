// Copyright (c) 2026 lucabRTrender contributors.
#include "RenderBuffer.h"

#include <pxr/base/gf/vec3i.h>

PXR_NAMESPACE_OPEN_SCOPE

bool HdLrtRenderBuffer::Allocate(GfVec3i const& dimensions, HdFormat format, bool) {
    _Deallocate();
    if (dimensions[2] != 1) {
        return false;
    }
    _width = static_cast<unsigned>(dimensions[0]);
    _height = static_cast<unsigned>(dimensions[1]);
    _format = format;
    _data.resize(size_t{_width} * _height * HdDataSizeOfFormat(format));
    return true;
}

PXR_NAMESPACE_CLOSE_SCOPE
