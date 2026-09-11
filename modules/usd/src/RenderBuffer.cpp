// Copyright (c) 2026 lucabRTrender contributors.
#include "RenderBuffer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <pxr/base/gf/half.h>
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

void HdLrtRenderBuffer::WriteColour(const float* rgba, unsigned int width, unsigned int height) {
    if (width != _width || height != _height || _data.empty()) {
        return;
    }
    const size_t channels = HdGetComponentCount(_format);
    const HdFormat component = HdGetComponentFormat(_format);
    const size_t pixelBytes = HdDataSizeOfFormat(_format);
    for (unsigned y = 0; y < height; ++y) {
        const size_t from = size_t{height - 1 - y} * width;
        for (unsigned x = 0; x < width; ++x) {
            const float* src = rgba + (from + x) * 4;
            uint8_t* dst = &_data[(size_t{y} * width + x) * pixelBytes];
            for (size_t c = 0; c < channels && c < 4; ++c) {
                const float v = src[c];
                switch (component) {
                case HdFormatFloat32: std::memcpy(dst + c * 4, &v, 4); break;
                case HdFormatFloat16: {
                    const uint16_t h = GfHalf(v).bits();
                    std::memcpy(dst + c * 2, &h, 2);
                    break;
                }
                case HdFormatUNorm8:
                    dst[c] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0F, 1.0F) * 255.0F));
                    break;
                default: break;
                }
            }
        }
    }
}

void HdLrtRenderBuffer::WriteDepth(const float* depth01, unsigned int width, unsigned int height) {
    if (width != _width || height != _height || _data.empty() ||
        HdGetComponentFormat(_format) != HdFormatFloat32) {
        return;
    }
    for (unsigned y = 0; y < height; ++y) {
        const size_t from = size_t{height - 1 - y} * width;
        std::memcpy(&_data[size_t{y} * width * 4], depth01 + from, size_t{width} * 4);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
