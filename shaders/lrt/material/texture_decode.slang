// Copyright (c) 2026 lucabRTrender contributors.
//
// A decoded image's bytes, as a file reader left them (top row first, 1 to 4
// channels of 8-bit, 16-bit or float components), into the first level of an
// RGBA texture with row 0 at the bottom -- where texture coordinate v = 0 is,
// as MaterialX and USD have it.
module texture_decode;
import lrt.common.packing;

struct DecodeParams {
    uint width;
    uint height;
    uint channels;     // 1 grey, 2 grey and alpha, 3 rgb, 4 rgba
    uint component;    // 0 unorm8, 1 unorm16, 2 float16, 3 float32
    uint toLinear;     // 1: the components are sRGB-encoded and the texture holds light
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>       bytes;   // packed little-endian, rows top first
RWTexture2D<float4>          level;
ConstantBuffer<DecodeParams> params;

uint byteAt(uint index) {
    return (bytes[index >> 2] >> ((index & 3u) * 8u)) & 0xFFu;
}

float componentAt(uint index) {
    if (params.component == 0) {
        return float(byteAt(index)) / 255.0;
    }
    if (params.component == 1) {
        return float(byteAt(index * 2) | (byteAt(index * 2 + 1) << 8)) / 65535.0;
    }
    if (params.component == 2) {
        return f16tof32(byteAt(index * 2) | (byteAt(index * 2 + 1) << 8));
    }
    const uint bits = byteAt(index * 4) | (byteAt(index * 4 + 1) << 8) | (byteAt(index * 4 + 2) << 16) |
                      (byteAt(index * 4 + 3) << 24);
    return asfloat(bits);
}

float decodeSrgb(float x) {
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void textureDecode(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint fileRow = params.height - 1 - tid.y;
    const uint first = (fileRow * params.width + tid.x) * params.channels;
    float4 texel = float4(0.0, 0.0, 0.0, 1.0);
    if (params.channels <= 2) {
        texel.rgb = float3(componentAt(first));
        if (params.channels == 2) {
            texel.a = componentAt(first + 1);
        }
    } else {
        texel.r = componentAt(first);
        texel.g = componentAt(first + 1);
        texel.b = componentAt(first + 2);
        if (params.channels == 4) {
            texel.a = componentAt(first + 3);
        }
    }
    if (params.toLinear != 0) {
        texel.rgb = float3(decodeSrgb(texel.r), decodeSrgb(texel.g), decodeSrgb(texel.b));
    }
    level[tid.xy] = texel;
}
