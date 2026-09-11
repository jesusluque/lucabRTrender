// Copyright (c) 2026 lucabRTrender contributors.
//
// An engine image into the bytes a Hydra render buffer holds: the buffer's
// format (8-bit, half or float components, 1 to 4 of them), top row first
// where the engine renders bottom row first, and depth as the host
// projection's [0, 1] instead of view z. One thread per output word, so any
// pixel size -- three bytes, twelve -- writes without two threads touching the
// same word.
module aov_convert;
import lrt.common.packing;

struct AovParams {
    uint  width;
    uint  height;
    uint  words;           // output words
    uint  source;          // 0 colour (float4), 1 depth (view z, 0 empty)
    uint  channels;        // components per pixel
    uint  componentBytes;  // 1, 2 or 4
    uint  componentKind;   // 0 unorm8, 1 float16, 2 float32
    uint  pad0;
    // The host projection's terms that take view z to clip depth (row-vector
    // matrices: clip = v * P).
    float p22; float p32; float p23; float p33;
};

StructuredBuffer<float4>   colour;
StructuredBuffer<float>    depth;
RWStructuredBuffer<uint>   words;
ConstantBuffer<AovParams>  params;

float valueOf(uint pixel, uint channel) {
    const uint x = pixel % params.width;
    const uint y = pixel / params.width;
    const uint at = (params.height - 1 - y) * params.width + x;
    if (params.source == 0) {
        return channel < 4 ? colour[at][channel] : 0.0;
    }
    const float z = depth[at];
    if (!(z > 0.0)) {
        return 1.0;   // nothing drawn: the far plane
    }
    const float clipZ = -z * params.p22 + params.p32;
    const float clipW = -z * params.p23 + params.p33;
    return saturate((clipZ / clipW) * 0.5 + 0.5);
}

uint bitsOf(float v) {
    if (params.componentKind == 0) {
        return uint(round(saturate(v) * 255.0));
    }
    if (params.componentKind == 1) {
        return f32tof16(halfSafe(v));
    }
    return bit_cast<uint>(v);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void aovConvert(uint3 tid: SV_DispatchThreadID) {
    const uint w = tid.x;
    if (w >= params.words) {
        return;
    }
    const uint pixelBytes = params.channels * params.componentBytes;
    const uint total = params.width * params.height * pixelBytes;
    uint word = 0;
    for (uint b = 0; b < 4; ++b) {
        const uint at = w * 4 + b;
        if (at >= total) {
            break;
        }
        const uint pixel = at / pixelBytes;
        const uint inPixel = at % pixelBytes;
        const uint bits = bitsOf(valueOf(pixel, inPixel / params.componentBytes));
        word |= ((bits >> (8 * (inPixel % params.componentBytes))) & 0xFFu) << (8 * b);
    }
    words[w] = word;
}
