// Copyright (c) 2026 lucabRTrender contributors.
//
// One mip level from the level above it, each texel the area-weighted mean of
// the texels it covers. A level is half the one above rounded down, so an odd
// edge of 2n + 1 texels folds into n: each target texel takes three source
// texels with weights (n - x, n, x + 1) / (2n + 1). Every source texel then
// gives exactly n / (2n + 1) of itself to the level below -- the chain keeps
// level 0's mean however the sizes fall (Nvidia, "Non-Power-of-Two Mipmapping").
module mips;

struct MipParams {
    uint sourceWidth;
    uint sourceHeight;
    uint width;
    uint height;
    uint srgb;   // 1: texels hold sRGB-encoded colour; averaged as light, stored encoded
    uint levels; // 255 for an 8-bit target: stored as its nearest code; 0 for a float target
    uint pad1; uint pad2;
};

float decodeSrgb(float x) {
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

float encodeSrgb(float x) {
    return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

Texture2D<float4>         source;
RWTexture2D<float4>       target;
ConstantBuffer<MipParams> params;

struct Taps {
    uint  first;
    uint  count;
    float3 weights;
};

/// The source texels and weights target texel `x` covers along an edge.
Taps tapsAlong(uint x, uint sourceSize, uint size) {
    Taps taps;
    if (sourceSize == size) {
        taps.first = x;
        taps.count = 1;
        taps.weights = float3(1.0, 0.0, 0.0);
    } else if (sourceSize == 2 * size) {
        taps.first = 2 * x;
        taps.count = 2;
        taps.weights = float3(0.5, 0.5, 0.0);
    } else {
        const float n = float(size);
        const float inverse = 1.0 / (2.0 * n + 1.0);
        taps.first = 2 * x;
        taps.count = 3;
        taps.weights = float3(n - float(x), n, float(x) + 1.0) * inverse;
    }
    return taps;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void mipDownsample(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const Taps tx = tapsAlong(tid.x, params.sourceWidth, params.width);
    const Taps ty = tapsAlong(tid.y, params.sourceHeight, params.height);
    float4 sum = float4(0.0);
    for (uint j = 0; j < ty.count; ++j) {
        for (uint i = 0; i < tx.count; ++i) {
            float4 texel = source.Load(int3(int(tx.first + i), int(ty.first + j), 0));
            if (params.srgb != 0) {
                texel.rgb = float3(decodeSrgb(texel.r), decodeSrgb(texel.g), decodeSrgb(texel.b));
            }
            sum += tx.weights[i] * ty.weights[j] * texel;
        }
    }
    if (params.srgb != 0) {
        sum.rgb = float3(encodeSrgb(sum.r), encodeSrgb(sum.g), encodeSrgb(sum.b));
    }
    // An 8-bit target is written its nearest code, a quarter code in from the
    // lower edge, so the store gives that code whether the driver rounds or
    // truncates: Vulkan leaves the conversion to the implementation, and on
    // an NVIDIA L4 each level lost about half a code to it -- the chain's
    // mean drifted 0.7 of a code below level 0's, where Metal's stayed within.
    if (params.levels != 0) {
        const float levels = float(params.levels);
        sum = (round(saturate(sum) * levels) + 0.25) / levels;
    }
    target[tid.xy] = sum;
}
