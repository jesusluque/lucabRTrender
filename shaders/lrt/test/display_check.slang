// Copyright (c) 2026 lucabRTrender contributors.
//
// The display transform against its formulas, written out again here:
// inputs generated on the device, and every output pixel recomputed -- the
// P3 matrix from the primaries' chromaticities rather than constants, the AgX
// sigmoid as powers rather than Horner, exposure as exp(stops * ln 2).
module display_check;

struct CheckParams {
    uint  width;
    uint  height;
    uint  view;
    uint  display;
    float exposure;
    float tolerance;
    float backgroundR; float backgroundG; float backgroundB;
    float pad0; float pad1; float pad2;
};

RWStructuredBuffer<float4>  source;    // generate: written; check: read
Texture2D<float4>           shown;
RWStructuredBuffer<uint>    counts;    // [over tolerance, pixels]
RWStructuredBuffer<float>   worst;
ConstantBuffer<CheckParams> params;

/// Premultiplied linear colours from 2^-10 to 2^6 across x, hues and
/// coverage down y (bottom row first).
[shader("compute")]
[numthreads(16, 16, 1)]
void displayGenerate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const float stops = -10.0 + 16.0 * float(tid.x) / float(params.width - 1);
    const float v = float(tid.y) / float(params.height - 1);
    const float3 hue = float3(0.5 + 0.5 * cos(6.2831853 * v), 0.5 + 0.5 * cos(6.2831853 * (v - 0.3333)),
                              0.5 + 0.5 * cos(6.2831853 * (v - 0.6667)));
    const float alpha = (tid.y % 4) == 3 ? 0.35 : 1.0;
    source[tid.y * params.width + tid.x] = float4(hue * pow(2.0, stops) * alpha, alpha);
}

float3x3 inverse3(float3x3 m) {
    const float3 a = m[0];
    const float3 b = m[1];
    const float3 c = m[2];
    const float3 r0 = cross(b, c);
    const float3 r1 = cross(c, a);
    const float3 r2 = cross(a, b);
    const float det = dot(a, r0);
    // The rows of the inverse transpose, transposed back.
    return transpose(float3x3(r0, r1, r2)) / det;
}

/// RGB to XYZ for primaries (x, y) and a white point.
float3x3 rgbToXyz(float2 r, float2 g, float2 b, float2 w) {
    const float3 R = float3(r.x / r.y, 1.0, (1.0 - r.x - r.y) / r.y);
    const float3 G = float3(g.x / g.y, 1.0, (1.0 - g.x - g.y) / g.y);
    const float3 B = float3(b.x / b.y, 1.0, (1.0 - b.x - b.y) / b.y);
    const float3 W = float3(w.x / w.y, 1.0, (1.0 - w.x - w.y) / w.y);
    const float3x3 columns = transpose(float3x3(R, G, B));
    const float3 S = mul(inverse3(columns), W);
    return transpose(float3x3(R * S.x, G * S.y, B * S.z));
}

float oetfSrgb(float x) {
    if (x > 0.0031308) {
        return 1.055 * exp(log(x) / 2.4) - 0.055;
    }
    return x * 12.92;
}

float3 expected(float4 premultiplied) {
    const float3 background = float3(params.backgroundR, params.backgroundG, params.backgroundB);
    float3 c = premultiplied.rgb + (1.0 - saturate(premultiplied.a)) * background;
    c = c * exp(params.exposure * 0.69314718);
    if (params.view == 1) {
        // Inset, as column vectors: each GLSL column of Wrensch's matrix is a row here, transposed.
        const float3x3 inset = transpose(float3x3(
            0.842479062253094, 0.0423282422610123, 0.0423756549057051,
            0.0784335999999992, 0.878468636469772, 0.0784336,
            0.0792237451477643, 0.0791661274605434, 0.879142973793104));
        const float3x3 outset = transpose(float3x3(
            1.19687900512017, -0.0528968517574562, -0.0529716355144438,
            -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
            -0.0990297440797205, -0.0989611768448433, 1.15107367264116));
        float3 e = mul(inset, max(c, 0.0));
        e = (clamp(log2(max(e, 1e-10)), -12.47393, 4.026069) + 12.47393) / (4.026069 + 12.47393);
        float3 y = 15.5 * pow(e, 6.0) - 40.14 * pow(e, 5.0) + 31.96 * pow(e, 4.0) - 6.868 * pow(e, 3.0) +
                   0.4298 * e * e + 0.1191 * e - 0.00232;
        y = mul(outset, y);
        c = pow(max(y, 0.0), 2.2);
    }
    if (params.display == 2) {
        const float2 white = float2(0.3127, 0.3290);
        const float3x3 from709 = rgbToXyz(float2(0.64, 0.33), float2(0.30, 0.60), float2(0.15, 0.06), white);
        const float3x3 fromP3 = rgbToXyz(float2(0.680, 0.320), float2(0.265, 0.690), float2(0.150, 0.060), white);
        c = mul(inverse3(fromP3), mul(from709, c));
    }
    c = saturate(c);
    if (params.display == 1) {
        return exp(log(max(c, 1e-30)) / 2.4) * float3(c > 0.0);
    }
    return float3(oetfSrgb(c.r), oetfSrgb(c.g), oetfSrgb(c.b));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void displayCheck(uint3 tid: SV_DispatchThreadID) {
    uint over = 0;
    float worstError = 0.0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            // The texture is top row first; the source bottom row first.
            const float3 want = expected(source[(params.height - 1 - y) * params.width + x]);
            const float3 got = shown.Load(int3(int(x), int(y), 0)).rgb;
            const float error = max(max(abs(got.r - want.r), abs(got.g - want.g)), abs(got.b - want.b));
            worstError = max(worstError, error);
            over += error > params.tolerance ? 1u : 0u;
        }
    }
    counts[0] = over;
    counts[1] = params.width * params.height;
    worst[0] = worstError;
}

/// Two displays of the same source against each other: pixels whose
/// largest channel differs by more than the tolerance, and the worst.
Texture2D<float4> other;

[shader("compute")]
[numthreads(1, 1, 1)]
void displayAgree(uint3 tid: SV_DispatchThreadID) {
    uint over = 0;
    float worstError = 0.0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            const float3 a = shown.Load(int3(int(x), int(y), 0)).rgb;
            const float3 b = other.Load(int3(int(x), int(y), 0)).rgb;
            const float error = max(max(abs(a.r - b.r), abs(a.g - b.g)), abs(a.b - b.b));
            worstError = max(worstError, error);
            over += error > params.tolerance ? 1u : 0u;
        }
    }
    counts[0] = over;
    counts[1] = params.width * params.height;
    worst[0] = worstError;
}
