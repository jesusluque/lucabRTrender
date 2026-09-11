// Copyright (c) 2026 lucabRTrender contributors.
//
// How far apart two linear images are where 8-bit code values say nothing --
// radiance above 1, the dark end of a path tracer's noise. Per pixel, the
// largest channel's relative difference |a - b| / max(|b|, 1e-3), binned
// logarithmically (eight bins an octave from 2^-16), and the relative squared
// error summed per row (relMSE: (a - b)^2 / (b^2 + 1e-2), channel mean).
module image_compare_hdr;

struct HdrParams {
    uint width;
    uint height;
    uint count;   // idsCompare: entries in the buffers
    uint pad0;
};

StructuredBuffer<float4>   a;
StructuredBuffer<float4>   b;
RWStructuredBuffer<uint>   histogram;   // height * 256
RWStructuredBuffer<float>  rowError;    // height
ConstantBuffer<HdrParams>  params;

[shader("compute")]
[numthreads(64, 1, 1)]
void imageCompareHdr(uint3 tid: SV_DispatchThreadID) {
    const uint y = tid.x;
    if (y >= params.height) {
        return;
    }
    const uint row = y * 256;
    for (uint k = 0; k < 256; ++k) {
        histogram[row + k] = 0;
    }
    float sum = 0.0;
    float carry = 0.0;   // Kahan: a row of small terms added to a large sum
    for (uint x = 0; x < params.width; ++x) {
        const float4 pa = a[y * params.width + x];
        const float4 pb = b[y * params.width + x];
        float worst = 0.0;
        float squared = 0.0;
        for (uint c = 0; c < 4; ++c) {
            const float d = abs(pa[c] - pb[c]);
            worst = max(worst, d / max(abs(pb[c]), 1e-3));
            squared += d * d / (pb[c] * pb[c] + 1e-2);
        }
        const uint bin = worst <= 0.0 ? 0u : uint(clamp(floor((log2(worst) + 16.0) * 8.0) + 1.0, 1.0, 255.0));
        histogram[row + bin] += 1;
        const float term = squared * 0.25;
        const float t = sum + term;
        carry += abs(sum) >= abs(term) ? (sum - t) + term : (term - t) + sum;
        sum = t;
    }
    rowError[y] = sum + carry;
}

// Differing entries of two uint buffers, counted per chunk of `width`.
StructuredBuffer<uint>    idsA;
StructuredBuffer<uint>    idsB;
RWStructuredBuffer<uint>  differing;   // height (chunks)

[shader("compute")]
[numthreads(64, 1, 1)]
void idsCompare(uint3 tid: SV_DispatchThreadID) {
    const uint chunk = tid.x;
    if (chunk >= params.height) {
        return;
    }
    uint n = 0;
    for (uint i = 0; i < params.width; ++i) {
        const uint at = chunk * params.width + i;
        if (at < params.count) {
            n += idsA[at] != idsB[at] ? 1u : 0u;
        }
    }
    differing[chunk] = n;
}

// A float buffer summed (Kahan), for relMSE's rows.
StructuredBuffer<float>   values;
RWStructuredBuffer<float> total;

[shader("compute")]
[numthreads(1, 1, 1)]
void floatSum(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float sum = 0.0;
    float carry = 0.0;
    for (uint i = 0; i < params.height; ++i) {
        const float term = values[i];
        const float t = sum + term;
        carry += abs(sum) >= abs(term) ? (sum - t) + term : (term - t) + sum;
        sum = t;
    }
    total[0] = sum + carry;
}
