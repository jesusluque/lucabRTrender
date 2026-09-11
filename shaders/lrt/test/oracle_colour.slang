// Copyright (c) 2026 lucabRTrender contributors.
//
// Colour against Storm's, per prim: where a pixel's 5x5 neighbourhood is one
// prim in the engine's ids and covered in both images, the engine's float
// colour and Storm's half-float colour must agree to what a half holds.
module oracle_colour;
import lrt.common.packing;

struct ColourParams {
    uint  width;
    uint  height;
    uint  prims;       // ids at or above are ignored
    float tolerance;   // absolute and relative
};

StructuredBuffer<float4>   ours;
StructuredBuffer<uint>     storm;        // RGBA16F: two words a pixel
StructuredBuffer<float>    ourDepth;     // [0, 1], 1 empty
StructuredBuffer<float>    stormDepth;
StructuredBuffer<int>      prim;         // the engine's primId
RWStructuredBuffer<uint>   counts;       // per prim: mismatches, compared, first pixel + 1
RWStructuredBuffer<float4> worst;        // per prim: worst difference, ours, Storm's (green of the worst)
ConstantBuffer<ColourParams> params;

bool interior(int x, int y) {
    const int id = prim[y * int(params.width) + x];
    if (id < 0) {
        return false;
    }
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int i = (y + dy) * int(params.width) + (x + dx);
            if (prim[i] != id || ourDepth[i] >= 1.0 || stormDepth[i] >= 1.0) {
                return false;
            }
        }
    }
    return true;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void oracleColour(uint3 tid: SV_DispatchThreadID) {
    for (uint p = 0; p < params.prims; ++p) {
        counts[p * 3] = 0;
        counts[p * 3 + 1] = 0;
        counts[p * 3 + 2] = 0;
        worst[p] = float4(0.0);
    }
    for (int y = 2; y + 2 < int(params.height); ++y) {
        for (int x = 2; x + 2 < int(params.width); ++x) {
            if (!interior(x, y)) {
                continue;
            }
            const uint i = uint(y) * params.width + uint(x);
            const uint id = uint(prim[i]);
            if (id >= params.prims) {
                continue;
            }
            const uint w0 = storm[i * 2];
            const uint w1 = storm[i * 2 + 1];
            const float3 theirs = float3(lowHalf(w0), highHalf(w0), lowHalf(w1));
            const float3 mine = ours[i].rgb;
            float difference = 0.0;
            bool differs = false;
            for (uint c = 0; c < 3; ++c) {
                const float d = abs(mine[c] - theirs[c]);
                const float allowed = params.tolerance * (1.0 + max(abs(mine[c]), abs(theirs[c])));
                const bool bad = !(d <= allowed);   // NaN is a mismatch
                differs = differs || bad;
                difference = max(difference, bad && isnan(d) ? 3.4e38 : d);
            }
            counts[id * 3] += differs ? 1u : 0u;
            counts[id * 3 + 1] += 1u;
            if (counts[id * 3 + 2] == 0) {
                counts[id * 3 + 2] = i + 1;
            }
            if (difference > worst[id].x) {
                worst[id] = float4(difference, mine.g, theirs.g, 0.0);
            }
        }
    }
}
