// Copyright (c) 2026 lucabRTrender contributors.
//
// The engine's Hydra outputs against Storm's (OpenUSD's own GPU renderer), as
// Hydra render buffers (bottom row first): primId, depth ([0, 1]) and Neye.
//
//   prim ids   each render index numbers its prims its own way, so what must
//              agree is the onlyA: where either image saw one prim
//              across a pixel's 3x3 neighbourhood, the other did too
//   coverage   depth below 1 (Storm's primId clears to 0, not -1)
//   depth      where both saw one prim across the pixel's 3x3 neighbourhood
//   Neye       Storm writes vec4(Neye, 1) into UNorm8Vec4 -- negative
//              components clamp to 0 -- so ours is compared in that space:
//              saturated, eight bits, within `byteTolerance` per channel
module oracle_compare;

struct OracleParams {
    uint  width;
    uint  height;
    float depthEpsilon;
    uint  byteTolerance;
};

StructuredBuffer<int>      primA;
StructuredBuffer<int>      primB;
StructuredBuffer<float>    depthA;
StructuredBuffer<float>    depthB;
StructuredBuffer<float>    eyeA;    // 3 per pixel
StructuredBuffer<uint>     eyeB;    // RGBA8 per pixel
RWStructuredBuffer<uint>   counts;  // [interior, uniform in A only, depth, normal mismatches, covered A, covered B,
                                    //  coverage mismatches, uniform in either, uniform in B only]
RWStructuredBuffer<float>  worst;   // [depth difference among the interior]
ConstantBuffer<OracleParams> params;

bool covered(StructuredBuffer<float> depths, uint i) {
    return depths[i] < 1.0;
}

bool uniformPrim(StructuredBuffer<int> prims, StructuredBuffer<float> depths, int x, int y) {
    const uint centre = uint(y) * params.width + uint(x);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const uint i = uint(y + dy) * params.width + uint(x + dx);
            if (!covered(depths, i) || prims[i] != prims[centre]) {
                return false;
            }
        }
    }
    return true;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void oracleCompare(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint interior = 0, onlyA = 0, depths = 0, normals = 0, coveredA = 0, coveredB = 0, coverage = 0, eitherUniform = 0, onlyB = 0;
    float depthWorst = 0.0;
    for (int y = 0; y < int(params.height); ++y) {
        for (int x = 0; x < int(params.width); ++x) {
            const uint i = uint(y) * params.width + uint(x);
            const bool a = covered(depthA, i);
            const bool b = covered(depthB, i);
            coveredA += a ? 1u : 0u;
            coveredB += b ? 1u : 0u;
            coverage += a != b ? 1u : 0u;
            if (x == 0 || y == 0 || x + 1 == int(params.width) || y + 1 == int(params.height)) {
                continue;
            }
            const bool uniformA = uniformPrim(primA, depthA, x, y);
            const bool uniformB = uniformPrim(primB, depthB, x, y);
            if (uniformA || uniformB) {
                ++eitherUniform;
                onlyA += uniformA && !uniformB ? 1u : 0u;
                onlyB += uniformB && !uniformA ? 1u : 0u;
            }
            if (!uniformA || !uniformB) {
                continue;
            }
            ++interior;
            const float d = abs(depthA[i] - depthB[i]);
            depthWorst = max(depthWorst, d);
            depths += d > params.depthEpsilon ? 1u : 0u;
            const float3 na = float3(eyeA[i * 3], eyeA[i * 3 + 1], eyeA[i * 3 + 2]);
            const uint nb = eyeB[i];
            bool differs = false;
            for (uint c = 0; c < 3; ++c) {
                const int ours = int(round(saturate(na[c]) * 255.0));
                const int theirs = int((nb >> (8 * c)) & 0xFFu);
                differs = differs || uint(abs(ours - theirs)) > params.byteTolerance;
            }
            normals += differs ? 1u : 0u;
        }
    }
    counts[0] = interior;
    counts[1] = onlyA;
    counts[2] = depths;
    counts[3] = normals;
    counts[4] = coveredA;
    counts[5] = coveredB;
    counts[6] = coverage;
    counts[7] = eitherUniform;
    counts[8] = onlyB;
    worst[0] = depthWorst;
}
