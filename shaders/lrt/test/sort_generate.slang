// Copyright (c) 2026 lucabRTrender contributors.
//
// Test input made on the GPU: keys by pattern, values = original index.
// Patterns: 0 hashed random, 1 all equal, 2 already sorted, 3 reverse sorted,
// 4 few distinct values (lots of ties -- what stability is for).
module sort_generate;

struct GenerateParams {
    uint count;
    uint mode;
    uint seed;
    uint keyMaskLo;
    uint keyMaskHi;
    uint pad0;
    uint pad1;
    uint pad2;
};

RWStructuredBuffer<uint>       keysLo;
RWStructuredBuffer<uint>       keysHi;
RWStructuredBuffer<uint>       values;
ConstantBuffer<GenerateParams> params;

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sortGenerate(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    uint lo = 0;
    uint hi = 0;
    switch (params.mode) {
    case 0: lo = pcg(i ^ params.seed); hi = pcg(lo + params.seed); break;
    case 1: lo = 0x5A5A5A5Au; hi = 0x0000A5A5u; break;
    case 2: lo = i; hi = 0; break;
    case 3: lo = params.count - 1 - i; hi = 7; break;
    default: lo = pcg(i ^ params.seed) % 13u; hi = pcg(i + params.seed) % 3u; break;
    }
    keysLo[i] = lo & params.keyMaskLo;
    keysHi[i] = hi & params.keyMaskHi;
    values[i] = i;
}
