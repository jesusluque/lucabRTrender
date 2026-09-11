// Copyright (c) 2026 lucabRTrender contributors.
//
// Counts for a prefix-sum test, made on the GPU: small hashed numbers with
// runs of zeros, like tiles-touched per splat.
module counts_generate;

struct CountsParams {
    uint count;
    uint seed;
    uint pad0;
    uint pad1;
};

RWStructuredBuffer<uint>     counts;
ConstantBuffer<CountsParams> params;

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void countsGenerate(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint h = pcg(i ^ params.seed);
    counts[i] = (h & 7u) == 0u ? 0u : h % 37u;
}
