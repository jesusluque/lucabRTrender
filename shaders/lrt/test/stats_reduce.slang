// Copyright (c) 2026 lucabRTrender contributors.
//
// Folds per-chunk stats rows (8 uints: a count, then sum/xor pairs, then a
// sum) into one row. Sums and counts add, xors xor. One invocation.
module stats_reduce;

struct ReduceParams {
    uint chunkCount;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint>       stats;
RWStructuredBuffer<uint>     result;   // 8
ConstantBuffer<ReduceParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void statsReduce(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint c = 0; c < params.chunkCount; ++c) {
        const uint row = c * 8;
        acc[0] += stats[row + 0];
        acc[1] += stats[row + 1]; acc[2] ^= stats[row + 2];
        acc[3] += stats[row + 3]; acc[4] ^= stats[row + 4];
        acc[5] += stats[row + 5]; acc[6] ^= stats[row + 6];
        acc[7] += stats[row + 7];
    }
    for (uint k = 0; k < 8; ++k) {
        result[k] = acc[k];
    }
}
