// Copyright (c) 2026 lucabRTrender contributors.
//
// Per-chunk facts about a list of (key, value) pairs, for checking a sort on
// the GPU against the same list before it was sorted -- never against a CPU
// sort.
//
//   stats[c*8 + 0]  pairs out of order (key descends, or ties with values
//                   descending: an unstable sort)
//   stats[c*8 + 1]  sum of keysLo           [2] xor of keysLo
//   stats[c*8 + 3]  sum of keysHi           [4] xor of keysHi
//   stats[c*8 + 5]  sum of values           [6] xor of values
//   stats[c*8 + 7]  sum of value * value (wrapping)
//
// Order is checked across chunk boundaries too (element i against i-1).
// Sums and xors over a permutation do not change; a sort that lost or
// duplicated a pair changes them.
module sort_stats;

struct StatsParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint wide;
};

StructuredBuffer<uint>      keysLo;
StructuredBuffer<uint>      keysHi;
StructuredBuffer<uint>      values;
RWStructuredBuffer<uint>    stats;
ConstantBuffer<StatsParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void sortStats(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint begin = c * params.chunkSize;
    const uint end = min(begin + params.chunkSize, params.count);
    uint disorder = 0;
    uint loSum = 0, loXor = 0, hiSum = 0, hiXor = 0, vSum = 0, vXor = 0, vSq = 0;
    for (uint i = begin; i < end; ++i) {
        const uint lo = keysLo[i];
        const uint hi = params.wide != 0 ? keysHi[i] : 0;
        const uint v = values[i];
        loSum += lo; loXor ^= lo; hiSum += hi; hiXor ^= hi;
        vSum += v; vXor ^= v; vSq += v * v;
        if (i > 0) {
            const uint plo = keysLo[i - 1];
            const uint phi = params.wide != 0 ? keysHi[i - 1] : 0;
            const uint pv = values[i - 1];
            const bool descends = phi > hi || (phi == hi && plo > lo);
            const bool tie = phi == hi && plo == lo;
            if (descends || (tie && pv > v)) {
                disorder += 1;
            }
        }
    }
    const uint row = c * 8;
    stats[row + 0] = disorder;
    stats[row + 1] = loSum; stats[row + 2] = loXor;
    stats[row + 3] = hiSum; stats[row + 4] = hiXor;
    stats[row + 5] = vSum;  stats[row + 6] = vXor;
    stats[row + 7] = vSq;
}
