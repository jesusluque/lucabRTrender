// Copyright (c) 2026 lucabRTrender contributors.
module radix_scatter;
import radix_common;

StructuredBuffer<uint>      srcKeysLo;
StructuredBuffer<uint>      srcKeysHi;
StructuredBuffer<uint>      srcValues;
RWStructuredBuffer<uint>    dstKeysLo;
RWStructuredBuffer<uint>    dstKeysHi;
RWStructuredBuffer<uint>    dstValues;
RWStructuredBuffer<uint>    chunkStarts;   // consumed: each row is its chunk's cursors
ConstantBuffer<RadixParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void radixScatter(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint row = c * kDigits;
    const uint begin = c * params.chunkSize;
    const uint end = min(begin + params.chunkSize, params.count);
    for (uint i = begin; i < end; ++i) {
        const uint lo = srcKeysLo[i];
        const uint hi = params.wide != 0 ? srcKeysHi[i] : 0;
        const uint slot = row + digitOf(lo, hi, params.shift, params.wide);
        const uint at = chunkStarts[slot];
        dstKeysLo[at] = lo;
        if (params.wide != 0) {
            dstKeysHi[at] = hi;
        }
        dstValues[at] = srcValues[i];
        // The cursor advances once the pair is placed, not before. Slang's
        // CUDA emitter keeps `at` as a dereference of the cursor's own address
        // rather than a value, and re-reads it after this store, so a write
        // ordered after it lands one slot past where the cursor pointed; the
        // Metal emitter materialises the load and never showed it. The walk is
        // one invocation's own, so placing the pair first costs nothing.
        chunkStarts[slot] = at + 1;
    }
}
