// Copyright (c) 2026 lucabRTrender contributors.
module radix_histogram;
import radix_common;

StructuredBuffer<uint>      keysLo;
StructuredBuffer<uint>      keysHi;
RWStructuredBuffer<uint>    histogram;   // chunkCount * 256
ConstantBuffer<RadixParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void radixHistogram(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint row = c * kDigits;
    for (uint d = 0; d < kDigits; ++d) {
        histogram[row + d] = 0;
    }
    const uint begin = c * params.chunkSize;
    const uint end = min(begin + params.chunkSize, params.count);
    for (uint i = begin; i < end; ++i) {
        const uint hi = params.wide != 0 ? keysHi[i] : 0;
        histogram[row + digitOf(keysLo[i], hi, params.shift, params.wide)] += 1;
    }
}
