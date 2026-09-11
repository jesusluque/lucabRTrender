// Copyright (c) 2026 lucabRTrender contributors.
module radix_totals;
import radix_common;

StructuredBuffer<uint>      histogram;
RWStructuredBuffer<uint>    digitTotals;   // 256
ConstantBuffer<RadixParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void radixTotals(uint3 tid: SV_DispatchThreadID) {
    const uint d = tid.x;
    if (d >= kDigits) {
        return;
    }
    uint sum = 0;
    for (uint c = 0; c < params.chunkCount; ++c) {
        sum += histogram[c * kDigits + d];
    }
    digitTotals[d] = sum;
}
