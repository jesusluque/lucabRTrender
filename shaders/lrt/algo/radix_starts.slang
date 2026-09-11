// Copyright (c) 2026 lucabRTrender contributors.
module radix_starts;
import radix_common;

StructuredBuffer<uint>      histogram;
StructuredBuffer<uint>      digitTotals;
RWStructuredBuffer<uint>    chunkStarts;   // chunkCount * 256
ConstantBuffer<RadixParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void radixStarts(uint3 tid: SV_DispatchThreadID) {
    const uint d = tid.x;
    if (d >= kDigits) {
        return;
    }
    uint running = 0;
    for (uint below = 0; below < d; ++below) {
        running += digitTotals[below];
    }
    for (uint c = 0; c < params.chunkCount; ++c) {
        chunkStarts[c * kDigits + d] = running;
        running += histogram[c * kDigits + d];
    }
}
