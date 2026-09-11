// Copyright (c) 2026 lucabRTrender contributors.
//
// What a radix pass's kernel actually receives in its parameter block, written
// out per pass: for telling a backend that delivers a dispatch's own constants
// from one that hands every dispatch the same ones.
module radix_params_probe;
import lrt.algo.radix_common;

RWStructuredBuffer<uint>    seen;      // 4 per pass: count, chunkCount, shift, wide
ConstantBuffer<RadixParams> params;

struct ProbeWhich {
    uint pass;
    uint pad0; uint pad1; uint pad2;
};
ConstantBuffer<ProbeWhich> which;

[shader("compute")]
[numthreads(1, 1, 1)]
void radixParamsProbe(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint row = which.pass * 4;
    seen[row] = params.count;
    seen[row + 1] = params.chunkCount;
    seen[row + 2] = params.shift;
    seen[row + 3] = params.wide;
}
