// Copyright (c) 2026 lucabRTrender contributors.
//
// The radix scatter's shape with none of its arithmetic: the same seven
// buffers bound the same way, one invocation walking the same two elements,
// writing out what each read returned. A backend that hands the kernel the
// right buffers (binding_probe says it does) but the wrong contents through
// them shows up here, one kernel away from the sort instead of four.
module scatter_shape_probe;

struct ShapeParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint shift;
    uint wide;
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>     srcKeysLo;
StructuredBuffer<uint>     srcKeysHi;
StructuredBuffer<uint>     srcValues;
RWStructuredBuffer<uint>   dstKeysLo;
RWStructuredBuffer<uint>   dstKeysHi;
RWStructuredBuffer<uint>   dstValues;
RWStructuredBuffer<uint>   chunkStarts;
RWStructuredBuffer<uint>   saw;          // count * 4: the lo, hi and value each read gave, and the slot used
ConstantBuffer<ShapeParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void scatterShapeProbe(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint begin = c * params.chunkSize;
    const uint end = min(begin + params.chunkSize, params.count);
    for (uint i = begin; i < end; ++i) {
        const uint lo = srcKeysLo[i];
        const uint hi = params.wide != 0 ? srcKeysHi[i] : 0;
        saw[i * 4] = lo;
        saw[i * 4 + 1] = hi;
        saw[i * 4 + 2] = srcValues[i];
        // The writes the scatter makes, to the slots it would use, so the
        // dispatch has the same reads and writes in flight as the real one.
        const uint at = chunkStarts[c * 256];
        saw[i * 4 + 3] = at;   // the slot every write below is meant to use
        dstKeysLo[at] = lo;
        if (params.wide != 0) {
            dstKeysHi[at] = hi;
        }
        dstValues[at] = srcValues[i];
        // The cursor advances last, as the scatter's does: see radix_scatter.
        chunkStarts[c * 256] = at + 1;
    }
}
