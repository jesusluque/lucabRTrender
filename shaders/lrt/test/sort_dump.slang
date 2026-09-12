// Copyright (c) 2026 lucabRTrender contributors.
//
// The first few (key, value) triples of a sort's buffers, copied into one
// buffer so a test can look at what a pass actually wrote -- for finding which
// stage of a sort a backend gets wrong, without a CPU sort to compare against.
module sort_dump;

struct DumpParams {
    uint count;    // how many entries to copy
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>     keysLo;
StructuredBuffer<uint>     keysHi;
StructuredBuffer<uint>     values;
RWStructuredBuffer<uint>   dump;     // count * 3: lo, hi, value
ConstantBuffer<DumpParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void sortDump(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    dump[i * 3] = keysLo[i];
    dump[i * 3 + 1] = keysHi[i];
    dump[i * 3 + 2] = values[i];
}
