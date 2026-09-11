// Copyright (c) 2026 lucabRTrender contributors.
module count_nonzero;

struct CountParams {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint>      values;
RWStructuredBuffer<uint>    total;   // 1
ConstantBuffer<CountParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void countNonzero(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint n = 0;
    for (uint i = 0; i < params.count; ++i) {
        n += values[i] != 0 ? 1 : 0;
    }
    total[0] = n;
}
