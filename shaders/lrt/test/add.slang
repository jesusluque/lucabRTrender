// Copyright (c) 2026 lucabRTrender contributors.
//
// values[i] += amount, for i < count. The slang-rhi half of the shared-device
// test: gpe writes with `iota`, this adds, gpe reads the sum back.
module add;

struct AddParams {
    uint count;
    uint amount;
};

RWStructuredBuffer<uint>  values;
ConstantBuffer<AddParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void addMain(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    values[tid.x] += params.amount;
}
