// Copyright (c) 2026 lucabRTrender contributors.
//
// Checks an exclusive prefix sum on the GPU, element by element:
// output[0] == 0 and output[i] == output[i-1] + input[i-1]. Writes, per chunk,
// how many elements disagree, as a stats row stats_reduce can fold; the last
// chunk also checks the total.
module prefix_check;

struct CheckParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint pad0;
};

StructuredBuffer<uint>      input;
StructuredBuffer<uint>      output;
StructuredBuffer<uint>      total;
RWStructuredBuffer<uint>    errors;   // chunkCount * 8, a stats row: [0] errors, rest zero
ConstantBuffer<CheckParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void prefixCheck(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint begin = c * params.chunkSize;
    const uint end = min(begin + params.chunkSize, params.count);
    uint wrong = 0;
    for (uint i = begin; i < end; ++i) {
        const uint expected = i == 0 ? 0 : output[i - 1] + input[i - 1];
        if (output[i] != expected) {
            wrong += 1;
        }
    }
    if (end == params.count && params.count > 0 &&
        total[0] != output[params.count - 1] + input[params.count - 1]) {
        wrong += 1;
    }
    errors[c * 8] = wrong;
    for (uint k = 1; k < 8; ++k) {
        errors[c * 8 + k] = 0;
    }
}
