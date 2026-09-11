// Copyright (c) 2026 lucabRTrender contributors.
module histogram_reduce;

struct ReduceParams {
    uint rows;
    uint bins;
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint>       histogram;
RWStructuredBuffer<uint>     total;   // bins
ConstantBuffer<ReduceParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void histogramReduce(uint3 tid: SV_DispatchThreadID) {
    const uint bin = tid.x;
    if (bin >= params.bins) {
        return;
    }
    uint sum = 0;
    for (uint r = 0; r < params.rows; ++r) {
        sum += histogram[r * params.bins + bin];
    }
    total[bin] = sum;
}
