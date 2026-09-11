// Copyright (c) 2026 lucabRTrender contributors.
//
// Two responses per sample -- a material's lobes evaluated, and MaterialX's
// genglsl closures for the same view and light -- compared: how many differ
// by more than a relative tolerance (with an absolute floor), the worst.
module material_compare;

struct CompareParams {
    uint  count;
    float relative;
    float floor;
    uint  pad0;
};

StructuredBuffer<float4>     ours;
StructuredBuffer<float4>     theirs;
RWStructuredBuffer<uint>     counts;   // [over tolerance, samples with any response, lobes seen at most]
RWStructuredBuffer<float4>   worst;    // [worst relative difference, its ours.g, its theirs.g, mean response]
ConstantBuffer<CompareParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void materialCompare(uint3 tid: SV_DispatchThreadID) {
    uint over = 0;
    uint lit = 0;
    uint lobes = 0;
    float worstError = 0.0;
    float worstOurs = 0.0;
    float worstTheirs = 0.0;
    float sum = 0.0;
    for (uint i = 0; i < params.count; ++i) {
        const float3 a = ours[i].rgb;
        const float3 b = theirs[i].rgb;
        lobes = max(lobes, uint(ours[i].a));
        sum += b.g;
        lit += max(max(b.r, b.g), b.b) > params.floor ? 1u : 0u;
        for (uint c = 0; c < 3; ++c) {
            const float error = abs(a[c] - b[c]) / max(max(abs(a[c]), abs(b[c])), params.floor);
            if (error > worstError) {
                worstError = error;
                worstOurs = a[c];
                worstTheirs = b[c];
            }
            over += error > params.relative ? 1u : 0u;
        }
    }
    counts[0] = over;
    counts[1] = lit;
    counts[2] = lobes;
    worst[0] = float4(worstError, worstOurs, worstTheirs, sum / float(params.count));
}
