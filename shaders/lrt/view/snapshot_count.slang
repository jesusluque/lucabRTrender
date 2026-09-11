// Copyright (c) 2026 lucabRTrender contributors.
//
// How many pixels of a shown frame are brighter than a threshold: that the
// viewer showed something, as a count.
module snapshot_count;

struct CountParams {
    uint  width;
    uint  height;
    float threshold;
    uint  pad0;
};

Texture2D<float4>          shown;
RWStructuredBuffer<uint>   lit;
ConstantBuffer<CountParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void snapshotCount(uint3 tid: SV_DispatchThreadID) {
    uint n = 0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            const float3 c = shown.Load(int3(int(x), int(y), 0)).rgb;
            n += max(max(c.r, c.g), c.b) > params.threshold ? 1u : 0u;
        }
    }
    lit[0] = n;
}
