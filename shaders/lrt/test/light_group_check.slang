// Copyright (c) 2026 lucabRTrender contributors.
//
// Light groups against the beauty: at every pixel the groups' planes summed
// must be the colour (to a relative tolerance: the same terms added in
// another order), and a group no light belongs to must be exactly zero.
// Counts, not images, come back.
struct GroupCheck {
    uint  pixels;
    uint  groups;      // planes in `planes`, one after another
    uint  empty;       // which plane must be zero; 0xFFFFFFFF for none
    float tolerance;   // relative, per channel
};

ConstantBuffer<GroupCheck>  check;
StructuredBuffer<float4>    beauty;
StructuredBuffer<float4>    planes;
RWStructuredBuffer<uint>    counts;   // [sum off, empty nonzero, covered, checked]
RWStructuredBuffer<uint>    worst;    // [largest relative difference] as float bits, by a max (positive floats order as their bits)

void maxFloat(uint slot, float value) {
    uint bits = asuint(max(value, 0.0));
    InterlockedMax(worst[slot], bits);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lightGroupSum(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p >= check.pixels) {
        return;
    }
    const float4 b = beauty[p];
    float4 sum = float4(0.0);
    for (uint g = 0; g < check.groups; ++g) {
        const float4 v = planes[g * check.pixels + p];
        sum.xyz += v.xyz;
        // The plane's w is the pixel's opacity, as the colour's; a group no
        // light belongs to must have no light in it.
        if (g == check.empty && any(v.xyz != float3(0.0))) {
            InterlockedAdd(counts[1], 1);
        }
    }
    InterlockedAdd(counts[3], 1);
    if (b.w > 0.0) {
        InterlockedAdd(counts[2], 1);
    }
    float largest = 0.0;
    for (uint c = 0; c < 3; ++c) {
        const float scale = max(abs(b[c]), 1.0e-3);
        largest = max(largest, abs(sum[c] - b[c]) / scale);
    }
    maxFloat(0, largest);
    if (largest > check.tolerance) {
        InterlockedAdd(counts[0], 1);
    }
}
