// Copyright (c) 2026 lucabRTrender contributors.
//
// A closed box whose every face emits E and reflects rho, seen from inside:
// after N bounces the radiance is E (1 + rho + ... + rho^N) exactly, and with
// cosine sampling of a Lambert lobe each bounce's weight over pdf is rho with
// no variance at all, so every covered pixel must read the closed form to
// float precision. Counts the pixels that do not.
module furnace_check;

struct FurnaceParams {
    float emission;
    float albedo;
    uint  bounces;
    float tolerance;   // relative
    uint  pixels;
    uint  pad0; uint pad1; uint pad2;
};

StructuredBuffer<float4>     colour;
RWStructuredBuffer<uint>     counts;   // [covered, beyond tolerance]
RWStructuredBuffer<float>    worst;    // [largest relative error]
ConstantBuffer<FurnaceParams> furnace;

[shader("compute")]
[numthreads(1, 1, 1)]
void furnaceCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    // The geometric series, summed the long way so it shares nothing with a
    // closed-form identity the renderer might also lean on.
    float expected = 0.0;
    float term = furnace.emission;
    for (uint k = 0; k <= furnace.bounces; ++k) {
        expected += term;
        term *= furnace.albedo;
    }
    uint covered = 0;
    uint beyond = 0;
    float largest = 0.0;
    for (uint i = 0; i < furnace.pixels; ++i) {
        const float4 c = colour[i];
        if (c.w <= 0.0) {
            continue;
        }
        ++covered;
        const float3 rel = abs(c.rgb - expected) / expected;
        const float r = max(rel.r, max(rel.g, rel.b));
        largest = max(largest, r);
        beyond += r > furnace.tolerance ? 1u : 0u;
    }
    counts[0] = covered;
    counts[1] = beyond;
    worst[0] = largest;
}
