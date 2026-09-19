// Copyright (c) 2026 lucabRTrender contributors.
//
// Counting what the per-part visibility got wrong: a body slab under a wing
// slab, one light from above. Written by a kernel, checked by a kernel.
module visibility_parts_check;

struct CheckParams {
    uint  count;       // gaussians
    uint  lights;      // factors a gaussian
    uint  light;       // which one to check
    float cut;         // a factor under this is "shadowed"
    float wingFromX;   // the wing covers x >= this ...
    float wingToX;     // ... and x < this, in the body's own space
    float bodyZ;       // the slab whose gaussians are receivers
    float band;        // this far either side of an edge is penumbra: not counted
};

StructuredBuffer<float4>   positions;   // the bind pose: x, y, z, opacity
StructuredBuffer<float>    factors;
RWStructuredBuffer<uint>   counts;      // [under & lit, clear & shadowed, under, clear]
ConstantBuffer<CheckParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void vpCount(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = positions[i];
    if (abs(p.z - params.bodyZ) > 1.0e-3) {
        return;   // a wing gaussian is not a receiver here
    }
    const float f = factors[i * params.lights + params.light];
    const bool under = p.x >= params.wingFromX + params.band && p.x < params.wingToX - params.band;
    const bool clear = p.x < params.wingFromX - params.band || p.x >= params.wingToX + params.band;
    if (under) {
        InterlockedAdd(counts[2], 1u);
        if (f >= params.cut) InterlockedAdd(counts[0], 1u);
    } else if (clear) {
        InterlockedAdd(counts[3], 1u);
        if (f < params.cut) InterlockedAdd(counts[1], 1u);
    }
}
