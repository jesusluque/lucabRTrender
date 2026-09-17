// Copyright (c) 2026 lucabRTrender contributors.
//
// What the packed shadow tables answer, and what a cloud did to a frame.
//
// `packedShadowRays` runs the packed query (rt_shadow_packed.slang) over the
// same rays the unpacked kernel takes, so the two can be held to the same
// number: the layout is an agreement between a packing kernel and a reader,
// and an agreement wants a test rather than a comment.
//
// `splatShadowRatio` reads two frames of the same scene -- one with the cloud
// shadowing, one without -- and counts the pixels whose ratio is not what the
// cloud lets through. Under the axis of an isotropic particle a ray peaks at
// its centre with power 0, so the ratio there is `1 - opacity` exactly; away
// from the cloud it is 1.
module splat_shadow_check;
import lrt.rt.rt_common;
import lrt.rt.rt_shadow_packed;

struct CheckParams {
    PackedShadow where;
    float cut;
    float nearZ;
    float farZ;
    uint  rays;
};

StructuredBuffer<float4>          packed;
StructuredBuffer<float4>          rays;       // two per ray: origin/tMin, direction/tMax
RWStructuredBuffer<float>         answers;    // one per ray
ConstantBuffer<CheckParams>       check;
uniform RaytracingAccelerationStructure cloud;

[shader("compute")]
[numthreads(64, 1, 1)]
void packedShadowRays(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= check.rays) {
        return;
    }
    const float4 a = rays[r * 2];
    const float4 b = rays[r * 2 + 1];
    LrtRay ray;
    ray.Origin = a.xyz;
    ray.TMin = a.w;
    ray.Direction = b.xyz;
    ray.TMax = b.w;
    answers[r] = packedShadowTransmittance(packed, check.where, cloud, ray, check.cut, check.nearZ, check.farZ);
}

struct RatioParams {
    uint  pixels;
    uint  under;        ///< the pixel whose shadow ray goes through the centre
    uint  beside;       ///< a pixel the cloud stands nowhere near
    float wantUnder;    ///< 1 - opacity there
    float wantBeside;   ///< 1, since nothing is in the way
    float tolerance;    ///< relative
    float floorLevel;   ///< a pixel darker than this in the unshadowed frame says nothing
    uint  pad0;
};

StructuredBuffer<float4>  shadowed;     // the frame the cloud shadowed
StructuredBuffer<float4>  unshadowed;   // the same frame without it
RWStructuredBuffer<uint>  counts;       // [lit, darker than they should be, beyond tolerance at the two pixels]
RWStructuredBuffer<float> ratios;       // [under, beside], for the test to print
ConstantBuffer<RatioParams> ratio;

[shader("compute")]
[numthreads(1, 1, 1)]
void splatShadowRatio(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint lit = 0;
    uint brighter = 0;
    for (uint i = 0; i < ratio.pixels; ++i) {
        const float b = unshadowed[i].r;
        if (!(b > ratio.floorLevel)) {
            continue;
        }
        ++lit;
        // A cloud only ever takes light away: no pixel may come out brighter
        // than the frame that had no cloud in it.
        if (shadowed[i].r > b * (1.0 + 1.0e-4)) {
            ++brighter;
        }
    }
    const float under = unshadowed[ratio.under].r > 0.0 ? shadowed[ratio.under].r / unshadowed[ratio.under].r : 0.0;
    const float beside = unshadowed[ratio.beside].r > 0.0 ? shadowed[ratio.beside].r / unshadowed[ratio.beside].r : 0.0;
    ratios[0] = under;
    ratios[1] = beside;
    uint wrong = 0;
    wrong += abs(under - ratio.wantUnder) > ratio.tolerance * max(ratio.wantUnder, 1.0e-3) ? 1u : 0u;
    wrong += abs(beside - ratio.wantBeside) > ratio.tolerance * max(ratio.wantBeside, 1.0e-3) ? 1u : 0u;
    counts[0] = lit;
    counts[1] = brighter;
    counts[2] = wrong;
}
