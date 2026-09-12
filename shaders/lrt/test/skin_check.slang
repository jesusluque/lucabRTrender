// Copyright (c) 2026 lucabRTrender contributors.
//
// Skinning checked against closed forms written a second time: a point
// under one joint of weight 1 is the matrix chain worldToPrim * skelToWorld
// * joint * geomBind applied to it; under two joints turning about one axis
// with equal weight, dual quaternion skinning turns it by the mean angle
// exactly; blend shapes add each sub-shape's offset times its weight.
module skin_check;

struct CheckParams {
    uint  count;
    uint  mode;        // 0 chain, 1 mean-angle turn (about z, then the chain), 2 blend shapes
    float angle;       // mode 1: the mean angle, radians
    float tolerance;
    uint  ranged;      // mode 2: points with a range (past them the ranges buffer ends: an unguarded
                       // read there once looped a thread for billions of steps and hung the device)
    uint  pad0; uint pad1; uint pad2;
    float4 geomBind0; float4 geomBind1; float4 geomBind2; float4 geomBind3;   // rows, M * v
    float4 joint0; float4 joint1; float4 joint2; float4 joint3;
    float4 skelToWorld0; float4 skelToWorld1; float4 skelToWorld2; float4 skelToWorld3;
    float4 worldToPrim0; float4 worldToPrim1; float4 worldToPrim2; float4 worldToPrim3;
};

StructuredBuffer<uint>     restWords;    // float3 a point
StructuredBuffer<float4>   skinned;
StructuredBuffer<float4>   offsets;      // mode 2
StructuredBuffer<uint2>    ranges;
StructuredBuffer<float>    weights;
RWStructuredBuffer<uint>   counts;       // [checked, beyond]
RWStructuredBuffer<float>  worst;
ConstantBuffer<CheckParams> check;

float3 restAt(uint i) {
    return float3(asfloat(restWords[i * 3]), asfloat(restWords[i * 3 + 1]), asfloat(restWords[i * 3 + 2]));
}

float3 apply(float4 r0, float4 r1, float4 r2, float4 r3, float3 v) {
    const float4 h = float4(v, 1.0);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void skinCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint beyond = 0;
    float largest = 0.0;
    for (uint i = 0; i < check.count; ++i) {
        float3 expected = restAt(i);
        if (check.mode == 2 && i < check.ranged) {
            const uint2 range = ranges[i];
            for (uint k = range.x; k < range.y; ++k) {
                expected += offsets[k].xyz * weights[uint(offsets[k].w)];
            }
        } else {
            float3 p = apply(check.geomBind0, check.geomBind1, check.geomBind2, check.geomBind3, expected);
            if (check.mode == 1) {
                const float c = cos(check.angle);
                const float s = sin(check.angle);
                p = float3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
            } else {
                p = apply(check.joint0, check.joint1, check.joint2, check.joint3, p);
            }
            p = apply(check.skelToWorld0, check.skelToWorld1, check.skelToWorld2, check.skelToWorld3, p);
            expected = apply(check.worldToPrim0, check.worldToPrim1, check.worldToPrim2, check.worldToPrim3, p);
        }
        const float gap = length(skinned[i].xyz - expected);
        largest = max(largest, gap);
        beyond += gap > check.tolerance ? 1u : 0u;
    }
    counts[0] = check.count;
    counts[1] = beyond;
    worst[0] = largest;
}
