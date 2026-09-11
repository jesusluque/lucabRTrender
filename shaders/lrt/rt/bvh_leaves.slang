// Copyright (c) 2026 lucabRTrender contributors.
//
// Per particle: the box around its cut ellipsoid and the Morton code of its
// centre within the cloud's bounds, for the radix sort that orders the leaves.
module bvh_leaves;
import lrt.common.packing;
import rt_common;
import bvh_common;

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
RWStructuredBuffer<float4>  leafBoxes;   // per particle: lo, hi (cloud-local)
RWStructuredBuffer<uint>    mortonKeys;
RWStructuredBuffer<uint>    mortonValues;
ConstantBuffer<BvhParams>   params;

uint spread(uint v) {
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bvhLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = positions[i];
    const uint row = i * 4;
    const float3 s = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    const float3x3 rs = mul(rotationOf(decodeQuaternion(shape[row])),
                            float3x3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z));
    // The ellipsoid x = c + (R S) u, |u| <= r, reaches r |row a of R S| along axis a.
    const float r = sqrt(rtCutoff(p.w));
    const float3 half = r * float3(length(rs[0]), length(rs[1]), length(rs[2]));
    leafBoxes[i * 2] = float4(p.xyz - half, 0.0);
    leafBoxes[i * 2 + 1] = float4(p.xyz + half, 0.0);

    const float3 lo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ);
    const float3 hi = float3(params.boundsHiX, params.boundsHiY, params.boundsHiZ);
    const float3 unit = saturate((p.xyz - lo) / max(hi - lo, float3(1e-20)));
    const uint3 q = min(uint3(unit * 1024.0), uint3(1023u));
    mortonKeys[i] = (spread(q.x) << 2) | (spread(q.y) << 1) | spread(q.z);
    mortonValues[i] = i;
}
