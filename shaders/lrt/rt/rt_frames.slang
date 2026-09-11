// Copyright (c) 2026 lucabRTrender contributors.
//
// One cloud's particles as the ray tracer reads them, into the combined
// frames buffer at `base`: the map that makes each particle the unit sphere,
// and its cut. Once per cloud, so that a ray's work per particle is four
// reads and arithmetic, not a quaternion decode and three exponentials.
//
//   frames[4 i + 0..2]  M row k (local = M (x - centre)), centre k
//   frames[4 i + 3]     cutoff power, opacity, 0, 0
module rt_frames;
import lrt.common.packing;
import rt_common;

StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
RWStructuredBuffer<float4> frames;
ConstantBuffer<RtParams>   params;

[shader("compute")]
[numthreads(256, 1, 1)]
void rtFrames(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = positions[i];
    const uint row = i * 4;
    const float3 s = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    // local = S^-1 R^T (x - centre)
    const float3x3 m = mul(float3x3(1.0 / s.x, 0.0, 0.0, 0.0, 1.0 / s.y, 0.0, 0.0, 0.0, 1.0 / s.z),
                           transpose(rotationOf(decodeQuaternion(shape[row]))));
    const uint at = (params.base + i) * 4;
    frames[at] = float4(m[0], p.x);
    frames[at + 1] = float4(m[1], p.y);
    frames[at + 2] = float4(m[2], p.z);
    frames[at + 3] = float4(rtCutoff(p.w), p.w, 0.0, 0.0);
}
