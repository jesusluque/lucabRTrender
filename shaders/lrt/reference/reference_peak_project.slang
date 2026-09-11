// Copyright (c) 2026 lucabRTrender contributors.
//
// The ray tracer's ground truth, first half: every particle in view space as
// the map that makes it the unit sphere (local = M (x - centre)), its colour
// and its opacity. Written apart from the ray tracer -- no proxies, no BVH, no
// segments, no carry -- so that none of those can hide a bug in both.
module reference_peak_project;
import lrt.common.packing;
import lrt.common.sh;
import lrt.splat.frame;

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
StructuredBuffer<uint>      sh;
RWStructuredBuffer<float4>  peakRow0;    // M row 0, centre.x
RWStructuredBuffer<float4>  peakRow1;    // M row 1, centre.y
RWStructuredBuffer<float4>  peakRow2;    // M row 2, centre.z
RWStructuredBuffer<float4>  peakColour;  // rgb (encoded), opacity; w < 0: skipped
ConstantBuffer<FrameParams> params;

float3x3 inverse3(float3x3 m) {
    const float3 a = m[0];
    const float3 b = m[1];
    const float3 c = m[2];
    const float3 r0 = cross(b, c);
    const float3 r1 = cross(c, a);
    const float3 r2 = cross(a, b);
    const float det = dot(a, r0);
    // Columns of the inverse are the cross products over the determinant.
    return transpose(float3x3(r0, r1, r2)) / det;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void referencePeakProject(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    peakColour[slot] = float4(0.0, 0.0, 0.0, -1.0);
    const float4 p = positions[i];
    if (!(p.w >= 1.0 / 255.0)) {
        return;
    }
    const uint row = i * 4;
    const float3 s = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    const float4 q = decodeQuaternion(shape[row]);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float3x3 r = float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                                2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                                2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
    // view = V object; object = R S local  =>  local = S^-1 R^T V^-1 (view - centre)
    const float3x3 m = mul(float3x3(1.0 / s.x, 0.0, 0.0, 0.0, 1.0 / s.y, 0.0, 0.0, 0.0, 1.0 / s.z),
                           mul(transpose(r), inverse3(viewLinear(params))));
    const float3 centre = toView(params, p.xyz);
    peakRow0[slot] = float4(m[0], centre.x);
    peakRow1[slot] = float4(m[1], centre.y);
    peakRow2[slot] = float4(m[2], centre.z);
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const uint degree = min(degreeOf(params.restPerColour), params.shLimit);
    const float3 dir = normalize(p.xyz - float3(params.eyeX, params.eyeY, params.eyeZ));
    peakColour[slot] = float4(max(base + evaluateRest(sh, i * params.shWords, degree, dir), 0.0), p.w);
}
