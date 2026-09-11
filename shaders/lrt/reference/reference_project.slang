// Copyright (c) 2026 lucabRTrender contributors.
//
// The ground truth's own projection. Written out plainly and separately from
// splat_project -- no tile rectangle, no tile test, no footprint extent, no
// depth key -- so a bug in how the rasteriser bins, counts, emits or sorts
// cannot also be in the thing it is checked against. The Gaussian itself (EWA,
// dilation, the cut at cutoffPower) is the same definition: the reference
// answers "is the rasteriser the renderer it claims to be", and analytic
// tests answer "is that the right Gaussian".
module reference_project;
import lrt.common.packing;
import lrt.common.sh;
import lrt.splat.frame;
import lrt.common.edit;

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
StructuredBuffer<uint>      sh;
RWStructuredBuffer<float4>  refScreen;   // centre.xy, view z (or -1: culled), cutoff
RWStructuredBuffer<float4>  refConic;    // a, b, c, alpha
RWStructuredBuffer<float4>  refColour;
ConstantBuffer<FrameParams> params;

float3x3 rotationOf(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void referenceProject(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    refScreen[slot] = float4(0.0, 0.0, -1.0, 0.0);

    float4 p = positions[i];
    const uint row = i * 4;
    const float3 s = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    const EditFate fate = editFate(params.edit, p.xyz, p.w, s);
    if (fate.removed) {
        return;
    }
    p.w = fate.opacity;
    const float3 v = toView(params, p.xyz);
    if (!(v.z > params.nearZ) || v.z > params.farZ) {
        return;
    }
    const float3x3 rs = mul(rotationOf(decodeQuaternion(shape[row])),
                            float3x3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z));
    const float3x3 cov3 = mul(rs, transpose(rs));
    const float3x3 w = viewLinear(params);
    const float3x3 covView = mul(w, mul(cov3, transpose(w)));

    float2 centre;
    float3x3 j = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (params.orthographic != 0) {
        centre = float2(params.focalX * v.x + params.centreX, params.focalY * v.y + params.centreY);
        j[0][0] = params.focalX;
        j[1][1] = params.focalY;
    } else {
        centre = float2(params.focalX * v.x / v.z + params.centreX,
                        params.focalY * v.y / v.z + params.centreY);
        const float lox = (-0.15 * float(params.width) - params.centreX) / params.focalX;
        const float hix = (1.15 * float(params.width) - params.centreX) / params.focalX;
        const float loy = (-0.15 * float(params.height) - params.centreY) / params.focalY;
        const float hiy = (1.15 * float(params.height) - params.centreY) / params.focalY;
        const float cx = clamp(v.x / v.z, lox, hix) * v.z;
        const float cy = clamp(v.y / v.z, loy, hiy) * v.z;
        j[0][0] = params.focalX / v.z;
        j[0][2] = -params.focalX * cx / (v.z * v.z);
        j[1][1] = params.focalY / v.z;
        j[1][2] = -params.focalY * cy / (v.z * v.z);
    }
    const float3x3 cov2 = mul(j, mul(covView, transpose(j)));
    const float before = cov2[0][0] * cov2[1][1] - cov2[0][1] * cov2[0][1];
    const float a = cov2[0][0] + 0.3;
    const float b = cov2[0][1];
    const float c = cov2[1][1] + 0.3;
    const float det = a * c - b * b;
    if (!(det > 1e-12)) {
        return;
    }
    float alpha = p.w;
    if (params.antialias != 0) {
        alpha *= sqrt(max(before, 0.0) / det);
    }
    if (!(alpha >= 1.0 / 255.0)) {
        return;
    }
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const uint degree = min(degreeOf(params.restPerColour), params.shLimit);
    const float3 dir = normalize(p.xyz - float3(params.eyeX, params.eyeY, params.eyeZ));
    refScreen[slot] = float4(centre, v.z, cutoffPower(alpha));
    refConic[slot] = float4(c / det, -b / det, a / det, alpha);
    float3 colour = max(base + evaluateRest(sh, i * params.shWords, degree, dir), 0.0);
    if (fate.graded) {
        colour = editGrade(params.edit, colour);
    }
    refColour[slot] = float4(colour, 0.0);
}
