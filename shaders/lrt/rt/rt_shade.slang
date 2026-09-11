// Copyright (c) 2026 lucabRTrender contributors.
//
// One instance's particle colours for this frame, into the colour buffer at
// `base`. A particle's colour depends on the direction from the eye to its
// centre (the rasteriser's rule, and how 3DGS trained it), not on the ray, so
// it is evaluated once a frame here instead of once for every ray it reaches.
module rt_shade;
import lrt.common.packing;
import lrt.common.sh;
import rt_common;

StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
StructuredBuffer<uint>     sh;
RWStructuredBuffer<float4> colours;
ConstantBuffer<RtParams>   params;   // count, base, shWords, restPerColour, shLimit; v03/v13/v23: eye in cloud space

[shader("compute")]
[numthreads(256, 1, 1)]
void rtShade(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint row = i * 4;
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const uint degree = min(degreeOf(params.restPerColour), params.shLimit);
    const float3 eye = float3(params.v03, params.v13, params.v23);
    const float3 dir = normalize(positions[i].xyz - eye);
    colours[params.base + i] = float4(max(base + evaluateRest(sh, i * params.shWords, degree, dir), 0.0), 0.0);
}
