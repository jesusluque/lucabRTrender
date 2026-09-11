// Copyright (c) 2026 lucabRTrender contributors.
//
// The reference's own projection of points as discs: where, how big, which
// colour -- without tiles or rectangles. See reference_project.slang.
module reference_points_project;
import lrt.common.packing;
import lrt.points.point_frame;
import lrt.splat.frame;

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      colours;
RWStructuredBuffer<float4>  refScreen;
RWStructuredBuffer<float4>  refConic;
RWStructuredBuffer<float4>  refColour;
ConstantBuffer<FrameParams> params;
ConstantBuffer<PointParams> points;

[shader("compute")]
[numthreads(64, 1, 1)]
void referencePointsProject(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    refScreen[slot] = float4(0.0, 0.0, -1.0, 0.0);
    const float3 v = toView(params, positions[i].xyz);
    if (!(v.z > params.nearZ) || v.z > params.farZ) {
        return;
    }
    const float r = points.sizeMode == 1 ? max(points.size * 0.5, 0.5)
                  : max((params.orthographic != 0 ? points.size * 0.5 * params.focalX
                                                  : points.size * 0.5 * params.focalX / v.z), 0.5);
    const float2 centre = params.orthographic != 0
        ? float2(params.focalX * v.x + params.centreX, params.focalY * v.y + params.centreY)
        : float2(params.focalX * v.x / v.z + params.centreX, params.focalY * v.y / v.z + params.centreY);
    refScreen[slot] = float4(centre, v.z, 1.0);
    refConic[slot] = float4(1.0 / (r * r), 0.0, 1.0 / (r * r), 1.0);
    const float3 rgb = points.colourMode == 1 ? float3(points.colourR, points.colourG, points.colourB)
                     : float3(lowHalf(colours[i * 2]), highHalf(colours[i * 2]), lowHalf(colours[i * 2 + 1]));
    refColour[slot] = float4(rgb, 1.0);
}
