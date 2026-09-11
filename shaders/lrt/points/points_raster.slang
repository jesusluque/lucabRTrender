// Copyright (c) 2026 lucabRTrender contributors.
//
// Points through the rasteriser: no vertex buffer, no index buffer -- the
// vertex shader pulls position and colour out of the cloud's own storage by
// SV_VertexID, six vertices a point, and draws a screen-aligned quad the
// fragment shader rounds into a disc. Depth-tested, so the nearest point wins
// with no sort; the view depth goes to a second target for eye-dome lighting
// and for splats composited over the points.
//
// Not PointSize: D3D12 has no point size, and a pipeline that works on two of
// the three raster backends is not a pipeline.
module points_raster;
import lrt.common.packing;
import point_frame;

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      colours;
ConstantBuffer<PointParams> params;

struct VertexOut {
    float4 position : SV_Position;
    float3 colour   : COLOUR;
    float2 uv       : UV;
    float  viewZ    : VIEWZ;
};

struct FragmentOut {
    float4 colour : SV_Target0;
    float4 depth  : SV_Target1;   // view z in r; weight in g for surface splatting
};

static const float2 kCorners[6] = { float2(-1, -1), float2(1, -1), float2(1, 1),
                                    float2(-1, -1), float2(1, 1), float2(-1, 1) };

VertexOut buildVertex(uint vertex, float depthOffset) {
    VertexOut out;
    const uint i = vertex / 6;
    const float2 corner = kCorners[vertex % 6];
    const float3 v = pointToView(params, positions[i].xyz);
    out.uv = corner;
    out.viewZ = v.z;
    if (!(v.z > params.nearZ) || v.z > params.farZ) {
        out.position = float4(2.0, 2.0, 2.0, 1.0);   // outside the clip volume
        out.colour = float3(0.0);
        return out;
    }
    const float2 screen = pointScreen(params, v) + corner * pointRadius(params, v.z);
    // Screen pixels (y up) to NDC (y up). Depth is linear in view z: every
    // quad faces the camera, so nothing needs perspective-correct depth.
    out.position = float4(screen.x / float(params.width) * 2.0 - 1.0,
                          screen.y / float(params.height) * 2.0 - 1.0,
                          saturate((v.z + depthOffset - params.nearZ) / (params.farZ - params.nearZ)), 1.0);
    if (params.colourMode == 1) {
        out.colour = float3(params.colourR, params.colourG, params.colourB);
    } else {
        out.colour = float3(lowHalf(colours[i * 2]), highHalf(colours[i * 2]),
                            lowHalf(colours[i * 2 + 1]));
    }
    return out;
}

[shader("fragment")]
FragmentOut pointsFragment(VertexOut in) {
    const float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    FragmentOut out;
    if (params.surface != 0) {
        // Accumulation: colour weighted by a Gaussian over the disc, summed
        // additively; normalised by the resolve.
        const float w = exp(-2.0 * r2);
        out.colour = float4(in.colour * w, w);
        out.depth = float4(in.viewZ, w, 0.0, 0.0);
    } else {
        out.colour = float4(in.colour, 1.0);
        out.depth = float4(in.viewZ, 1.0, 0.0, 0.0);
    }
    return out;
}

[shader("vertex")]
VertexOut pointsVertex(uint vertex : SV_VertexID) {
    return buildVertex(vertex, 0.0);
}

/// Surface splatting's first pass: depth only, pushed back by `depthOffset`,
/// so the accumulation pass blends every point within that slack of the
/// nearest surface instead of only the nearest point. (A helper, not a call
/// from one entry point into another: Slang 2026.14 loops forever compiling
/// that.)
[shader("vertex")]
VertexOut pointsDepthVertex(uint vertex : SV_VertexID) {
    return buildVertex(vertex, params.depthOffset);
}
