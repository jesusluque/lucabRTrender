// Copyright (c) 2026 lucabRTrender contributors.
//
// Eye-dome lighting (Boucheny 2009): shade each pixel by how much nearer
// its neighbours at `edlRadius` pixels are, in log depth. No normals, no
// lights -- the only depth cue a bare point cloud has, and what CloudCompare
// and Potree draw by default.
module points_edl;
import point_frame;

StructuredBuffer<float4>    colourIn;
StructuredBuffer<float>     depth;
RWStructuredBuffer<float4>  colourOut;
ConstantBuffer<PointParams> params;

static const int2 kNeighbours[8] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
                                     int2(1, 1), int2(-1, -1), int2(1, -1), int2(-1, 1) };

[shader("compute")]
[numthreads(16, 16, 1)]
void pointsEdl(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const float z = depth[at];
    const float4 c = colourIn[at];
    if (!(z > 0.0)) {
        colourOut[at] = c;
        return;
    }
    const float here = log2(z);
    float response = 0.0;
    for (uint k = 0; k < 8; ++k) {
        const int2 q = int2(tid.xy) + kNeighbours[k] * int(max(params.edlRadius, 1.0));
        if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height)) {
            continue;
        }
        const float n = depth[uint(q.y) * params.width + uint(q.x)];
        // Background counts as infinitely far: an edge against the sky is
        // the strongest edge there is.
        response += n > 0.0 ? max(0.0, here - log2(n)) : 4.0;
    }
    const float shade = exp(-response / 8.0 * 300.0 * params.edlStrength);
    colourOut[at] = float4(c.rgb * shade, c.w);
}
