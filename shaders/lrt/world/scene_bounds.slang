// Copyright (c) 2026 lucabRTrender contributors.
//
// Every instance's mesh box in world space, as two float4 points (its low and
// high corners) for lrt/scene/bounds_chunks to fold: where a scene is, for a
// viewer to frame it.
module scene_bounds;
import scene_types;

struct SceneBoundsParams {
    uint count;
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<MeshRecord>      meshes;
StructuredBuffer<InstanceRecord>  instances;
RWStructuredBuffer<float4>        corners;   // 2 per instance
ConstantBuffer<SceneBoundsParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void instanceWorldBoxes(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const InstanceRecord r = instances[i];
    const MeshRecord m = meshes[r.mesh];
    // A box through an affine map: the centre moves, the half extents go
    // through the absolute linear part.
    const float3 centre = (m.boundsLo.xyz + m.boundsHi.xyz) * 0.5;
    const float3 half = (m.boundsHi.xyz - m.boundsLo.xyz) * 0.5;
    const float3 c = rowsApply(r.world0, r.world1, r.world2, centre, 1.0);
    const float3 e = float3(dot(abs(r.world0.xyz), half), dot(abs(r.world1.xyz), half), dot(abs(r.world2.xyz), half));
    corners[i * 2] = float4(c - e, 1.0);
    corners[i * 2 + 1] = float4(c + e, 1.0);
}
