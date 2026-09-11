// Copyright (c) 2026 lucabRTrender contributors.
//
// Which triangle of which instance each pixel sees, by rasterisation. The
// vertex shader pulls its vertex through the scene's pools (no vertex buffer);
// the fragment writes (instance + 1, triangle), 0 meaning nothing. What the
// pixel shows is then computed from those two numbers, the same way a ray
// traced visibility pass's are.
module visibility_raster;
import camera;
import lrt.world.scene_types;

struct DrawParams {
    uint firstInstance;
    uint firstPoint;
    uint firstTriangle;
    uint pad0;
};

StructuredBuffer<float4>          positions;
StructuredBuffer<uint>            indices;
StructuredBuffer<InstanceRecord>  instances;
ConstantBuffer<CameraParams>      camera;
ConstantBuffer<DrawParams>        draw;

struct VertexOut {
    float4 position : SV_Position;
    nointerpolation uint instance : INSTANCE;
    nointerpolation uint triangle : TRIANGLE;
};

[shader("vertex")]
VertexOut visibilityVertex(uint vertex : SV_VertexID, uint instanceIndex : SV_InstanceID) {
    VertexOut out;
    const uint instance = draw.firstInstance + instanceIndex;
    const uint triangle = vertex / 3;
    const uint point = draw.firstPoint + indices[(draw.firstTriangle * 3) + vertex];
    const InstanceRecord record = instances[instance];
    const float3 v = rowsApply(record.view0, record.view1, record.view2, positions[point].xyz, 1.0);
    out.position = viewToClip(camera, v);
    out.instance = instance;
    out.triangle = triangle;
    return out;
}

[shader("fragment")]
uint4 visibilityFragment(VertexOut input) : SV_Target0 {
    return uint4(input.instance + 1, input.triangle, 0, 0);
}
