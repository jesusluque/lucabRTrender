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
    nointerpolation uint cullBack : CULL;   // 1: single sided, keep the front only
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
    // Hydra's rule (Storm's, HdCullStyle's default): a mesh that is not
    // double sided shows its front only -- counter-clockwise for a
    // right-handed mesh, and turned round when its transform mirrors it.
    const float3x3 linear = float3x3(record.view0.xyz, record.view1.xyz, record.view2.xyz);
    const bool mirrored = determinant(linear) < 0.0;
    out.cullBack = (record.flags & 1u) == 0 ? (mirrored ? 2u : 1u) : 0u;
    return out;
}

[shader("fragment")]
uint4 visibilityFragment(VertexOut input, bool front : SV_IsFrontFace) : SV_Target0 {
    // The view flips z (+z forward), which turns the winding round once more.
    if ((input.cullBack == 1 && front) || (input.cullBack == 2 && !front)) {
        discard;
    }
    return uint4(input.instance + 1, input.triangle, 0, 0);
}
