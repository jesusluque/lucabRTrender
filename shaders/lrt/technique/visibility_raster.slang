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

struct PassParams {
    uint idsIncludeStart;   // SV_VertexID and SV_InstanceID count from the draw's starts (Metal)
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<float4>          positions;
StructuredBuffer<uint>            indices;
StructuredBuffer<MeshRecord>      meshes;
StructuredBuffer<InstanceRecord>  instances;
ConstantBuffer<CameraParams>      camera;
ConstantBuffer<PassParams>        pass;

struct VertexOut {
    float4 position : SV_Position;
    nointerpolation uint instance : INSTANCE;
    nointerpolation uint triangle : TRIANGLE;
    nointerpolation uint cullBack : CULL;   // 1: single sided, keep the front only
};

// One draw per mesh, all sharing the pass's bindings: a draw starts at its
// mesh's first index corner and its first instance record.
[shader("vertex")]
VertexOut visibilityVertex(uint vertex : SV_VertexID, uint instanceIndex : SV_InstanceID,
                           uint startVertex : SV_StartVertexLocation, uint startInstance : SV_StartInstanceLocation) {
    VertexOut out;
    const uint corner = pass.idsIncludeStart != 0 ? vertex : vertex + startVertex;
    const uint instance = pass.idsIncludeStart != 0 ? instanceIndex : instanceIndex + startInstance;
    const InstanceRecord record = instances[instance];
    const MeshRecord mesh = meshes[record.mesh];
    const uint triangle = corner / 3 - mesh.firstTriangle;
    const uint point = mesh.firstPoint + indices[corner];
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
