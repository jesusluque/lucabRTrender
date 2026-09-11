// Copyright (c) 2026 lucabRTrender contributors.
//
// What the visibility rasteriser draws: one draw per mesh whose vertices are
// pulled through the scene's pools (no vertex buffer), each carrying the
// (instance, triangle) its fragments write. The entry points are elsewhere --
// visibility_raster for the plain pass, a generated module for the one that
// cuts fragments out by their material's opacity -- so both draw the same
// geometry the same way.
module visibility_geometry;
__exported import lrt.technique.surface;

public struct PassParams {
    public uint idsIncludeStart;   // SV_VertexID and SV_InstanceID count from the draw's starts (Metal)
    public uint pad0; public uint pad1; public uint pad2;
};

public ConstantBuffer<PassParams> pass;

public struct VertexOut {
    public float4 position : SV_Position;
    public nointerpolation uint instance : INSTANCE;
    public nointerpolation uint triangle : TRIANGLE;
    public nointerpolation uint cullBack : CULL;   // 1: single sided, keep the front only
};

/// A draw starts at its mesh's first index corner and its first instance record.
public VertexOut visibilityVertexOf(CameraParams camera, uint vertex, uint instanceIndex, uint startVertex,
                                    uint startInstance) {
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

/// The fragment's own side against what its draw keeps. The view flips z
/// (+z forward), which turns the winding round once more.
public bool visibilityCulled(VertexOut input, bool front) {
    return (input.cullBack == 1 && front) || (input.cullBack == 2 && !front);
}

/// The pixel (bottom row first) a fragment at SV_Position covers.
public uint2 visibilityPixel(CameraParams camera, float4 position) {
    return uint2(uint(position.x), camera.height - 1 - uint(position.y));
}
