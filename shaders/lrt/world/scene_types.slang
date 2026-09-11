// Copyright (c) 2026 lucabRTrender contributors.
//
// The scene as kernels see it (lrt::world::GpuScene): meshes packed into
// shared pools, and one record per instance. Raster and ray traced visibility,
// shading and AOVs all read these.
module scene_types;

/// Where a mesh's data sits in the pools.
public struct MeshRecord {
    public uint firstPoint;      // positions, normals
    public uint points;
    public uint firstTriangle;   // indices (3 per triangle), triangle corners, triangle faces
    public uint triangles;
    public uint hasNormals;      // 0: shade with the geometric normal
    public uint nodeBase;        // compute BVH: the mesh's first internal node
    public uint slotBase;        // primvars: the mesh's row in the slot table
    public uint pad2;
    public float4 boundsLo;      // object space
    public float4 boundsHi;
};

/// One drawn copy of a mesh. Rows of 3x4 matrices; `view` is object to view
/// (+z forward), `normal` its inverse transpose.
public struct InstanceRecord {
    public float4 view0;   public float4 view1;   public float4 view2;
    public float4 normal0; public float4 normal1; public float4 normal2;
    public float4 world0;  public float4 world1;  public float4 world2;
    public float4 colour;       // displayColor rgb, displayOpacity
    public uint   mesh;
    public uint   primId;
    public uint   instanceId;
    public uint   flags;        // bit 0: double sided
};

public float3 rowsApply(float4 r0, float4 r1, float4 r2, float3 x, float w) {
    return float3(dot(r0.xyz, x) + r0.w * w, dot(r1.xyz, x) + r1.w * w, dot(r2.xyz, x) + r2.w * w);
}
