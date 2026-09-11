// Copyright (c) 2026 lucabRTrender contributors.
//
// What a pixel's visibility sample means: the triangle it saw, re-intersected
// with the pixel's primary ray in view space (Moller-Trumbore), so position,
// depth and barycentrics are exact whichever pass found the triangle, and the
// shading normal from the mesh's "normals" primvar where it has one. Shading
// and AOVs both start here.
module surface;
__exported import camera;
__exported import lrt.world.scene_types;
__exported import lrt.world.primvars;

public StructuredBuffer<float4>          positions;
public StructuredBuffer<uint>            indices;
public StructuredBuffer<MeshRecord>      meshes;
public StructuredBuffer<InstanceRecord>  instances;

public struct Hit {
    public float  t;
    public float2 barycentrics;   // of vertices 1 and 2
};

/// The ray against a triangle; barycentrics clamped to it, since a pixel's
/// centre can fall just outside the triangle that covers the pixel's sample.
public Hit intersect(float3 origin, float3 direction, float3 a, float3 b, float3 c) {
    Hit hit;
    const float3 e1 = b - a;
    const float3 e2 = c - a;
    const float3 p = cross(direction, e2);
    const float det = dot(e1, p);
    const float inverse = abs(det) > 1e-20 ? 1.0 / det : 0.0;
    const float3 s = origin - a;
    float u = dot(s, p) * inverse;
    const float3 q = cross(s, e1);
    float v = dot(direction, q) * inverse;
    u = saturate(u);
    v = clamp(v, 0.0, 1.0 - u);
    hit.barycentrics = float2(u, v);
    // t from the plane, not the clamped point: exact where the centre is inside.
    hit.t = dot(e2, q) * inverse;
    return hit;
}

public struct Surface {
    public uint           instanceIndex;
    public InstanceRecord instance;
    public MeshRecord     mesh;
    public uint           triangle;       // mesh-local
    public uint3          points;         // mesh-local point indices
    public float3         weights;        // barycentrics (w0, w1, w2)
    public float3         viewPosition;
    public float3         viewDirection;  // the primary ray's, not normalised (view z = 1)
    public float          depth;          // view z
    public float3         geometricNormal;   // view space, unit
    public float3         normal;            // view space, unit: the primvar's, else geometric
};

/// The surface at pixel (px, py) (bottom row first) seen as `seen`.
public Surface surfaceAt(CameraParams camera, uint px, uint py, uint4 seen) {
    Surface s;
    s.instanceIndex = seen.x - 1;
    s.instance = instances[s.instanceIndex];
    s.mesh = meshes[s.instance.mesh];
    s.triangle = seen.y;
    const uint t = s.mesh.firstTriangle + s.triangle;
    s.points = uint3(indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
    const InstanceRecord r = s.instance;
    const float3 a = rowsApply(r.view0, r.view1, r.view2, positions[s.mesh.firstPoint + s.points.x].xyz, 1.0);
    const float3 b = rowsApply(r.view0, r.view1, r.view2, positions[s.mesh.firstPoint + s.points.y].xyz, 1.0);
    const float3 c = rowsApply(r.view0, r.view1, r.view2, positions[s.mesh.firstPoint + s.points.z].xyz, 1.0);
    float3 origin;
    viewRay(camera, float2(px, py) + 0.5, origin, s.viewDirection);
    const Hit hit = intersect(origin, s.viewDirection, a, b, c);
    s.weights = float3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y);
    s.depth = hit.t;
    s.viewPosition = origin + s.viewDirection * hit.t;
    s.geometricNormal = normalize(cross(b - a, c - a));
    s.normal = s.geometricNormal;
    const uint normals = primvarRecordOf(s.mesh, kSlotNormals);
    if (normals != kNoPrimvar) {
        const float3 objectNormal = primvarAt(normals, s.mesh, s.triangle, s.points, s.weights, float4(0.0)).xyz;
        const float3 viewNormal = rowsApply(r.normal0, r.normal1, r.normal2, objectNormal, 0.0);
        const float length2 = dot(viewNormal, viewNormal);
        if (length2 > 0.0) {
            s.normal = viewNormal / sqrt(length2);
        }
    }
    return s;
}

/// A primvar of the surface by slot.
public float4 surfacePrimvar(Surface s, uint slot, float4 fallback) {
    return primvarAt(primvarRecordOf(s.mesh, slot), s.mesh, s.triangle, s.points, s.weights, fallback);
}
