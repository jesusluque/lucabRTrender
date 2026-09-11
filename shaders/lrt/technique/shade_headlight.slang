// Copyright (c) 2026 lucabRTrender contributors.
//
// A visibility buffer into an image, lit from the eye. For every pixel the
// primary ray is intersected with the triangle the pixel saw (Moller-Trumbore
// in view space), so position, depth and barycentrics are exact whichever pass
// found the triangle; the normal is the mesh's smooth normal where it has one
// and the triangle's own otherwise, turned to face the eye; the colour is
// displayColor times the cosine to the eye. Depth is view z (0: nothing), as
// every engine image carries it.
module shade_headlight;
import camera;
import lrt.world.scene_types;

StructuredBuffer<float4>          positions;
StructuredBuffer<float4>          normals;
StructuredBuffer<uint>            indices;
StructuredBuffer<MeshRecord>      meshes;
StructuredBuffer<InstanceRecord>  instances;
Texture2D<uint4>                  visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>        colour;
RWStructuredBuffer<float>         depth;
ConstantBuffer<CameraParams>      camera;

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

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeHeadlight(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        colour[at] = float4(0.0);
        depth[at] = 0.0;
        return;
    }
    const InstanceRecord instance = instances[seen.x - 1];
    const MeshRecord mesh = meshes[instance.mesh];
    const uint triangle = mesh.firstTriangle + seen.y;
    const uint i0 = mesh.firstPoint + indices[triangle * 3];
    const uint i1 = mesh.firstPoint + indices[triangle * 3 + 1];
    const uint i2 = mesh.firstPoint + indices[triangle * 3 + 2];
    const float3 a = rowsApply(instance.view0, instance.view1, instance.view2, positions[i0].xyz, 1.0);
    const float3 b = rowsApply(instance.view0, instance.view1, instance.view2, positions[i1].xyz, 1.0);
    const float3 c = rowsApply(instance.view0, instance.view1, instance.view2, positions[i2].xyz, 1.0);
    float3 origin;
    float3 direction;
    viewRay(camera, float2(tid.xy) + 0.5, origin, direction);
    const Hit hit = intersect(origin, direction, a, b, c);
    const float w0 = 1.0 - hit.barycentrics.x - hit.barycentrics.y;

    float3 n = normalize(cross(b - a, c - a));
    if (mesh.hasNormals != 0) {
        const float3 objectNormal = normals[i0].xyz * w0 + normals[i1].xyz * hit.barycentrics.x +
                                    normals[i2].xyz * hit.barycentrics.y;
        const float3 viewNormal = rowsApply(instance.normal0, instance.normal1, instance.normal2, objectNormal, 0.0);
        const float length2 = dot(viewNormal, viewNormal);
        if (length2 > 0.0) {
            n = viewNormal / sqrt(length2);
        }
    }
    const float3 toEye = -normalize(direction);
    const float cosine = abs(dot(n, toEye));
    const float opacity = instance.colour.w;
    colour[at] = float4(instance.colour.rgb * cosine * opacity, opacity);
    depth[at] = camera.orthographic != 0 ? origin.z + hit.t : hit.t;
}
