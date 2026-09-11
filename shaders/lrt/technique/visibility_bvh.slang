// Copyright (c) 2026 lucabRTrender contributors.
//
// Visibility by rays through the scene's compute BVHs, for a device with no
// ray tracing hardware: a top level over the instances in view space, a
// bottom level per mesh in object space, both Karras LBVHs (bvh_common), both
// walked with a stack. The nearest triangle is written as the rasteriser
// writes it.
// The scene buffers are surface.slang's, so a program that both traverses and
// shades -- the cutout pass -- has one of each, bound once by name.
module visibility_bvh;
__exported import lrt.technique.surface;
import lrt.rt.bvh_common;

public struct SceneBvhParams {
    public uint instances;
    public uint pad0; public uint pad1; public uint pad2;
};

public StructuredBuffer<float4>          topBoxes;       // 2 per internal node
public StructuredBuffer<uint>            topChildren;
public StructuredBuffer<uint>            topLeaves;      // leaf -> instance
public StructuredBuffer<float4>          meshBoxes;      // every mesh's nodes, at its nodeBase
public StructuredBuffer<uint>            meshChildren;
public StructuredBuffer<uint>            meshLeaves;     // at each mesh's firstTriangle: leaf -> triangle
public RWTexture2D<uint4>                ids;
public ConstantBuffer<CameraParams>      camera;
public ConstantBuffer<SceneBvhParams>    scene;

static const uint kStack = 64;

bool boxHit(float3 lo, float3 hi, float3 origin, float3 inverse, float tMin, float tMax) {
    const float3 t0 = (lo - origin) * inverse;
    const float3 t1 = (hi - origin) * inverse;
    const float enter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), tMin));
    const float leave = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), tMax));
    return enter <= leave;
}

public struct Best {
    public float t;
    public uint  instance;
    public uint  triangle;

    /// What the rasteriser would have written for this hit.
    public uint4 seen() { return instance != 0xFFFFFFFF ? uint4(instance + 1, triangle, 0, 0) : uint4(0); }
};

/// Moller-Trumbore, t in (tMin, best.t).
void triangleHit(inout Best best, float3 origin, float3 direction, float tMin, MeshRecord m, uint instance,
                 uint triangle) {
    const uint base = (m.firstTriangle + triangle) * 3;
    const float3 a = positions[m.firstPoint + indices[base]].xyz;
    const float3 e1 = positions[m.firstPoint + indices[base + 1]].xyz - a;
    const float3 e2 = positions[m.firstPoint + indices[base + 2]].xyz - a;
    const float3 p = cross(direction, e2);
    const float det = dot(e1, p);
    if (abs(det) < 1e-30) {
        return;
    }
    // Single sided: the front only, counter-clockwise, judged in object space.
    // det = -dot(direction, cross(e1, e2)): negative where the triangle's
    // counter-clockwise normal points along the ray, away from its origin.
    if ((instances[instance].flags & 1u) == 0 && det < 0.0) {
        return;
    }
    const float inverse = 1.0 / det;
    const float3 s = origin - a;
    const float u = dot(s, p) * inverse;
    if (u < 0.0 || u > 1.0) {
        return;
    }
    const float3 q = cross(s, e1);
    const float v = dot(direction, q) * inverse;
    if (v < 0.0 || u + v > 1.0) {
        return;
    }
    const float t = dot(e2, q) * inverse;
    if (t > tMin && t < best.t) {
        best.t = t;
        best.instance = instance;
        best.triangle = triangle;
    }
}

void meshTraverse(inout Best best, float3 viewOrigin, float3 viewDirection, float tMin, uint instance) {
    const InstanceRecord r = instances[instance];
    const MeshRecord m = meshes[r.mesh];
    // Into object space: the inverse of view's 3x3 is its normal matrix transposed.
    const float3x3 inverseLinear = transpose(float3x3(r.normal0.xyz, r.normal1.xyz, r.normal2.xyz));
    const float3 origin = mul(inverseLinear, viewOrigin - float3(r.view0.w, r.view1.w, r.view2.w));
    const float3 direction = mul(inverseLinear, viewDirection);
    const float3 inverse = 1.0 / direction;
    if (m.triangles == 1) {
        triangleHit(best, origin, direction, tMin, m, instance, 0);
        return;
    }
    uint stack[kStack];
    uint depth = 0;
    stack[depth++] = 0;
    while (depth > 0) {
        const uint node = m.nodeBase + stack[--depth];
        if (!boxHit(meshBoxes[node * 2].xyz, meshBoxes[node * 2 + 1].xyz, origin, inverse, tMin, best.t)) {
            continue;
        }
        for (uint k = 0; k < 2; ++k) {
            const uint child = meshChildren[node * 2 + k];
            if ((child & kLeafBit) != 0) {
                triangleHit(best, origin, direction, tMin, m, instance,
                            meshLeaves[m.firstTriangle + (child & ~kLeafBit)]);
            } else if (depth < kStack) {
                stack[depth++] = child;
            }
        }
    }
}

/// The nearest triangle beyond `tMin` along a view space ray.
public Best traverseNearest(float3 origin, float3 direction, float tMin) {
    Best best;
    best.t = 3.0e38;
    best.instance = 0xFFFFFFFF;
    best.triangle = 0;
    if (scene.instances == 1) {
        meshTraverse(best, origin, direction, tMin, 0);
        return best;
    }
    if (scene.instances == 0) {
        return best;
    }
    const float3 inverse = 1.0 / direction;
    uint stack[kStack];
    uint depth = 0;
    stack[depth++] = 0;
    while (depth > 0) {
        const uint node = stack[--depth];
        if (!boxHit(topBoxes[node * 2].xyz, topBoxes[node * 2 + 1].xyz, origin, inverse, tMin, best.t)) {
            continue;
        }
        for (uint k = 0; k < 2; ++k) {
            const uint child = topChildren[node * 2 + k];
            if ((child & kLeafBit) != 0) {
                meshTraverse(best, origin, direction, tMin, topLeaves[child & ~kLeafBit]);
            } else if (depth < kStack) {
                stack[depth++] = child;
            }
        }
    }
    return best;
}

/// Where a pixel's ray starts: the rasteriser's near plane.
public float bvhTMin() {
    return camera.orthographic != 0 ? 0.0 : camera.nearZ;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void visibilityBvh(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    float3 origin;
    float3 direction;
    viewRay(camera, float2(tid.xy) + 0.5, origin, direction);
    const Best best = traverseNearest(origin, direction, bvhTMin());
    ids[int2(int(tid.x), int(camera.height - 1 - tid.y))] = best.seen();
}
