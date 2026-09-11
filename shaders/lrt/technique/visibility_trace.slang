// Copyright (c) 2026 lucabRTrender contributors.
//
// Visibility by rays: the pixel centre's primary ray against the scene's
// hardware acceleration structure, the nearest triangle written as the
// rasteriser writes it -- (instance + 1, triangle), row 0 on top -- so
// shading cannot tell the two passes apart. The ray and the traversal are
// public: the cutout pass, generated so it can call into the materials,
// walks the same ray past the samples their opacity removes.
module visibility_trace;
__exported import camera;

public struct TraceParams {
    // view to world, rows of a 3x4
    public float v00; public float v01; public float v02; public float v03;
    public float v10; public float v11; public float v12; public float v13;
    public float v20; public float v21; public float v22; public float v23;
};

public uniform RaytracingAccelerationStructure scene;
public RWTexture2D<uint4>           ids;
public ConstantBuffer<CameraParams> camera;
public ConstantBuffer<TraceParams>  trace;

public struct TraceHit {
    public uint4 seen;   // (instance + 1, triangle, 0, 0); 0 for nothing
    public float t;      // view depth: the ray's direction has view z = 1
};

/// The pixel's primary ray in world space.
public void traceRay(uint2 pixel, out float3 origin, out float3 direction) {
    float3 viewOrigin;
    float3 viewDirection;
    viewRay(camera, float2(pixel) + 0.5, viewOrigin, viewDirection);
    const float3x4 toWorld = float3x4(trace.v00, trace.v01, trace.v02, trace.v03, trace.v10, trace.v11, trace.v12,
                                      trace.v13, trace.v20, trace.v21, trace.v22, trace.v23);
    origin = mul(toWorld, float4(viewOrigin, 1.0));
    direction = mul(toWorld, float4(viewDirection, 0.0));
}

/// The nearest triangle beyond `tMin`. Back faces are culled where an
/// instance is single sided (its descriptor disables culling where it is
/// not): Hydra's rule, the rasteriser's. There is no far plane, as the
/// rasteriser's reversed infinite depth has none.
public TraceHit traceNearest(float3 origin, float3 direction, float tMin) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = 3.0e38;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
    query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, ray);
    query.Proceed();
    TraceHit hit;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        hit.seen = uint4(query.CommittedInstanceID() + 1, query.CommittedPrimitiveIndex(), 0, 0);
        hit.t = query.CommittedRayT();
    } else {
        hit.seen = uint4(0);
        hit.t = 0.0;
    }
    return hit;
}

/// Where a pixel's ray starts: the rasteriser's near plane.
public float traceTMin() {
    return camera.orthographic != 0 ? 0.0 : camera.nearZ;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void visibilityTrace(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    float3 origin;
    float3 direction;
    traceRay(tid.xy, origin, direction);
    const TraceHit hit = traceNearest(origin, direction, traceTMin());
    ids[int2(int(tid.x), int(camera.height - 1 - tid.y))] = hit.seen;
}
