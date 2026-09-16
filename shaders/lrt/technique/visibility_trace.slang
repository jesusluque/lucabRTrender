// Copyright (c) 2026 lucabRTrender contributors.
//
// Visibility by rays, traced inline: the pixel centre's primary ray against
// the scene's acceleration structure, the nearest triangle written as the
// rasteriser writes it -- (instance + 1, triangle), row 0 on top -- so
// shading cannot tell the two passes apart. The ray and the traversal are
// public: the cutout pass, generated so it can call into the materials,
// walks the same ray past the samples their opacity removes.
module visibility_trace;
__exported import visibility_trace_common;

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

void visibilityAt(uint2 pixel) {
    if (pixel.x >= camera.width || pixel.y >= camera.height) {
        return;
    }
    float3 origin;
    float3 direction;
    traceRay(pixel, origin, direction);
    const TraceHit hit = traceNearest(origin, direction, traceTMin());
    ids[int2(int(pixel.x), int(camera.height - 1 - pixel.y))] = hit.seen;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void visibilityTrace(uint3 tid: SV_DispatchThreadID) {
    visibilityAt(tid.xy);
}
