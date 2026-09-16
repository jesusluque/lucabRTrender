// Copyright (c) 2026 lucabRTrender contributors.
//
// Mesh visibility for a device that traces only in a pipeline.
//
// CUDA has OptiX and no inline rays -- Slang offers no RayQuery for that
// target at all ("uses features that are not available in '_raygen' stage for
// 'cuda'") -- so there the ray is traced the classic way: a ray generation
// program, a payload, a miss and a closest hit. The pixel's work either side
// of the trace is visibility_trace.slang's, unchanged.
//
// A module of its own because a module compiles all of its entry points:
// with these beside the compute kernel, loading it on Metal fails on the
// closest hit, which Metal has no stage for (measured, and it took the
// whole visibility pass down with it).
module visibility_trace_rays;
import lrt.technique.visibility_trace_common;

struct VisibilityPayload {
    uint4 seen;
    float t;
};

[shader("miss")]
void visibilityMiss(inout VisibilityPayload payload) {
    payload.seen = uint4(0);
    payload.t = 0.0;
}

[shader("closesthit")]
void visibilityHit(inout VisibilityPayload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    payload.seen = uint4(InstanceID() + 1, PrimitiveIndex(), 0, 0);
    payload.t = RayTCurrent();
}

public TraceHit traceNearestPipeline(float3 origin, float3 direction, float tMin) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = 3.0e38;
    VisibilityPayload payload;
    payload.seen = uint4(0);
    payload.t = 0.0;
    TraceRay(scene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 1, 0, ray, payload);
    TraceHit hit;
    hit.seen = payload.seen;
    hit.t = payload.t;
    return hit;
}

[shader("raygeneration")]
void visibilityTraceGen() {
    const uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= camera.width || pixel.y >= camera.height) {
        return;
    }
    float3 origin;
    float3 direction;
    traceRay(pixel, origin, direction);
    const TraceHit hit = traceNearestPipeline(origin, direction, traceTMin());
    ids[int2(int(pixel.x), int(camera.height - 1 - pixel.y))] = hit.seen;
}
