// Copyright (c) 2026 lucabRTrender contributors.
//
// The hardware route: every particle a stretched icosahedron in a BVH the
// device builds and traverses (3DGRT's proxies), entries offered to the
// Integrator (rt_integrate.slang, where the algorithm and its costs are).
module rt_render;
import rt_common;
__exported import rt_integrate;

uniform RaytracingAccelerationStructure scene;

/// Twenty triangles a proxy, in particle order within a chunk.
struct ProxyTriangles : IPrimitives {
    static uint particleOf(uint raw) { return raw / kIcoTriangles; }
};

/// What the cloud does to one ray over [ray.TMin, ray.TMax]: the radiance it
/// gathers and what is left through it. The camera's kernel below is one
/// caller; a secondary ray -- a reflection, a ray between two surfaces -- is
/// the other, and asks for its own slice, which is what `beginRange` clips
/// the peaks to.
public SegmentResult rtTraceSegment(LrtRay ray, float windowMin, float windowMax) {
    Integrator<ProxyTriangles> integrator;
    Cursor cursor;
    integrator.beginRange(cursor, windowMin, windowMax);
    for (uint segment = 0; segment < params.maxSegments; ++segment) {
        integrator.beginSegment(cursor);
        // Back faces culled: a proxy counts where the ray enters it.
        RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
        RayDesc traced;
        traced.Origin = ray.Origin;
        traced.Direction = ray.Direction;
        traced.TMin = ray.TMin;
        traced.TMax = ray.TMax;
        query.TraceRayInline(scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, traced);
        while (query.Proceed()) {
            // Primitive and instance index: the id waits for the record's end.
            if (integrator.offer(cursor, query.CandidateTriangleRayT(), query.CandidatePrimitiveIndex(),
                                 query.CandidateInstanceIndex()) == Offer.Dropped) {
                // Nothing past a dropped entry is wanted this segment.
                query.CommitNonOpaqueTriangleHit();
            }
        }
        if (!integrator.endSegment(cursor, ray, segment + 1 == params.maxSegments)) {
            break;
        }
    }
    return integrator.result();
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rtRender(uint3 tid: SV_DispatchThreadID) {
    const uint px = tid.x;
    const uint py = tid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    LrtRay ray = primaryRay(px, py);
    Integrator<ProxyTriangles> integrator;
    Cursor cursor;
    integrator.begin(cursor);
    for (uint segment = 0; segment < params.maxSegments; ++segment) {
        integrator.beginSegment(cursor);
        RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
        RayDesc traced;
        traced.Origin = ray.Origin;
        traced.Direction = ray.Direction;
        traced.TMin = ray.TMin;
        traced.TMax = ray.TMax;
        query.TraceRayInline(scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, traced);
        while (query.Proceed()) {
            if (integrator.offer(cursor, query.CandidateTriangleRayT(), query.CandidatePrimitiveIndex(),
                                 query.CandidateInstanceIndex()) == Offer.Dropped) {
                query.CommitNonOpaqueTriangleHit();
            }
        }
        if (!integrator.endSegment(cursor, ray, segment + 1 == params.maxSegments)) {
            break;
        }
    }
    integrator.write(px, py);
}

/// The same frame, drawn as two queries composed: [nearZ, splitAt] over
/// [splitAt, farZ]. What a secondary ray will do with a slice of a ray, on
/// the frame the camera draws, so a test can hold the two to each other.
[shader("compute")]
[numthreads(16, 16, 1)]
void rtRenderSplit(uint3 tid: SV_DispatchThreadID) {
    const uint px = tid.x;
    const uint py = tid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    const LrtRay ray = primaryRay(px, py);
    // The near query may stop traversing at the cut -- a peak is never before
    // the proxy's entry -- but the far one must walk from the ray's start: a
    // proxy entered before the cut can peak after it.
    LrtRay near = ray;
    near.TMax = min(ray.TMax, params.splitAt);
    writeSegment(px, py, segmentOver(rtTraceSegment(near, ray.TMin, params.splitAt),
                                     rtTraceSegment(ray, params.splitAt, ray.TMax)));
}
