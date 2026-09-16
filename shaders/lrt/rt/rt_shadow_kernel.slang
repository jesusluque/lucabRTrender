// Copyright (c) 2026 lucabRTrender contributors.
//
// The shadow query as a kernel of its own: rays in, transmittance out. What
// asks for it today is the test that holds rt_shadow.slang to its closed
// forms; the same bindings are what a caller inside another kernel needs
// (splat_project.slang relights a splat behind this).
module rt_shadow_kernel;
import rt_common;
import rt_shadow;

// Rays come as two float4 each -- origin and tMin, direction and tMax -- so
// that whoever asks (a test, and the path tracer's shadow rays) writes them
// the same way. The transmittance of each goes back per ray; the counters are
// the frame's totals, for a test to read.

public StructuredBuffer<float4>   frames;          // per particle, four rows
public StructuredBuffer<float4>   colours;         // per instance particle: rt_shade.slang
public StructuredBuffer<float4>   instanceData;    // per instance: world->cloud rows 0..2
public StructuredBuffer<uint>     instanceIndices; // per instance: id base, colour offset
public StructuredBuffer<float4>   rays;            // two per ray
public RWStructuredBuffer<float>  shadow;          // one per ray
public RWStructuredBuffer<uint>   counters;        // [particles, duplicates, rays cut]
public ConstantBuffer<ShadowParams> shadowParams;

uniform RaytracingAccelerationStructure scene;

[shader("compute")]
[numthreads(64, 1, 1)]
void rtShadowRays(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= shadowParams.rays) {
        return;
    }
    const float4 a = rays[r * 2];
    const float4 b = rays[r * 2 + 1];
    LrtRay ray;
    ray.Origin = a.xyz;
    ray.TMin = a.w;
    ray.Direction = b.xyz;
    ray.TMax = b.w;

    ShadowState state = shadowBegin();
    RayDesc traced;
    traced.Origin = ray.Origin;
    traced.Direction = ray.Direction;
    traced.TMin = ray.TMin;
    traced.TMax = ray.TMax;
    // The template flags are what every ray may do; the call narrows them.
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, shadowParams.cull != 0 ? RAY_FLAG_CULL_BACK_FACING_TRIANGLES : RAY_FLAG_NONE,
                         0xFF, traced);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }
        const uint instance = query.CandidateInstanceIndex();
        const uint id = instanceIndices[instance * 2] + query.CandidatePrimitiveIndex() / kIcoTriangles;
        const float4 w0 = instanceData[instance * 3];
        const float4 w1 = instanceData[instance * 3 + 1];
        const float4 w2 = instanceData[instance * 3 + 2];
        const float3x3 toCloud = float3x3(w0.xyz, w1.xyz, w2.xyz);
        const float3 origin = mul(toCloud, ray.Origin) + float3(w0.w, w1.w, w2.w);
        const float3 direction = mul(toCloud, ray.Direction);
        const float opacity = colours[id + instanceIndices[instance * 2 + 1]].w;
        float tPeak;
        const float alpha = shadowAlpha(frames[id * 4], frames[id * 4 + 1], frames[id * 4 + 2], opacity, origin,
                                        direction, ray.TMin, ray.TMax, shadowParams.nearZ, shadowParams.farZ, tPeak);
        if (alpha > 0.0) {
            shadowTake(state, id, instance, alpha);
            if (shadowDone(state, shadowParams.cut)) {
                query.Abort();
                break;
            }
        }
    }
    shadow[r] = state.transmittance;
    if (state.particles > 0) {
        InterlockedAdd(counters[0], state.particles);
    }
    if (state.duplicates > 0) {
        InterlockedAdd(counters[1], state.duplicates);
    }
    if (shadowDone(state, shadowParams.cut)) {
        InterlockedAdd(counters[2], 1u);
    }
}
