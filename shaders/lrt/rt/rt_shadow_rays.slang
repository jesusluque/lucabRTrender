// Copyright (c) 2026 lucabRTrender contributors.
//
// The shadow query for a device that traces only in a pipeline.
//
// CUDA has OptiX and no inline rays -- Slang offers no `RayQuery` for that
// target at all -- so the candidate loop of rt_shadow_kernel.slang becomes an
// **any hit** program: every proxy the traversal meets is offered here, the
// particle is evaluated at its peak, its `1 - alpha` multiplied into the
// payload, and the hit ignored so the traversal carries on. The ray ends
// early by accepting a hit once what is left is under the cut, which is the
// same stopping rule the inline route's `query.Abort()` is.
//
// The payload carries the ring that collapses a proxy offered twice (entry
// face and exit face: a shadow ray culls neither, so that a ray born inside a
// proxy still takes the particle it stands in). Eight entries here rather
// than sixteen, because a payload is registers -- the test counts what the
// ring caught, so a ring too short shows as a transmittance that is too dark
// rather than as nothing at all.
module rt_shadow_rays;
import rt_common;
import rt_shadow;

public static const uint kRayRing = 8;

public struct ShadowPayload {
    public float transmittance;
    /// The ray's own far end. An any hit knows where the current candidate
    /// is (`RayTCurrent`) and where the ray started (`RayTMin`), but not
    /// where it was told to stop, and a particle peaking past the light is
    /// not between the surface and the light.
    public float tMax;
    public uint  particles;
    public uint  duplicates;
    public uint  ringCount;
    public uint  ringAt;
    public uint  ringId[kRayRing];
    public uint  ringInstance[kRayRing];
};

public StructuredBuffer<float4>   frames;
public StructuredBuffer<float4>   colours;
public StructuredBuffer<float4>   instanceData;
public StructuredBuffer<uint>     instanceIndices;
public StructuredBuffer<float4>   rays;
public RWStructuredBuffer<float>  shadow;
public RWStructuredBuffer<uint>   counters;
public ConstantBuffer<ShadowParams> shadowParams;

uniform RaytracingAccelerationStructure scene;

[shader("miss")]
void rtShadowMiss(inout ShadowPayload payload) {
    // Nothing to do: a ray that reaches the light unimpeded keeps what it has.
}

[shader("anyhit")]
void rtShadowAnyHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    const uint instance = InstanceIndex();
    const uint id = instanceIndices[instance * 2] + PrimitiveIndex() / kIcoTriangles;
    const float4 w0 = instanceData[instance * 3];
    const float4 w1 = instanceData[instance * 3 + 1];
    const float4 w2 = instanceData[instance * 3 + 2];
    const float3x3 toCloud = float3x3(w0.xyz, w1.xyz, w2.xyz);
    const float3 origin = mul(toCloud, WorldRayOrigin()) + float3(w0.w, w1.w, w2.w);
    const float3 direction = mul(toCloud, WorldRayDirection());
    const float opacity = colours[id + instanceIndices[instance * 2 + 1]].w;
    float tPeak;
    const float alpha = shadowAlpha(frames[id * 4], frames[id * 4 + 1], frames[id * 4 + 2], opacity, origin,
                                    direction, RayTMin(), payload.tMax, shadowParams.nearZ, shadowParams.farZ,
                                    tPeak);
    if (alpha <= 0.0) {
        IgnoreHit();
        return;
    }
    // The ring: a proxy offers two faces, and the particle is one.
    for (uint k = 0; k < payload.ringCount; ++k) {
        if (payload.ringId[k] == id && payload.ringInstance[k] == instance) {
            payload.duplicates += 1;
            IgnoreHit();
            return;
        }
    }
    payload.ringId[payload.ringAt] = id;
    payload.ringInstance[payload.ringAt] = instance;
    payload.ringAt = (payload.ringAt + 1) % kRayRing;
    payload.ringCount = min(payload.ringCount + 1, kRayRing);
    payload.transmittance *= 1.0 - alpha;
    payload.particles += 1;
    if (payload.transmittance < shadowParams.cut) {
        // Nothing left to gather: stop the ray where it is.
        AcceptHitAndEndSearch();
        return;
    }
    IgnoreHit();
}

[shader("raygeneration")]
void rtShadowRaysGen() {
    const uint r = DispatchRaysIndex().x;
    if (r >= shadowParams.rays) {
        return;
    }
    const float4 a = rays[r * 2];
    const float4 b = rays[r * 2 + 1];
    RayDesc ray;
    ray.Origin = a.xyz;
    ray.TMin = a.w;
    ray.Direction = b.xyz;
    ray.TMax = b.w;
    ShadowPayload payload;
    payload.transmittance = 1.0;
    payload.tMax = ray.TMax;
    payload.particles = 0;
    payload.duplicates = 0;
    payload.ringCount = 0;
    payload.ringAt = 0;
    for (uint k = 0; k < kRayRing; ++k) {
        payload.ringId[k] = 0xFFFFFFFFu;
        payload.ringInstance[k] = 0xFFFFFFFFu;
    }
    // Back faces culled only where the caller asks: a shadow ray wants them,
    // so that a ray born inside a proxy still takes the particle it stands
    // in, and the test's control turns them off to show what that is worth.
    TraceRay(scene, shadowParams.cull != 0 ? RAY_FLAG_CULL_BACK_FACING_TRIANGLES : RAY_FLAG_NONE, 0xFF, 0, 1, 0,
             ray, payload);
    shadow[r] = payload.transmittance;
    if (payload.particles > 0) {
        InterlockedAdd(counters[0], payload.particles);
    }
    if (payload.duplicates > 0) {
        InterlockedAdd(counters[1], payload.duplicates);
    }
    if (payload.transmittance < shadowParams.cut) {
        InterlockedAdd(counters[2], 1u);
    }
}
