// Copyright (c) 2026 lucabRTrender contributors.
//
// What a secondary ray sees of a cloud when all it needs is how much light
// gets through: transmittance, and nothing else.
//
// A shadow ray does not draw, so none of what rt_integrate.slang carries is
// needed here -- no k-buffer, no segments, no order. Transmittance is a
// product of (1 - alpha) and a product does not care in what order its terms
// arrive, so every proxy the traversal offers is accounted for as it comes
// and the ray stops as soon as the product falls under `cut`. That is the
// cheapest thing a ray can do against this geometry, and it is what makes
// splats able to shadow the meshes around them at interactive rates.
//
// What is approximate, said plainly:
//   - a particle is taken at its maximum response along the ray, as the
//     primary route takes it, not integrated along the segment it covers;
//   - a particle whose peak lies behind the ray's origin is not taken at
//     all (the origin is on a surface, and the light is ahead of it);
//   - a ray born *inside* a proxy still takes that particle: back faces are
//     not culled here, so a proxy offers its exit face as well as its entry
//     face and the ring below collapses the two into one particle. The
//     primary ray culls back faces (it starts at the camera, outside), a
//     secondary one starts on a surface that may be surrounded by proxies,
//     and missing those is what makes a splat's own shadow stop at the
//     geometry it touches. `cull` switches it back on, which is the control
//     the test measures.
//
// Duplicates: two faces of one proxy can be offered in one traversal (an
// edge crossing, a degenerate stretch). rt_integrate removes them after
// sorting, by comparing with the last particle blended; here there is no
// order, so a short ring of the particles already taken is kept and each
// candidate is looked up in it. `ShadowState.duplicates` counts what it
// caught, which is what a test reads back.
module rt_shadow;
import rt_common;

/// Particles remembered to catch a proxy offered twice.
public static const uint kShadowRing = 16;

public struct ShadowParams {
    public uint  rays;        // how many the buffer holds
    public float cut;         // stop once transmittance falls under this
    public float nearZ;       // a peak nearer than this is not taken
    public float farZ;
    public uint  cull;        // 1: cull back faces, as the primary ray does (the test's control)
    public uint  pad1; public uint pad2; public uint pad3;
};

public struct ShadowState {
    public float transmittance;
    public uint  particles;    // taken, after every cutoff
    public uint  duplicates;   // offered again and recognised
    uint  ringId[kShadowRing];
    uint  ringInstance[kShadowRing];
    uint  ringCount;
    uint  ringAt;
};

public ShadowState shadowBegin() {
    ShadowState s;
    s.transmittance = 1.0;
    s.particles = 0;
    s.duplicates = 0;
    s.ringCount = 0;
    s.ringAt = 0;
    for (uint k = 0; k < kShadowRing; ++k) {
        s.ringId[k] = 0xFFFFFFFFu;
        s.ringInstance[k] = 0xFFFFFFFFu;
    }
    return s;
}

/// Whether the ray is done: what is left through it is under the cut.
public bool shadowDone(ShadowState s, float cut) {
    return s.transmittance < cut;
}

/// The response of one particle along one ray, at the point where it peaks.
/// `frame` is the particle's four rows (rt_frames.slang), `opacity` what the
/// instance draws it with. Returns 0 where the particle is cut away, and
/// `tPeak` where it peaks.
public float shadowAlpha(float4 f0, float4 f1, float4 f2, float opacity, float3 origin, float3 direction,
                         float tMin, float tMax, float nearZ, float farZ, out float tPeak) {
    tPeak = 0.0;
    if (!(opacity >= 1.0 / 255.0)) {
        return 0.0;
    }
    // Into the particle's frame, where it is the unit sphere.
    const float3x3 m = float3x3(f0.xyz, f1.xyz, f2.xyz);
    const float3 o = mul(m, origin - float3(f0.w, f1.w, f2.w));
    const float3 d = mul(m, direction);
    const float t = -dot(o, d) / max(dot(d, d), 1e-30);
    tPeak = t;
    if (t < tMin || t > tMax || t < nearZ || t > farZ) {
        return 0.0;
    }
    const float3 closest = o + t * d;
    const float power = dot(closest, closest);
    if (power > rtCutoff(opacity)) {
        return 0.0;
    }
    const float alpha = min(0.99, opacity * exp(-0.5 * power));
    return alpha < 1.0 / 255.0 ? 0.0 : alpha;
}

/// Accounts for one particle, unless the ring says it was already taken.
/// Order does not matter: transmittance is a product.
[mutating]
public void shadowTake(inout ShadowState s, uint id, uint instance, float alpha) {
    for (uint k = 0; k < s.ringCount; ++k) {
        if (s.ringId[k] == id && s.ringInstance[k] == instance) {
            s.duplicates += 1;
            return;
        }
    }
    s.ringId[s.ringAt] = id;
    s.ringInstance[s.ringAt] = instance;
    s.ringAt = (s.ringAt + 1) % kShadowRing;
    s.ringCount = min(s.ringCount + 1, kShadowRing);
    s.transmittance *= 1.0 - alpha;
    s.particles += 1;
}

// ---- the hardware route's kernel ---------------------------------------------
//
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
