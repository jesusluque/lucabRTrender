// Copyright (c) 2026 lucabRTrender contributors.
//
// The splat shadow query with its tables in one buffer.
//
// rt_shadow_kernel.slang binds the cloud's four tables separately, which is
// what a kernel of its own can afford. A kernel that already binds a scene,
// its materials, its textures and its lights cannot: the path tracer sits at
// Metal's limit of 31 buffers, and four more would not fit. So the tables are
// packed into one `float4` buffer before the frame -- by `rtShadowPack`, on
// the device -- and read back out here through the offsets a `PackedShadow`
// carries.
//
// The two uint tables travel as float4 lanes: `asfloat` going in, `asuint`
// coming out, which is a reinterpretation and not a conversion.
module rt_shadow_packed;
// LrtRay is in this module's public signature, so whoever imports this one
// has it too: Slang's import is not transitive.
__exported import rt_common;
import rt_shadow;

/// Where each table starts, in float4 entries.
public struct PackedShadow {
    public uint frames;      ///< four rows a particle
    public uint colours;     ///< one a particle, per instance's colour offset
    public uint instances;   ///< three rows an instance: world to cloud
    public uint indices;     ///< one an instance: id base in .x, colour offset in .y
    public uint instanceCount;
    public uint pad0; public uint pad1; public uint pad2;
};

/// The transmittance of one ray through the packed cloud, between `tMin` and
/// `tMax`: the product of what each particle it meets lets through, stopping
/// once there is less left than `cut`. The same loop as rt_shadow_kernel's,
/// over the same `shadowAlpha` and `shadowTake`, reading the packed tables.
public float packedShadowTransmittance(StructuredBuffer<float4> packed, PackedShadow where,
                                       RaytracingAccelerationStructure cloud, LrtRay ray, float cut,
                                       float nearZ, float farZ) {
    ShadowState state = shadowBegin();
    RayDesc traced;
    traced.Origin = ray.Origin;
    traced.Direction = ray.Direction;
    traced.TMin = ray.TMin;
    traced.TMax = ray.TMax;
    // No culling: a ray born inside a proxy still takes the particle it
    // stands in, which is rt_shadow_kernel's rule and the test's.
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(cloud, RAY_FLAG_NONE, 0xFF, traced);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }
        const uint instance = query.CandidateInstanceIndex();
        if (instance >= where.instanceCount) {
            continue;
        }
        const float4 index = packed[where.indices + instance];
        const uint id = asuint(index.x) + query.CandidatePrimitiveIndex() / kIcoTriangles;
        const float4 w0 = packed[where.instances + instance * 3];
        const float4 w1 = packed[where.instances + instance * 3 + 1];
        const float4 w2 = packed[where.instances + instance * 3 + 2];
        const float3x3 toCloud = float3x3(w0.xyz, w1.xyz, w2.xyz);
        const float3 origin = mul(toCloud, ray.Origin) + float3(w0.w, w1.w, w2.w);
        const float3 direction = mul(toCloud, ray.Direction);
        const float opacity = packed[where.colours + id + asuint(index.y)].w;
        float tPeak;
        const float alpha = shadowAlpha(packed[where.frames + id * 4], packed[where.frames + id * 4 + 1],
                                        packed[where.frames + id * 4 + 2], opacity, origin, direction, ray.TMin,
                                        ray.TMax, nearZ, farZ, tPeak);
        if (alpha > 0.0) {
            shadowTake(state, id, instance, alpha);
            if (shadowDone(state, cut)) {
                query.Abort();
                break;
            }
        }
    }
    return state.transmittance;
}
