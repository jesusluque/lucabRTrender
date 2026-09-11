// Copyright (c) 2026 lucabRTrender contributors.
//
// The compute route: each instance's LBVH (bvh_common.slang) traversed with a
// stack, each particle entered where the ray meets its cut ellipsoid, entries
// offered to the same Integrator the hardware route uses
// (rt_integrate.slang). For devices with compute and no ray tracing.
module rt_bvh_render;
import rt_common;
import rt_integrate;
import bvh_common;

StructuredBuffer<float4> bvhBoxes;
StructuredBuffer<uint>   bvhChildren;
StructuredBuffer<uint>   bvhLeaves;
StructuredBuffer<uint>   bvhInstances;   // per instance: node base, leaf base, particles

/// A Karras tree over 30-bit codes extended by 32 index bits is at most 62
/// deep, and a depth-first stack holds at most depth + 1.
static const uint kStack = 64;

/// A leaf is its particle, recorded by index in its cloud.
struct BvhParticles : IPrimitives {
    static uint particleOf(uint raw) { return raw; }
};

/// Where [tMin, tMax] of the ray enters box `node`, or a miss (> tMax).
float boxEntry(uint node, float3 origin, float3 inverse, float tMin, float tMax) {
    const float3 lo = bvhBoxes[node * 2].xyz;
    const float3 hi = bvhBoxes[node * 2 + 1].xyz;
    const float3 a = (lo - origin) * inverse;
    const float3 b = (hi - origin) * inverse;
    const float3 near = min(a, b);
    const float3 far = max(a, b);
    const float enter = max(max(near.x, near.y), max(near.z, tMin));
    const float leave = min(min(far.x, far.y), min(far.z, tMax));
    return enter <= leave ? enter : 3.4e38;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rtBvhRender(uint3 tid: SV_DispatchThreadID) {
    const uint px = tid.x;
    const uint py = tid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    RayDesc ray = primaryRay(px, py);
    Integrator<BvhParticles> integrator;
    Cursor cursor;
    integrator.begin(cursor);
    uint stack[kStack];
    float stackEntry[kStack];   // where the ray enters what is on the stack
    for (uint segment = 0; segment < params.maxSegments; ++segment) {
        integrator.beginSegment(cursor);
        for (uint instance = 0; instance < params.instances; ++instance) {
            float3 origin;
            float3 direction;
            toCloud(ray, instance, origin, direction);
            // IEEE division gives the infinities a slab test wants for a zero component.
            const float3 inverse = float3(1.0) / direction;
            const uint nodeBase = bvhInstances[instance * 3];
            const uint leafBase = bvhInstances[instance * 3 + 1];
            const uint count = bvhInstances[instance * 3 + 2];
            const uint particleBase = instanceIndices[instance * 2];
            uint depthOfStack = 0;
            // A one-particle cloud has no internal node: its root is its leaf.
            stack[depthOfStack] = count == 1 ? kLeafBit : 0u;
            stackEntry[depthOfStack] = ray.TMin;
            depthOfStack += 1;
            while (depthOfStack > 0) {
                depthOfStack -= 1;
                const uint child = stack[depthOfStack];
                // Tested when pushed; only the closing record can rule it out now.
                if (stackEntry[depthOfStack] >= cursor.closed) {
                    continue;
                }
                if ((child & kLeafBit) != 0) {
                    const uint particle = bvhLeaves[leafBase + (child & ~kLeafBit)];
                    const uint id = particleBase + particle;
                    const float4 f0 = frames[id * 4];
                    const float4 f1 = frames[id * 4 + 1];
                    const float4 f2 = frames[id * 4 + 2];
                    const float cutoff = frames[id * 4 + 3].x;
                    const float3x3 m = float3x3(f0.xyz, f1.xyz, f2.xyz);
                    const float3 o = mul(m, origin - float3(f0.w, f1.w, f2.w));
                    const float3 d = mul(m, direction);
                    // |o + t d|^2 = cutoff: where the ray enters the cut
                    // ellipsoid. From the peak and the distance to it, not
                    // b^2 - ac: a flat particle makes d huge along its thin
                    // axis, and the textbook discriminant cancels to nothing
                    // (it lost 176 of 400 thin layers).
                    const float a = dot(d, d);
                    if (!(a > 0.0)) {
                        continue;
                    }
                    const float tPeak = -dot(o, d) / a;
                    const float3 closest = o + tPeak * d;
                    const float power = dot(closest, closest);
                    if (power > cutoff) {
                        continue;
                    }
                    const float entry = tPeak - sqrt((cutoff - power) / a);
                    if (entry < ray.TMin || entry > ray.TMax) {
                        continue;
                    }
                    integrator.offer(cursor, entry, particle, instance);
                    continue;
                }
                const uint node = nodeBase + child;
                const float limit = min(cursor.closed, ray.TMax);
                const uint left = bvhChildren[node * 2];
                const uint right = bvhChildren[node * 2 + 1];
                // Children's boxes tested here, once. Leaves are tested when
                // popped, against their ellipsoid.
                const float leftEntry = (left & kLeafBit) != 0 ? ray.TMin : boxEntry(nodeBase + left, origin, inverse, ray.TMin, limit);
                const float rightEntry = (right & kLeafBit) != 0 ? ray.TMin : boxEntry(nodeBase + right, origin, inverse, ray.TMin, limit);
                if (depthOfStack + 2 > kStack) {
                    continue;   // cannot happen: see kStack
                }
                // Nearer child on top.
                const bool leftFirst = leftEntry <= rightEntry;
                const uint farChild = leftFirst ? right : left;
                const float farEntry = leftFirst ? rightEntry : leftEntry;
                const uint nearChild = leftFirst ? left : right;
                const float nearEntry = leftFirst ? leftEntry : rightEntry;
                if (farEntry <= limit) {
                    stack[depthOfStack] = farChild;
                    stackEntry[depthOfStack] = farEntry;
                    depthOfStack += 1;
                }
                if (nearEntry <= limit) {
                    stack[depthOfStack] = nearChild;
                    stackEntry[depthOfStack] = nearEntry;
                    depthOfStack += 1;
                }
            }
        }
        if (!integrator.endSegment(cursor, ray, segment + 1 == params.maxSegments)) {
            break;
        }
    }
    integrator.write(px, py);
}
