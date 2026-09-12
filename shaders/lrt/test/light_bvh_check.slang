// Copyright (c) 2026 lucabRTrender contributors.
//
// The light BVH from the outside: every node's box and cone must hold its
// children's; the choice's probabilities must sum to one over the lights at
// any point; the probability a draw reports must be the one lightPdfChoice
// recomputes for the light it drew; and a histogram of many draws must
// follow those probabilities (checked on the host by chi-square).
module light_bvh_check;
import lrt.light.lights;
import lrt.light.light_bvh;
import lrt.rt.bvh_common;

struct BvhCheckParams {
    uint  treeNodes;
    uint  unboundedCount;
    uint  lightCount;
    uint  draws;
    uint  points;      // for the sum: random points
    uint  seed;
    uint  pad0; uint pad1;
    float3 position; float pad2;   // the draws' shading point
    float3 normal;   float pad3;
};

StructuredBuffer<LightRecord> lights;
StructuredBuffer<float>       nodeValues;
uniform uint                  nodeBase;
RWStructuredBuffer<uint>      counts;     // containment: [box violations, cone violations, nodes]; sum: [points off, points]
RWStructuredBuffer<float>     worst;
RWStructuredBuffer<uint>      histogram;  // per light: draws that chose it
RWStructuredBuffer<uint>      mismatches; // [draws whose probability differs from lightPdfChoice, draws]
RWStructuredBuffer<float>     expected;   // per light: its probability at the draws' point
ConstantBuffer<BvhCheckParams> check;

uint hashOf(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float unitOf(uint x) { return float(hashOf(x) & 0xFFFFFFu) / 16777216.0; }

[shader("compute")]
[numthreads(1, 1, 1)]
void lightBvhContainment(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint boxes = 0;
    uint cones = 0;
    const uint internalCount = check.treeNodes > 0 ? (check.treeNodes + 1) / 2 - 1 : 0;
    for (uint u = 0; u < internalCount; ++u) {
        const LightNode node = lightNodeAt(nodeValues, nodeBase, u);
        for (uint k = 0; k < 2; ++k) {
            const LightNode child = lightNodeAt(nodeValues, nodeBase, k == 0 ? node.links.x : node.links.y);
            if (any(child.lo.xyz < node.lo.xyz - 1e-5) || any(child.hi.xyz > node.hi.xyz + 1e-5)) {
                boxes += 1;
            }
            // The child's cone within the parent's: the angle between the
            // axes plus the child's half angle within the parent's, and the
            // child's emission angle within the parent's.
            const float thetaP = acos(clamp(node.hi.w, -1.0, 1.0));
            const float thetaC = acos(clamp(child.hi.w, -1.0, 1.0));
            const float between = acos(clamp(dot(node.cone.xyz, child.cone.xyz), -1.0, 1.0));
            const bool orientationHeld = thetaP >= 3.14159 || between + thetaC <= thetaP + 1e-3;
            const bool emissionHeld = child.cone.w >= node.cone.w - 1e-5;
            if (!orientationHeld || !emissionHeld) {
                cones += 1;
            }
        }
    }
    counts[0] = boxes;
    counts[1] = cones;
    counts[2] = internalCount;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lightBvhPdfSum(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint off = 0;
    float largest = 0.0;
    for (uint i = 0; i < check.points; ++i) {
        const float3 p = float3(unitOf(check.seed + i * 3), unitOf(check.seed + i * 3 + 1), unitOf(check.seed + i * 3 + 2)) * 8.0 - 4.0;
        const float3 n = normalize(float3(unitOf(i * 7 + 1), unitOf(i * 7 + 2), unitOf(i * 7 + 3)) * 2.0 - 1.0);
        float sum = 0.0;
        for (uint k = 0; k < check.lightCount; ++k) {
            sum += lightPdfChoiceAny(nodeValues, nodeBase, check.treeNodes, check.unboundedCount, lights, k, p, n);
        }
        const float gap = abs(sum - 1.0);
        largest = max(largest, gap);
        off += gap > 1e-4 ? 1u : 0u;
    }
    counts[0] = off;
    counts[1] = check.points;
    worst[0] = largest;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lightBvhDraws(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.draws) {
        return;
    }
    const float u = unitOf(check.seed ^ (i * 2654435761u));
    const LightChoice choice =
        chooseLightAny(nodeValues, nodeBase, check.treeNodes, check.unboundedCount, check.position, check.normal, u);
    if (!choice.valid) {
        return;
    }
    InterlockedAdd(histogram[choice.index], 1);
    const float pdf = lightPdfChoiceAny(nodeValues, nodeBase, check.treeNodes, check.unboundedCount, lights, choice.index,
                                        check.position, check.normal);
    if (abs(pdf - choice.probability) > 1e-4 * max(pdf, choice.probability)) {
        InterlockedAdd(mismatches[0], 1);
    }
    InterlockedAdd(mismatches[1], 1);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lightBvhExpected(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= check.lightCount) {
        return;
    }
    expected[k] = lightPdfChoiceAny(nodeValues, nodeBase, check.treeNodes, check.unboundedCount, lights, k, check.position,
                                    check.normal);
}
