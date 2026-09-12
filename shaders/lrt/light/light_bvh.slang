// Copyright (c) 2026 lucabRTrender contributors.
//
// A BVH over the frame's bounded lights, so a shading point chooses one
// light by how much it can matter there -- power over distance squared,
// within the cone of directions the light emits into (Conty and Kulla
// 2018) -- instead of by power alone. One buffer of nodes: a box, an
// orientation cone, the power under it, and its links; a leaf is a light.
// Dome and distant lights are unbounded and never enter the tree: they sit
// in a list always visited, chosen against the tree by their share of the
// power, or a dome would vanish in silence.
//
// The build (light_bvh_build.slang) reuses the compute LBVH's kernels
// (bvh_hierarchy, bvh_refit) over the lights' boxes; lightLeaves writes
// those and the Morton codes, lightPack turns the hierarchy into nodes,
// lightSettle sums the power and unions the cones up the tree until
// nothing changes. This module holds no global parameter of its own, so
// importing it costs a shading kernel no binding.
module light_bvh;
import lrt.light.lights;
import lrt.rt.bvh_common;

public struct LightNode {
    public float4 lo;       // box; w: the power under the node
    public float4 hi;       // box; w: cos of the orientation cone's half angle theta_o
    public float4 cone;     // axis; w: cos of the emission half angle theta_e
    public uint4  links;    // child0, child1 (kLeafBit | light index at a leaf), parent, 1 where a leaf, 2 unbounded
};

// The nodes are sixteen floats each in a float buffer -- the light table's
// IES values buffer, after the profiles -- because a kernel on Metal binds
// thirty-one buffers at most and the shading kernels bind them all: one
// buffer more for the tree put a binding out of range (buffer(38)).
public LightNode lightNodeAt(StructuredBuffer<float> values, uint base, uint i) {
    const uint at = base + i * 16;
    LightNode node;
    node.lo = float4(values[at], values[at + 1], values[at + 2], values[at + 3]);
    node.hi = float4(values[at + 4], values[at + 5], values[at + 6], values[at + 7]);
    node.cone = float4(values[at + 8], values[at + 9], values[at + 10], values[at + 11]);
    node.links = uint4(asuint(values[at + 12]), asuint(values[at + 13]), asuint(values[at + 14]), asuint(values[at + 15]));
    return node;
}

public LightNode lightNodeAtRW(RWStructuredBuffer<float> values, uint base, uint i) {
    const uint at = base + i * 16;
    LightNode node;
    node.lo = float4(values[at], values[at + 1], values[at + 2], values[at + 3]);
    node.hi = float4(values[at + 4], values[at + 5], values[at + 6], values[at + 7]);
    node.cone = float4(values[at + 8], values[at + 9], values[at + 10], values[at + 11]);
    node.links = uint4(asuint(values[at + 12]), asuint(values[at + 13]), asuint(values[at + 14]), asuint(values[at + 15]));
    return node;
}

public void storeLightNode(RWStructuredBuffer<float> values, uint base, uint i, LightNode node) {
    const uint at = base + i * 16;
    for (uint k = 0; k < 4; ++k) {
        values[at + k] = node.lo[k];
        values[at + 4 + k] = node.hi[k];
        values[at + 8 + k] = node.cone[k];
        values[at + 12 + k] = asfloat(node.links[k]);
    }
}

public static const uint kNoNode = 0xFFFFFFFF;
static const float kBvhPi = 3.14159265358979;

/// A bounded light's box and cone from its record.
public void lightBounds(LightRecord l, out float3 lo, out float3 hi, out float3 axis, out float cosThetaO,
                 out float cosThetaE) {
    const float3 o = lightOrigin(l);
    const float3 az = lightAxisZ(l);
    float r = max(l.sizeX, 0.0);
    if (l.kind == kLightRect) {
        r = 0.5 * length(float2(l.sizeX, l.sizeY));
    } else if (l.kind == kLightCylinder) {
        r = length(float2(0.5 * l.sizeY, l.sizeX));
    }
    lo = o - float3(r);
    hi = o + float3(r);
    // Emission: a sphere or a cylinder everywhere (theta_o = pi), a disk or
    // rect down -Z within a hemisphere; a shaping cone narrows theta_e.
    axis = -az;
    if (l.kind == kLightSphere || l.kind == kLightCylinder) {
        cosThetaO = -1.0;
        cosThetaE = 0.0;
    } else {
        cosThetaO = 1.0;
        cosThetaE = 0.0;
    }
    if (l.coneCos > -1.0) {
        cosThetaE = max(cosThetaE, l.coneCos);
    }
}

/// The union of two cones about their axes (Conty and Kulla, algorithm 1):
/// the wider one, or a cone about an axis turned toward the other by half
/// the excess, wide enough to hold both.
public void coneUnion(float3 aAxis, float aO, float aE, float3 bAxis, float bO, float bE, out float3 axis,
                      out float o, out float e) {
    if (bO < aO) {   // make a the wider cone: the smaller cosine
        const float3 t = aAxis; aAxis = bAxis; bAxis = t;
        const float t2 = aO; aO = bO; bO = t2;
    }
    const float thetaA = acos(clamp(aO, -1.0, 1.0));
    const float thetaB = acos(clamp(bO, -1.0, 1.0));
    const float thetaD = acos(clamp(dot(aAxis, bAxis), -1.0, 1.0));
    e = max(aE, bE);
    if (min(thetaD + thetaB, kBvhPi) <= thetaA) {
        axis = aAxis;
        o = aO;
        return;
    }
    const float thetaO = 0.5 * (thetaA + thetaD + thetaB);
    if (thetaO >= kBvhPi) {
        axis = aAxis;
        o = -1.0;
        return;
    }
    const float thetaR = thetaO - thetaA;
    const float3 perp = cross(aAxis, bAxis);
    if (dot(perp, perp) < 1e-20) {
        axis = aAxis;
    } else {
        const float3 k = normalize(perp);
        const float c = cos(thetaR);
        const float sn = sin(thetaR);
        axis = aAxis * c + cross(k, aAxis) * sn + k * dot(k, aAxis) * (1.0 - c);
    }
    o = cos(thetaO);
}

// ---- choosing --------------------------------------------------------------

/// How much a node can matter at p with normal n (Conty and Kulla eq. 1).
public float lightImportance(LightNode node, float3 p, float3 n) {
    const float3 centre = 0.5 * (node.lo.xyz + node.hi.xyz);
    const float radius = 0.5 * length(node.hi.xyz - node.lo.xyz);
    const float3 toP = p - centre;
    const float d2 = max(dot(toP, toP), 1e-12);
    const float d = sqrt(d2);
    const float3 wi = toP / d;
    // The angular uncertainty of the box seen from p.
    const float sinU = clamp(radius / d, 0.0, 1.0);
    const float thetaU = asin(sinU);
    const float thetaO = acos(clamp(node.hi.w, -1.0, 1.0));
    const float thetaE = acos(clamp(node.cone.w, -1.0, 1.0));
    const float theta = acos(clamp(dot(node.cone.xyz, wi), -1.0, 1.0));
    const float thetaPrime = max(theta - thetaO - thetaU, 0.0);
    if (thetaPrime >= thetaE) {
        return 0.0;
    }
    // The surface's own cosine, with the same uncertainty.
    const float thetaI = acos(clamp(dot(n, -wi), -1.0, 1.0));
    const float thetaIPrime = max(thetaI - thetaU, 0.0);
    const float cosI = thetaIPrime < 0.5 * kBvhPi ? cos(thetaIPrime) : 0.0;
    return node.lo.w * cos(thetaPrime) * cosI / max(d2, radius * radius);
}

/// The descent from the root: one light, and the probability of it.
public LightChoice chooseLightBvh(StructuredBuffer<float> tree, uint base, uint nodeCount, float3 p, float3 n, float u) {
    LightChoice choice;
    choice.index = 0;
    choice.probability = 0.0;
    choice.valid = false;
    if (nodeCount == 0) {
        return choice;
    }
    uint node = 0;
    float probability = 1.0;
    float pick = clamp(u, 0.0, 0.999999);
    for (uint step = 0; step < 64; ++step) {
        const LightNode current = lightNodeAt(tree, base, node);
        if (current.links.w != 0) {
            choice.index = current.links.x & ~kLeafBit;
            choice.probability = probability;
            choice.valid = probability > 0.0;
            return choice;
        }
        const LightNode childA = lightNodeAt(tree, base, current.links.x);
        const LightNode childB = lightNodeAt(tree, base, current.links.y);
        float ia = lightImportance(childA, p, n);
        float ib = lightImportance(childB, p, n);
        if (!(ia + ib > 0.0)) {
            // Neither child can light p by its bound, though the parent's
            // looser bound let the descent in: choose by power, so that no
            // probability is lost and every light keeps a non-zero chance.
            ia = childA.lo.w;
            ib = childB.lo.w;
        }
        const float total = ia + ib;
        if (!(total > 0.0)) {
            return choice;
        }
        const float pa = ia / total;
        if (pick < pa) {
            probability *= pa;
            pick = pick / pa;
            node = current.links.x;
        } else {
            probability *= 1.0 - pa;
            pick = (pick - pa) / (1.0 - pa);
            node = current.links.y;
        }
    }
    return choice;
}

/// The probability the descent gives leaf node `leaf`, walked up by parents.
public float lightPdfChoiceBvh(StructuredBuffer<float> tree, uint base, uint leaf, float3 p, float3 n) {
    float probability = 1.0;
    uint node = leaf;
    for (uint step = 0; step < 64; ++step) {
        const uint parent = lightNodeAt(tree, base, node).links.z;
        if (parent == kNoNode) {
            break;
        }
        const uint4 links = lightNodeAt(tree, base, parent).links;
        const LightNode childA = lightNodeAt(tree, base, links.x);
        const LightNode childB = lightNodeAt(tree, base, links.y);
        float ia = lightImportance(childA, p, n);
        float ib = lightImportance(childB, p, n);
        if (!(ia + ib > 0.0)) {
            ia = childA.lo.w;
            ib = childB.lo.w;
        }
        const float total = ia + ib;
        if (!(total > 0.0)) {
            return 0.0;
        }
        probability *= (node == links.x ? ia : ib) / total;
        node = parent;
    }
    return probability;
}

// ---- the tree and the unbounded list together --------------------------------
//
// The nodes buffer holds the tree's nodes first (2n - 1 of them for n bounded
// lights), then one entry per unbounded light (links.w = 2), so one buffer
// carries both. A light's record keeps its node's index in pad0.

/// The frame's power in the tree, and in the unbounded list.
void lightPowers(StructuredBuffer<float> tree, uint base, uint treeNodes, uint unboundedCount, out float inTree,
                 out float unbounded) {
    inTree = treeNodes > 0 ? tree[base + 3] : 0.0;
    unbounded = 0.0;
    for (uint i = 0; i < unboundedCount; ++i) {
        unbounded += tree[base + (treeNodes + i) * 16 + 3];
    }
}

/// One light for the point p with normal n: the tree against the list by
/// their shares of the power, then the descent, or the list by power.
public LightChoice chooseLightAny(StructuredBuffer<float> tree, uint base, uint treeNodes, uint unboundedCount,
                                  float3 p, float3 n, float u) {
    LightChoice choice;
    choice.index = 0;
    choice.probability = 0.0;
    choice.valid = false;
    float inTree, unbounded;
    lightPowers(tree, base, treeNodes, unboundedCount, inTree, unbounded);
    const float total = inTree + unbounded;
    if (!(total > 0.0)) {
        return choice;
    }
    const float pTree = inTree / total;
    float pick = clamp(u, 0.0, 0.999999);
    if (pick < pTree) {
        choice = chooseLightBvh(tree, base, treeNodes, p, n, pick / pTree);
        choice.probability *= pTree;
        choice.valid = choice.valid && choice.probability > 0.0;
        return choice;
    }
    pick = (pick - pTree) / (1.0 - pTree);
    float run = 0.0;
    for (uint i = 0; i < unboundedCount; ++i) {
        const LightNode entry = lightNodeAt(tree, base, treeNodes + i);
        const float share = entry.lo.w / unbounded;
        if (pick < run + share || i + 1 == unboundedCount) {
            choice.index = entry.links.x & ~kLeafBit;
            choice.probability = (1.0 - pTree) * share;
            choice.valid = choice.probability > 0.0;
            return choice;
        }
        run += share;
    }
    return choice;
}

/// The probability chooseLightAny gives light `light` at p, n.
public float lightPdfChoiceAny(StructuredBuffer<float> tree, uint base, uint treeNodes, uint unboundedCount,
                               StructuredBuffer<LightRecord> lights, uint light, float3 p, float3 n) {
    float inTree, unbounded;
    lightPowers(tree, base, treeNodes, unboundedCount, inTree, unbounded);
    const float total = inTree + unbounded;
    if (!(total > 0.0)) {
        return 0.0;
    }
    const uint node = lights[light].pad0;
    if (node >= treeNodes) {
        return (unbounded / total) * (tree[base + node * 16 + 3] / unbounded);
    }
    return (inTree / total) * lightPdfChoiceBvh(tree, base, node, p, n);
}
