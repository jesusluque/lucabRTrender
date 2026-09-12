// Copyright (c) 2026 lucabRTrender contributors.
//
// The light BVH's build: the kernels and the buffers they bind. Kept out
// of light_bvh.slang on purpose -- a module's global parameters join every
// kernel that imports it, and the shading kernels import the tree to
// choose from it; these buffers would have put them past Metal's thirty-one.
module light_bvh_build;
import lrt.light.lights;
import lrt.light.light_bvh;
import lrt.rt.bvh_common;
import lrt.common.morton;

// ---- build ------------------------------------------------------------------

StructuredBuffer<LightRecord>  lights;
StructuredBuffer<uint>         leafLights;    // the bounded lights, in leaf order
RWStructuredBuffer<float4>     leafBoxes;
RWStructuredBuffer<uint>       mortonKeys;
RWStructuredBuffer<uint>       mortonValues;
StructuredBuffer<float4>       boxes;         // the hierarchy's internal boxes (bvh_refit)
StructuredBuffer<uint>         children;      // the hierarchy's (kLeafBit | leaf)
StructuredBuffer<uint>         leaves;        // leaf order -> leaf index
RWStructuredBuffer<float>      nodeValues;    // [internal nodes][leaf nodes][unbounded], 16 floats each, at nodeBase
uniform uint                   nodeBase;
RWStructuredBuffer<uint>       changed;
ConstantBuffer<BvhParams>      params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lightLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    float3 lo, hi, axis;
    float cosO, cosE;
    lightBounds(lights[leafLights[i]], lo, hi, axis, cosO, cosE);
    leafBoxes[i * 2] = float4(lo, 0.0);
    leafBoxes[i * 2 + 1] = float4(hi, 0.0);
    const float3 boundsLo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ);
    const float3 boundsHi = float3(params.boundsHiX, params.boundsHiY, params.boundsHiZ);
    mortonKeys[i] = morton30((0.5 * (lo + hi) - boundsLo) / max(boundsHi - boundsLo, float3(1e-20)));
    mortonValues[i] = i;
}

/// Nodes from the hierarchy: internal node u at u, leaf k at (count - 1) + k;
/// a leaf's power is its light's share, its cone the light's.
[shader("compute")]
[numthreads(256, 1, 1)]
void lightPack(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.count;
    const uint u = tid.x;
    const uint internalCount = n > 1 ? n - 1 : 0;
    if (u >= internalCount + n) {
        return;
    }
    LightNode node;
    if (u < internalCount) {
        node.lo = float4(boxes[u * 2].xyz, 0.0);
        node.hi = float4(boxes[u * 2 + 1].xyz, 1.0);
        node.cone = float4(0.0, 0.0, 1.0, 0.0);
        uint2 kids;
        for (uint k = 0; k < 2; ++k) {
            const uint c = children[u * 2 + k];
            kids[k] = (c & kLeafBit) != 0 ? internalCount + leaves[c & ~kLeafBit] : c;
        }
        node.links = uint4(kids.x, kids.y, kNoNode, 0);
    } else {
        const uint leaf = u - internalCount;
        const uint light = leafLights[leaf];
        const LightRecord l = lights[light];
        float3 lo, hi, axis;
        float cosO, cosE;
        lightBounds(l, lo, hi, axis, cosO, cosE);
        const float power = l.cumulative - (light > 0 ? lights[light - 1].cumulative : 0.0);
        node.lo = float4(lo, max(power, 0.0));
        node.hi = float4(hi, cosO);
        node.cone = float4(axis, cosE);
        node.links = uint4(kLeafBit | light, kLeafBit | light, kNoNode, 1);
    }
    storeLightNode(nodeValues, nodeBase, u, node);
}

/// Parents from the children, once the nodes stand.
[shader("compute")]
[numthreads(256, 1, 1)]
void lightParents(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.count;
    const uint u = tid.x;
    if (n < 2 || u >= n - 1) {
        return;
    }
    const uint4 links = lightNodeAtRW(nodeValues, nodeBase, u).links;
    nodeValues[nodeBase + links.x * 16 + 14] = asfloat(u);
    nodeValues[nodeBase + links.y * 16 + 14] = asfloat(u);
}

/// One pass: every internal node from its children; converges in depth passes.
[shader("compute")]
[numthreads(256, 1, 1)]
void lightSettle(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.count;
    const uint u = tid.x;
    if (n < 2 || u >= n - 1) {
        return;
    }
    LightNode node = lightNodeAtRW(nodeValues, nodeBase, u);
    const LightNode a = lightNodeAtRW(nodeValues, nodeBase, node.links.x);
    const LightNode b = lightNodeAtRW(nodeValues, nodeBase, node.links.y);
    const float power = a.lo.w + b.lo.w;
    float3 axis;
    float o, e;
    coneUnion(a.cone.xyz, a.hi.w, a.cone.w, b.cone.xyz, b.hi.w, b.cone.w, axis, o, e);
    const float3 lo = min(a.lo.xyz, b.lo.xyz);
    const float3 hi = max(a.hi.xyz, b.hi.xyz);
    const bool same = node.lo.w == power && node.hi.w == o && all(node.cone.xyz == axis) && node.cone.w == e &&
                      all(node.lo.xyz == lo) && all(node.hi.xyz == hi);
    node.lo = float4(lo, power);
    node.hi = float4(hi, o);
    node.cone = float4(axis, e);
    storeLightNode(nodeValues, nodeBase, u, node);
    changed[u] = same ? 0u : 1u;
}

RWStructuredBuffer<LightRecord> records;      // lightPack, lightUnbounded: pad0 written
StructuredBuffer<uint>          unboundedLights;
uniform uint                    treeNodeCount;

[shader("compute")]
[numthreads(256, 1, 1)]
void lightUnbounded(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {   // here: the unbounded count
        return;
    }
    const uint light = unboundedLights[i];
    const LightRecord l = records[light];
    LightNode node;
    node.lo = float4(0.0, 0.0, 0.0, max(l.cumulative - (light > 0 ? records[light - 1].cumulative : 0.0), 0.0));
    node.hi = float4(0.0, 0.0, 0.0, -1.0);
    node.cone = float4(0.0, 0.0, 1.0, 1.0);
    node.links = uint4(kLeafBit | light, 0, kNoNode, 2);
    storeLightNode(nodeValues, nodeBase, treeNodeCount + i, node);
    records[light].pad0 = treeNodeCount + i;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lightLeafIndices(uint3 tid: SV_DispatchThreadID) {
    const uint leaf = tid.x;
    if (leaf >= params.count) {
        return;
    }
    const uint internalCount = params.count > 1 ? params.count - 1 : 0;
    records[leafLights[leaf]].pad0 = internalCount + leaf;
}

