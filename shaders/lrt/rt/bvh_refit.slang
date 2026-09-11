// Copyright (c) 2026 lucabRTrender contributors.
//
// One refit pass: every internal node's box becomes the union of its
// children's, in place. Boxes only grow toward their final extent, so a node
// reading a child mid-update reads something between the old and the new box
// and the passes still converge; the tree is done after a pass that changed
// nothing, which `changed` reports. Passes needed: the tree's depth.
module bvh_refit;
import bvh_common;

StructuredBuffer<float4>   leafBoxes;   // per particle, cloud-local
StructuredBuffer<uint>     children;
StructuredBuffer<uint>     leaves;
RWStructuredBuffer<float4> boxes;
RWStructuredBuffer<uint>   changed;     // per internal node (cloud-local)
ConstantBuffer<BvhParams>  params;

void childBox(uint child, out float3 lo, out float3 hi) {
    if ((child & kLeafBit) != 0) {
        const uint particle = leaves[params.leafBase + (child & ~kLeafBit)];
        lo = leafBoxes[particle * 2].xyz;
        hi = leafBoxes[particle * 2 + 1].xyz;
    } else {
        lo = boxes[(params.nodeBase + child) * 2].xyz;
        hi = boxes[(params.nodeBase + child) * 2 + 1].xyz;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bvhRefit(uint3 tid: SV_DispatchThreadID) {
    const uint u = tid.x;
    if (u + 1 >= params.count) {
        return;
    }
    const uint node = params.nodeBase + u;
    float3 aLo, aHi, bLo, bHi;
    childBox(children[node * 2], aLo, aHi);
    childBox(children[node * 2 + 1], bLo, bHi);
    const float3 lo = min(aLo, bLo);
    const float3 hi = max(aHi, bHi);
    const bool same = all(lo == boxes[node * 2].xyz) && all(hi == boxes[node * 2 + 1].xyz);
    boxes[node * 2] = float4(lo, 0.0);
    boxes[node * 2 + 1] = float4(hi, 0.0);
    changed[u] = same ? 0u : 1u;
}
