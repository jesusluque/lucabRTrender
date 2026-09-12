// Copyright (c) 2026 lucabRTrender contributors.
//
// A mesh's LBVH checked from the outside: every internal node's box must
// hold its children's, a leaf's box being the one its triangle's positions
// make right now. After a refit over deformed positions this must count 0;
// over the tree as it stood before, it does not.
module bvh_check;
import lrt.rt.bvh_common;
import lrt.world.scene_types;

StructuredBuffer<float4>     positions;
StructuredBuffer<uint>       indices;
StructuredBuffer<float4>     meshBoxes;
StructuredBuffer<uint>       meshChildren;
StructuredBuffer<uint>       meshLeaves;
RWStructuredBuffer<uint>     violations;   // [nodes whose box misses a child, nodes checked]
ConstantBuffer<BvhParams>    params;
ConstantBuffer<MeshRecord>   mesh;

void childBox(uint child, out float3 lo, out float3 hi) {
    if ((child & kLeafBit) != 0) {
        const uint t = meshLeaves[params.leafBase + (child & ~kLeafBit)];
        const uint base = (mesh.firstTriangle + t) * 3;
        const float3 a = positions[mesh.firstPoint + indices[base]].xyz;
        const float3 b = positions[mesh.firstPoint + indices[base + 1]].xyz;
        const float3 c = positions[mesh.firstPoint + indices[base + 2]].xyz;
        lo = min(a, min(b, c));
        hi = max(a, max(b, c));
    } else {
        lo = meshBoxes[(params.nodeBase + child) * 2].xyz;
        hi = meshBoxes[(params.nodeBase + child) * 2 + 1].xyz;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bvhCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint bad = 0;
    const uint n = params.count > 1 ? params.count - 1 : 0;
    for (uint u = 0; u < n; ++u) {
        const uint node = params.nodeBase + u;
        const float3 lo = meshBoxes[node * 2].xyz;
        const float3 hi = meshBoxes[node * 2 + 1].xyz;
        for (uint k = 0; k < 2; ++k) {
            float3 cLo, cHi;
            childBox(meshChildren[node * 2 + k], cLo, cHi);
            if (any(cLo < lo) || any(cHi > hi)) {
                bad += 1;
                break;
            }
        }
    }
    violations[0] = bad;
    violations[1] = n;
}
