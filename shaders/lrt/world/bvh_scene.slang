// Copyright (c) 2026 lucabRTrender contributors.
//
// The leaves of a scene's compute BVHs (the route for devices without ray
// tracing hardware), for the same LBVH build the splat ray tracer uses
// (lrt/rt/bvh_*.slang): sort, hierarchy, refit.
//   triangleLeaves   one mesh: each triangle's box in object space and the
//                    Morton code of its centre within the mesh's bounds
//   instanceLeaves   the frame: each instance's mesh bounds as a box in view
//                    space, and its centre for the codes
//   instanceCodes    the codes, once the centres' bounds are known
module bvh_scene;
import lrt.common.morton;
import lrt.rt.bvh_common;
import scene_types;

StructuredBuffer<float4>          positions;
StructuredBuffer<uint>            indices;
StructuredBuffer<MeshRecord>      meshes;
StructuredBuffer<InstanceRecord>  instances;
RWStructuredBuffer<float4>        leafBoxes;     // per leaf: lo, hi
RWStructuredBuffer<float4>        centres;       // instanceLeaves: per instance
RWStructuredBuffer<uint>          mortonKeys;
RWStructuredBuffer<uint>          mortonValues;
ConstantBuffer<BvhParams>         params;
ConstantBuffer<MeshRecord>        mesh;           // triangleLeaves: the mesh

[shader("compute")]
[numthreads(256, 1, 1)]
void triangleLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= params.count) {
        return;
    }
    const uint base = (mesh.firstTriangle + t) * 3;
    const float3 a = positions[mesh.firstPoint + indices[base]].xyz;
    const float3 b = positions[mesh.firstPoint + indices[base + 1]].xyz;
    const float3 c = positions[mesh.firstPoint + indices[base + 2]].xyz;
    const float3 lo = min(a, min(b, c));
    const float3 hi = max(a, max(b, c));
    leafBoxes[t * 2] = float4(lo, 0.0);
    leafBoxes[t * 2 + 1] = float4(hi, 0.0);
    const float3 boundsLo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ);
    const float3 boundsHi = float3(params.boundsHiX, params.boundsHiY, params.boundsHiZ);
    mortonKeys[t] = morton30(((lo + hi) * 0.5 - boundsLo) / max(boundsHi - boundsLo, float3(1e-20)));
    mortonValues[t] = t;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void instanceLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const InstanceRecord r = instances[i];
    const MeshRecord m = meshes[r.mesh];
    // The box's centre and half extent carried through the 3x4: exact for
    // the transformed box's bounds (Arvo).
    const float3 centre = (m.boundsLo.xyz + m.boundsHi.xyz) * 0.5;
    const float3 half = (m.boundsHi.xyz - m.boundsLo.xyz) * 0.5;
    const float3x3 linear = float3x3(r.view0.xyz, r.view1.xyz, r.view2.xyz);
    const float3 c = mul(linear, centre) + float3(r.view0.w, r.view1.w, r.view2.w);
    const float3 e = mul(abs(linear), half);
    leafBoxes[i * 2] = float4(c - e, 0.0);
    leafBoxes[i * 2 + 1] = float4(c + e, 0.0);
    centres[i] = float4(c, 1.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void instanceCodes(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float3 boundsLo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ);
    const float3 boundsHi = float3(params.boundsHiX, params.boundsHiY, params.boundsHiZ);
    mortonKeys[i] = morton30((centres[i].xyz - boundsLo) / max(boundsHi - boundsLo, float3(1e-20)));
    mortonValues[i] = i;
}
