// Copyright (c) 2026 lucabRTrender contributors.
//
// The arbitrary outputs Hydra asks for, from the same visibility and the
// same reconstructed surface shading uses: ids (prim, instance, element -- the
// authored face -- with -1 for nothing), the normal in eye and world space,
// and primvars by slot. Bottom row first, like every engine image.
module aovs;
import surface;

struct AovParams {
    uint primvarSlots;   // how many primvar outputs
    uint pad0; uint pad1; uint pad2;
};

Texture2D<uint4>            visibility;
RWStructuredBuffer<uint>    ids;          // 3 per pixel: prim, instance, element (int32 bits)
RWStructuredBuffer<float4>  eyeNormals;   // Neye
RWStructuredBuffer<float4>  worldNormals; // normal
RWStructuredBuffer<float4>  primvarsOut;  // primvarSlots per pixel
StructuredBuffer<uint>      primvarSlotOf; // per output: the scene slot
ConstantBuffer<CameraParams> camera;
ConstantBuffer<AovParams>   params;

[shader("compute")]
[numthreads(16, 16, 1)]
void aovs(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        ids[at * 3] = 0xFFFFFFFF;
        ids[at * 3 + 1] = 0xFFFFFFFF;
        ids[at * 3 + 2] = 0xFFFFFFFF;
        eyeNormals[at] = float4(0.0);
        worldNormals[at] = float4(0.0);
        for (uint k = 0; k < params.primvarSlots; ++k) {
            primvarsOut[at * params.primvarSlots + k] = float4(0.0);
        }
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    ids[at * 3] = s.instance.primId;
    ids[at * 3 + 1] = s.instance.instanceId;
    ids[at * 3 + 2] = triangleFaces[s.mesh.firstTriangle + s.triangle];
    // Facing the eye, as Hydra's renderers report shading normals.
    const float3 n = dot(s.normal, s.viewDirection) > 0.0 ? -s.normal : s.normal;
    // Hydra's eye space looks down -Z; the engine's view down +z.
    eyeNormals[at] = float4(n.x, n.y, -n.z, 0.0);
    // World: back through view's rotation (world rows have the object's
    // scale; the normal matrix of world is view's with the camera removed).
    const float3x3 objectToWorld = float3x3(s.instance.world0.xyz, s.instance.world1.xyz, s.instance.world2.xyz);
    const float3x3 normalToView = float3x3(s.instance.normal0.xyz, s.instance.normal1.xyz, s.instance.normal2.xyz);
    const float3x3 objectToView = float3x3(s.instance.view0.xyz, s.instance.view1.xyz, s.instance.view2.xyz);
    // world normal = worldToView^T * viewNormal, with worldToView = objectToView * inverse(objectToWorld);
    // for a rigid camera its inverse is its transpose.
    const float3x3 worldToView = mul(objectToView, inverse3(objectToWorld));
    worldNormals[at] = float4(normalize(mul(transpose(worldToView), n)), 0.0);
    for (uint k = 0; k < params.primvarSlots; ++k) {
        primvarsOut[at * params.primvarSlots + k] = surfacePrimvar(s, primvarSlotOf[k], float4(0.0));
    }
}

float3x3 inverse3(float3x3 m) {
    const float3 r0 = cross(m[1], m[2]);
    const float3 r1 = cross(m[2], m[0]);
    const float3 r2 = cross(m[0], m[1]);
    const float det = dot(m[0], r0);
    const float invDet = abs(det) > 1e-30 ? 1.0 / det : 0.0;
    return transpose(float3x3(r0, r1, r2)) * invDet;
}
