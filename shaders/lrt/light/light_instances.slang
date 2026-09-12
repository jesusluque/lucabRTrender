// Copyright (c) 2026 lucabRTrender contributors.
//
// An instanced light's records: the prototype was copied once per instance
// by the host (bookkeeping), and each copy's placement is written here as
// the instance's transform times the prototype's own, on the device, from
// the rows world::Instancing composed -- the same rows a mesh instance is
// placed by.
module light_instances;
import lrt.light.lights;

RWStructuredBuffer<LightRecord> records;
StructuredBuffer<float4>        chainRows;   // 3 float4 rows an instance
uniform uint                    base;        // the first of this light's records
uniform uint                    count;

[shader("compute")]
[numthreads(64, 1, 1)]
void lightInstances(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count) {
        return;
    }
    LightRecord r = records[base + i];
    const float4 a0 = chainRows[i * 3 + 0];
    const float4 a1 = chainRows[i * 3 + 1];
    const float4 a2 = chainRows[i * 3 + 2];
    const float4 b0 = r.row0;
    const float4 b1 = r.row1;
    const float4 b2 = r.row2;
    // (A B) for 3x4 affine rows: the 3x3 product, and A's rotation of B's
    // translation plus A's own.
    const float3 c0 = float3(b0.x, b1.x, b2.x);
    const float3 c1 = float3(b0.y, b1.y, b2.y);
    const float3 c2 = float3(b0.z, b1.z, b2.z);
    const float3 ct = float3(b0.w, b1.w, b2.w);
    r.row0 = float4(dot(a0.xyz, c0), dot(a0.xyz, c1), dot(a0.xyz, c2), dot(a0.xyz, ct) + a0.w);
    r.row1 = float4(dot(a1.xyz, c0), dot(a1.xyz, c1), dot(a1.xyz, c2), dot(a1.xyz, ct) + a1.w);
    r.row2 = float4(dot(a2.xyz, c0), dot(a2.xyz, c1), dot(a2.xyz, c2), dot(a2.xyz, ct) + a2.w);
    records[base + i] = r;
}
