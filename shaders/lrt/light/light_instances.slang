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

/// (A B) for 3x4 affine rows: the 3x3 product, and A's rotation of B's
/// translation plus A's own.
void composeRows(float4 a0, float4 a1, float4 a2, inout float4 b0, inout float4 b1, inout float4 b2) {
    const float3 c0 = float3(b0.x, b1.x, b2.x);
    const float3 c1 = float3(b0.y, b1.y, b2.y);
    const float3 c2 = float3(b0.z, b1.z, b2.z);
    const float3 ct = float3(b0.w, b1.w, b2.w);
    b0 = float4(dot(a0.xyz, c0), dot(a0.xyz, c1), dot(a0.xyz, c2), dot(a0.xyz, ct) + a0.w);
    b1 = float4(dot(a1.xyz, c0), dot(a1.xyz, c1), dot(a1.xyz, c2), dot(a1.xyz, ct) + a1.w);
    b2 = float4(dot(a2.xyz, c0), dot(a2.xyz, c1), dot(a2.xyz, c2), dot(a2.xyz, ct) + a2.w);
}

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
    composeRows(a0, a1, a2, r.row0, r.row1, r.row2);
    records[base + i] = r;
}

// A moving instanced light's samples: the host copied the prototype's rows
// at the two samples into each copy's 26 floats (24 rows, two times), and
// here each is placed by the chain's rows at the same sample.
RWStructuredBuffer<float> motionValues;
StructuredBuffer<float4>  chainRowsStart;
StructuredBuffer<float4>  chainRowsEnd;
uniform uint              motionAt;      // the first copy's 26 floats

[shader("compute")]
[numthreads(64, 1, 1)]
void lightInstancesMotion(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count) {
        return;
    }
    const uint at = motionAt + i * 26;
    for (uint s = 0; s < 2; ++s) {
        const uint rows = at + s * 12;
        float4 b0 = float4(motionValues[rows], motionValues[rows + 1], motionValues[rows + 2], motionValues[rows + 3]);
        float4 b1 = float4(motionValues[rows + 4], motionValues[rows + 5], motionValues[rows + 6], motionValues[rows + 7]);
        float4 b2 = float4(motionValues[rows + 8], motionValues[rows + 9], motionValues[rows + 10], motionValues[rows + 11]);
        if (s == 0) {
            composeRows(chainRowsStart[i * 3], chainRowsStart[i * 3 + 1], chainRowsStart[i * 3 + 2], b0, b1, b2);
        } else {
            composeRows(chainRowsEnd[i * 3], chainRowsEnd[i * 3 + 1], chainRowsEnd[i * 3 + 2], b0, b1, b2);
        }
        for (uint c = 0; c < 4; ++c) {
            motionValues[rows + c] = b0[c];
            motionValues[rows + 4 + c] = b1[c];
            motionValues[rows + 8 + c] = b2[c];
        }
    }
}
