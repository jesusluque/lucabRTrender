// Copyright (c) 2026 lucabRTrender contributors.
//
// Motion blur's buckets: the shutter cut into `buckets` slices, and for each
// slice a copy of every moving instance's record at that slice's time, and
// of every deforming mesh's positions. A path's samples draw a time, take the
// slice it falls in, and trace the structure that slice's records answer to
// (their mask bit). Between the two shutter samples everything is linear --
// a transform's rows as well as a point -- so two buckets are exact for a
// translation and an approximation for a turn.
module motion;
import scene_types;

struct MotionInput {
    float4 world0Lo; float4 world1Lo; float4 world2Lo;   // the transform at shutter open
    float4 world0Hi; float4 world1Hi; float4 world2Hi;   // and at shutter close
    float4 colour;
    uint   mesh;
    uint   primId;
    uint   instanceId;
    uint   flags;
    uint   categoriesLo;
    uint   categoriesHi;
    uint   deforms;      // 1: its mesh's positions are per bucket, at pointsStride apart
    uint   pad0;
    float  time0;        // when the two transforms (and meshes) are, in the shutter's units
    float  time1;
    float  pad1; float pad2;
};

struct MotionParams {
    uint  count;         // instances
    uint  buckets;
    uint  first;         // the first bucket record: bucket b's copy of instance i is at first + b * count + i
    uint  pointsStride;  // positions a bucket, in the pool
    float open;          // the shutter, in the same units as a sample's time
    float close;
    float pad0; float pad1;
    // world to view, 3x4 rows
    float v00; float v01; float v02; float v03;
    float v10; float v11; float v12; float v13;
    float v20; float v21; float v22; float v23;
};

StructuredBuffer<MotionInput>       motionInputs;
RWStructuredBuffer<InstanceRecord>  records;
ConstantBuffer<MotionParams>        params;

// positionsLerp
StructuredBuffer<float4>            pointsLo;
StructuredBuffer<float4>            pointsHi;
RWStructuredBuffer<float4>          positions;
struct LerpParams {
    uint  count;         // the mesh's points
    uint  firstPoint;    // in the pool
    uint  buckets;
    uint  pointsStride;
    float time0;         // when pointsLo and pointsHi are
    float time1;
    float open;          // the shutter
    float close;
};
ConstantBuffer<LerpParams>          lerpParams;

/// Where bucket b's centre falls between two samples at time0 and time1,
/// the shutter running from open to close: the lerp factor.
float bucketFactor(uint b, uint buckets, float open, float close, float time0, float time1) {
    const float at = open + (float(b) + 0.5) / float(buckets) * (close - open);
    const float span = time1 - time0;
    return abs(span) > 1.0e-12 ? (at - time0) / span : 0.0;
}

float3x3 inverse3(float3x3 m) {
    const float3 c0 = float3(m[0][0], m[1][0], m[2][0]);
    const float3 c1 = float3(m[0][1], m[1][1], m[2][1]);
    const float3 c2 = float3(m[0][2], m[1][2], m[2][2]);
    const float3 r0 = cross(c1, c2);
    const float3 r1 = cross(c2, c0);
    const float3 r2 = cross(c0, c1);
    const float det = dot(c0, r0);
    const float inv = abs(det) > 1.0e-30 ? 1.0 / det : 0.0;
    return float3x3(r0 * inv, r1 * inv, r2 * inv);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void motionRecords(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count * params.buckets) {
        return;
    }
    const uint b = k / params.count;
    const uint i = k - b * params.count;
    const MotionInput m = motionInputs[i];
    const float t = bucketFactor(b, params.buckets, params.open, params.close, m.time0, m.time1);
    const float3x4 world = float3x4(lerp(m.world0Lo, m.world0Hi, t), lerp(m.world1Lo, m.world1Hi, t),
                                    lerp(m.world2Lo, m.world2Hi, t));
    const float3x4 worldToView = float3x4(params.v00, params.v01, params.v02, params.v03, params.v10, params.v11,
                                          params.v12, params.v13, params.v20, params.v21, params.v22, params.v23);
    const float4x4 w4 = float4x4(world[0], world[1], world[2], float4(0.0, 0.0, 0.0, 1.0));
    const float4x4 v4 = mul(float4x4(worldToView[0], worldToView[1], worldToView[2], float4(0.0, 0.0, 0.0, 1.0)), w4);
    const float3x3 normal = transpose(inverse3(float3x3(v4[0].xyz, v4[1].xyz, v4[2].xyz)));
    InstanceRecord r;
    r.view0 = v4[0];
    r.view1 = v4[1];
    r.view2 = v4[2];
    r.normal0 = float4(normal[0], 0.0);
    r.normal1 = float4(normal[1], 0.0);
    r.normal2 = float4(normal[2], 0.0);
    r.world0 = world[0];
    r.world1 = world[1];
    r.world2 = world[2];
    r.colour = m.colour;
    r.mesh = m.mesh;
    r.primId = m.primId;
    r.instanceId = m.instanceId;
    r.flags = m.flags;
    r.categoriesLo = m.categoriesLo;
    r.categoriesHi = m.categoriesHi;
    r.mask = 1u << b;
    r.pointsOffset = m.deforms != 0 ? b * params.pointsStride : 0u;
    records[params.first + k] = r;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void positionsLerp(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= lerpParams.count * lerpParams.buckets) {
        return;
    }
    const uint b = k / lerpParams.count;
    const uint i = k - b * lerpParams.count;
    const float t = bucketFactor(b, lerpParams.buckets, lerpParams.open, lerpParams.close, lerpParams.time0,
                                 lerpParams.time1);
    positions[b * lerpParams.pointsStride + lerpParams.firstPoint + i] = lerp(pointsLo[i], pointsHi[i], t);
}
