// Copyright (c) 2026 lucabRTrender contributors.
//
// Skinning, as usdSkelImaging's own kernel does it (skinning.glslfx), so a
// Hydra host that runs the ext computation itself and this one read the
// same: blend shape offsets summed into the rest points by their sub-shape
// weights, then either linear blend skinning -- each influence's skinning
// transform applied to the point in bind space and weighed -- or dual
// quaternion skinning, the influences' dual quaternions blended on the
// pivot's hemisphere and normalised, with any scale applied linearly; then
// skeleton space to the prim's own through the common space.
module skinning;

struct SkinningParams {
    uint  count;                  // points
    uint  numInfluences;          // per point (0: no joints, only shapes)
    uint  constantInfluences;     // 1: the same influences for every point
    uint  numOffsetRanges;        // points with a blend shape range
    uint  method;                 // 0 linear, 1 dual quaternion
    uint  hasScaleXforms;         // dual quaternion: scale per joint, applied linearly
    uint  pad0; uint pad1;
    float4 geomBind0; float4 geomBind1; float4 geomBind2; float4 geomBind3;   // rows of a 4x4
    float4 skelToWorld0; float4 skelToWorld1; float4 skelToWorld2; float4 skelToWorld3;
    float4 worldToPrim0; float4 worldToPrim1; float4 worldToPrim2; float4 worldToPrim3;
};

StructuredBuffer<float4>   restPoints;
StructuredBuffer<float4>   blendShapeOffsets;      // xyz offset, w sub-shape index
StructuredBuffer<uint2>    blendShapeOffsetRanges; // per point: [start, end) into the offsets
StructuredBuffer<float>    blendShapeWeights;      // per sub-shape
StructuredBuffer<float2>   influences;             // (joint, weight)
StructuredBuffer<float4>   skinningXforms;         // 4 rows a joint (row-major 4x4)
StructuredBuffer<float4>   skinningDualQuats;      // real, dual a joint (x, y, z, w)
StructuredBuffer<float4>   skinningScaleXforms;    // 3 rows a joint (3x3 in the xyz)
RWStructuredBuffer<float4> skinned;
ConstantBuffer<SkinningParams> params;

static const float kEps = 1.0e-5;
static const float kNormEps = 1.0e-10;

float4x4 rows4(float4 a, float4 b, float4 c, float4 d) {
    return float4x4(a, b, c, d);
}

float4x4 jointXform(uint j) {
    return rows4(skinningXforms[j * 4], skinningXforms[j * 4 + 1], skinningXforms[j * 4 + 2],
                 skinningXforms[j * 4 + 3]);
}

float3 transformByQuaternion(float4 q, float3 v) {
    const float r = q.w;
    const float3 i = q.xyz;
    const float3 i2 = float3(r * v.x + (i.y * v.z - i.z * v.y), r * v.y + (i.z * v.x - i.x * v.z),
                             r * v.z + (i.x * v.y - i.y * v.x));
    return float3(v.x + 2.0 * (i.y * i2.z - i.z * i2.y), v.y + 2.0 * (i.z * i2.x - i.x * i2.z),
                  v.z + 2.0 * (i.x * i2.y - i.y * i2.x));
}

float3 dualQuaternionTranslation(float4 real, float4 dual) {
    const float rw = real.w;
    const float3 ri = real.xyz;
    const float dw = dual.w;
    const float3 di = dual.xyz;
    return -2.0 * float3(dw * ri.x - rw * di.x + di.y * ri.z - di.z * ri.y,
                         dw * ri.y - rw * di.y + di.z * ri.x - di.x * ri.z,
                         dw * ri.z - rw * di.z + di.x * ri.y - di.y * ri.x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void skinPoints(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    float3 rest = restPoints[i].xyz;
    if (i < params.numOffsetRanges) {
        const uint2 range = blendShapeOffsetRanges[i];
        for (uint k = range.x; k < range.y; ++k) {
            const float4 offset = blendShapeOffsets[k];
            rest += offset.xyz * blendShapeWeights[uint(offset.w)];
        }
    }
    float3 p = rest;
    if (params.numInfluences > 0) {
        const float4x4 geomBind = rows4(params.geomBind0, params.geomBind1, params.geomBind2, params.geomBind3);
        const float4 init = mul(geomBind, float4(rest, 1.0));
        const uint offset = params.constantInfluences != 0 ? 0u : params.numInfluences * i;
        if (params.method == 0) {
            float3 sum = float3(0.0);
            for (uint k = 0; k < params.numInfluences; ++k) {
                const float2 influence = influences[offset + k];
                if (influence.y > kEps) {
                    sum += (mul(jointXform(uint(influence.x)), init) * influence.y).xyz;
                }
            }
            p = sum;
        } else {
            // The pivot: the heaviest influence's real part.
            float4 pivot = float4(0.0);
            float heaviest = -1.0;
            for (uint k = 0; k < params.numInfluences; ++k) {
                const float2 influence = influences[offset + k];
                if (heaviest < influence.y) {
                    heaviest = influence.y;
                    pivot = skinningDualQuats[uint(influence.x) * 2];
                }
            }
            float3 scaled = float3(0.0);
            float4 real = float4(0.0);
            float4 dual = float4(0.0);
            for (uint k = 0; k < params.numInfluences; ++k) {
                const float2 influence = influences[offset + k];
                float w = influence.y;
                if (w > kEps) {
                    const uint j = uint(influence.x);
                    if (params.hasScaleXforms != 0) {
                        const float3x3 scale = float3x3(skinningScaleXforms[j * 3].xyz, skinningScaleXforms[j * 3 + 1].xyz,
                                                        skinningScaleXforms[j * 3 + 2].xyz);
                        scaled += mul(scale, init.xyz) * w;
                    }
                    const float4 jr = skinningDualQuats[j * 2];
                    const float4 jd = skinningDualQuats[j * 2 + 1];
                    if (dot(jr, pivot) < 0.0) {
                        w = -w;
                    }
                    real += jr * w;
                    dual += jd * w;
                }
            }
            const float realLength = length(real);
            if (realLength < kNormEps) {
                real = float4(0.0, 0.0, 0.0, 1.0);
                dual = float4(0.0);
            } else {
                real /= realLength;
                dual /= realLength;
                dual -= dot(real, dual) * real;
            }
            const float3 base = params.hasScaleXforms != 0 ? scaled : init.xyz;
            p = transformByQuaternion(real, base) + dualQuaternionTranslation(real, dual);
        }
        const float4x4 skelToWorld = rows4(params.skelToWorld0, params.skelToWorld1, params.skelToWorld2,
                                           params.skelToWorld3);
        const float4x4 worldToPrim = rows4(params.worldToPrim0, params.worldToPrim1, params.worldToPrim2,
                                           params.worldToPrim3);
        p = mul(worldToPrim, mul(skelToWorld, float4(p, 1.0))).xyz;
    }
    skinned[i] = float4(p, 1.0);
}
