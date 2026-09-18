// Copyright (c) 2026 lucabRTrender contributors.
//
// The splat skinner's inputs and its answer, both made and checked on the
// device. What it pins is the two invariants a skinned cloud stands on: a
// skeleton at rest leaves every gaussian where it was, and a skeleton that
// moves rigidly carries every gaussian rigidly -- its position by the
// transform, its frame turned with it, and its size not at all.
module splat_skin_check;
import lrt.common.packing;

struct SkinCheckParams {
    uint   count;
    uint   perSplat;
    uint   joints;
    uint   pad0;
    float4 rotation;      ///< the rigid turn, as (x, y, z, w)
    float4 translation;   ///< and the slide, in xyz
    float  tolerance;
    uint   pad1;
    uint   pad2;
    uint   pad3;
};

RWStructuredBuffer<float4> restPositions;
RWStructuredBuffer<uint>   restShape;
RWStructuredBuffer<float2> influences;
RWStructuredBuffer<float4> xforms;
StructuredBuffer<float4>   positions;    ///< what the skinner wrote
StructuredBuffer<uint>     shape;
RWStructuredBuffer<uint>   counts;       ///< [compared, moved, resized, turned]
ConstantBuffer<SkinCheckParams> params;

float3x3 turnOf(float4 q) {
    return axesOfQuaternion(q);
}

/// A cloud of gaussians on a helix, each turned differently and each a
/// different size, so that nothing in the check can pass by accident.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatSkinMake(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const float t = float(at) / max(float(params.count), 1.0);
    const float angle = t * 17.0;
    restPositions[at] = float4(cos(angle) * (1.0 + t), t * 4.0 - 2.0, sin(angle) * (1.0 + t), 0.5 + 0.5 * t);

    // A rotation of its own: about an axis that wanders with the index.
    const float3 axis = normalize(float3(0.3 + t, 1.0 - t, 0.2 + t * t));
    const float half = angle * 0.5;
    const float4 q = float4(axis * sin(half), cos(half));
    const uint row = at * 4;
    restShape[row] = encodeQuaternion(q);
    restShape[row + 1] = packHalves(log(0.05 + 0.2 * t), log(0.03 + 0.3 * (1.0 - t)));
    restShape[row + 2] = packHalves(log(1.0e-5), 0.4 + 0.2 * t);
    restShape[row + 3] = packHalves(0.3, 0.6 - 0.2 * t);

    // Every gaussian on the one joint, with all of the weight: what the check
    // is about is the transform, not the blend.
    for (uint k = 0; k < params.perSplat; ++k) {
        influences[at * params.perSplat + k] = float2(0.0, k == 0 ? 1.0 : 0.0);
    }
}

/// Every joint given the same rigid transform, as rows of a 4x4 with the
/// translation in the last column -- the layout the kernel multiplies.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatSkinJoints(uint3 tid: SV_DispatchThreadID) {
    const uint joint = tid.x;
    if (joint >= params.joints) {
        return;
    }
    const float3x3 turn = turnOf(params.rotation);
    xforms[joint * 4 + 0] = float4(turn[0][0], turn[0][1], turn[0][2], params.translation.x);
    xforms[joint * 4 + 1] = float4(turn[1][0], turn[1][1], turn[1][2], params.translation.y);
    xforms[joint * 4 + 2] = float4(turn[2][0], turn[2][1], turn[2][2], params.translation.z);
    xforms[joint * 4 + 3] = float4(0.0, 0.0, 0.0, 1.0);
}

/// What the skinner answered, against what a rigid transform must give.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatSkinCompare(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float3x3 turn = turnOf(params.rotation);
    const float4 rest = restPositions[at];
    const float3 want = mul(turn, rest.xyz) + params.translation.xyz;
    const float4 got = positions[at];
    if (length(got.xyz - want) > params.tolerance || abs(got.w - rest.w) > 1.0e-6) {
        InterlockedAdd(counts[1], 1u);
    }
    const uint row = at * 4;
    // The two sizes in the plane are the ones that move; the third and the
    // colour are copied, so they are checked as words.
    const float2 wantScale = float2(lowHalf(restShape[row + 1]), highHalf(restShape[row + 1]));
    const float2 gotScale = float2(lowHalf(shape[row + 1]), highHalf(shape[row + 1]));
    if (abs(exp(gotScale.x) - exp(wantScale.x)) > params.tolerance ||
        abs(exp(gotScale.y) - exp(wantScale.y)) > params.tolerance ||
        shape[row + 2] != restShape[row + 2] || shape[row + 3] != restShape[row + 3]) {
        InterlockedAdd(counts[2], 1u);
    }
    // THE FRAME, compared as axes and not as a quaternion: `q` and `-q` are
    // the same turn, and the axes have no such ambiguity.
    const float3x3 was = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float3x3 is = axesOfQuaternion(decodeQuaternion(shape[row]));
    const float3x3 wanted = mul(turn, was);
    float worst = 0.0;
    for (uint c = 0; c < 3; ++c) {
        const float3 a = float3(is[0][c], is[1][c], is[2][c]);
        const float3 b = float3(wanted[0][c], wanted[1][c], wanted[2][c]);
        worst = max(worst, length(a - b));
    }
    if (worst > params.tolerance) {
        InterlockedAdd(counts[3], 1u);
    }
}
