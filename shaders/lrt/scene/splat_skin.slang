// Copyright (c) 2026 lucabRTrender contributors.
//
// A CLOUD CARRIED BY A SKELETON, one thread a gaussian.
//
// The cloud is built once, in the bind pose, and each gaussian keeps the four
// joints that move it (mesh2splat's conversion blends them from the corners of
// the triangle it stands on). Here those joints are applied: the position by
// exactly the linear blend `geom/skinning.slang` does for a mesh point, and
// the frame by the same chain's linear part -- which is what makes a gaussian
// stretch with its triangle instead of sliding along beside it.
//
// What it is for: an animated cloud that is three megabytes rather than three
// hundred, exact at every instant of the range rather than at the frames
// somebody sampled, and whose buffers do not change identity from one frame to
// the next -- so a ray tracer refits rather than rebuilds.
module splat_skin;
import lrt.common.packing;

struct SplatSkinParams {
    uint   count;          ///< gaussians
    uint   perSplat;       ///< influences each carries
    uint   pad0;
    uint   pad1;
    float4 geomBind0;      ///< rows of a 4x4, as geom/skinning.slang takes them
    float4 geomBind1;
    float4 geomBind2;
    float4 geomBind3;
    float4 skelToWorld0;
    float4 skelToWorld1;
    float4 skelToWorld2;
    float4 skelToWorld3;
    float4 worldToPrim0;
    float4 worldToPrim1;
    float4 worldToPrim2;
    float4 worldToPrim3;
};

StructuredBuffer<float4>   restPositions;   // xyz, w opacity
StructuredBuffer<uint>     restShape;       // 4 words: rotation, log scales, base colour
StructuredBuffer<float2>   influences;      // (joint, weight), perSplat a gaussian
StructuredBuffer<float4>   skinningXforms;  // 4 rows a joint (row-major 4x4)
RWStructuredBuffer<float4> positions;
RWStructuredBuffer<uint>   shape;
ConstantBuffer<SplatSkinParams> params;

static const float kEps = 1.0e-5;

float4x4 rows4(float4 a, float4 b, float4 c, float4 d) {
    return float4x4(a, b, c, d);
}

float3x3 linearOf(float4x4 m) {
    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatSkin(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const uint row = at * 4;
    const float4 rest = restPositions[at];
    const float4x4 geomBind = rows4(params.geomBind0, params.geomBind1, params.geomBind2, params.geomBind3);
    const float4x4 skelToWorld =
        rows4(params.skelToWorld0, params.skelToWorld1, params.skelToWorld2, params.skelToWorld3);
    const float4x4 worldToPrim =
        rows4(params.worldToPrim0, params.worldToPrim1, params.worldToPrim2, params.worldToPrim3);

    // The blend, as a whole matrix: the position needs it applied to a point
    // and the frame needs its linear part applied to two axes, and taking one
    // sum for both is what keeps them from disagreeing.
    const float4 bound = mul(geomBind, float4(rest.xyz, 1.0));
    float4   moved = float4(0.0, 0.0, 0.0, 0.0);
    float3x3 blended = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    float    total = 0.0;
    const uint offset = at * params.perSplat;
    for (uint k = 0; k < params.perSplat; ++k) {
        const float2 influence = influences[offset + k];
        const float weight = influence.y;
        if (!(weight > kEps)) {
            continue;
        }
        const uint joint = uint(influence.x);
        const float4x4 xform = rows4(skinningXforms[joint * 4], skinningXforms[joint * 4 + 1],
                                     skinningXforms[joint * 4 + 2], skinningXforms[joint * 4 + 3]);
        moved += mul(xform, bound) * weight;
        blended += linearOf(xform) * weight;
        total += weight;
    }
    if (!(total > kEps)) {
        // Carried by nothing: it stays where the bind pose left it, taken
        // through the same two transforms so it lands in the same space.
        moved = bound;
        blended = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    }
    const float3 world = mul(skelToWorld, float4(moved.xyz, 1.0)).xyz;
    positions[at] = float4(mul(worldToPrim, float4(world, 1.0)).xyz, rest.w);

    // THE FRAME. The chain's linear part carries the two axes the gaussian
    // spreads along; the third is a disc's and stays as it was. Squared up
    // again afterwards, because a joint may shear where a rotation would not.
    const float3x3 chain = mul(linearOf(worldToPrim), mul(linearOf(skelToWorld), mul(blended, linearOf(geomBind))));
    const float3 scale = exp(float3(lowHalf(restShape[row + 1]), highHalf(restShape[row + 1]),
                                    lowHalf(restShape[row + 2])));
    const float3x3 axes = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float3 restU = float3(axes[0][0], axes[1][0], axes[2][0]);
    const float3 restV = float3(axes[0][1], axes[1][1], axes[2][1]);
    const float3 movedU = mul(chain, restU * scale.x);
    const float3 movedV = mul(chain, restV * scale.y);
    const float lengthU = length(movedU);
    const float lengthV = length(movedV);
    float3 u = lengthU > 1.0e-20 ? movedU / lengthU : restU;
    float3 v = lengthV > 1.0e-20 ? movedV / lengthV : restV;
    v = v - u * dot(u, v);
    const float squared = length(v);
    v = squared > 1.0e-10 ? v / squared : normalize(cross(float3(0.0, 0.0, 1.0), u) + float3(1.0e-6, 0.0, 0.0));
    const float3 n = cross(u, v);
    const float3x3 frame = float3x3(u.x, v.x, n.x, u.y, v.y, n.y, u.z, v.z, n.z);

    shape[row] = encodeQuaternion(quaternionOfAxes(frame));
    shape[row + 1] = packHalves(log(max(lengthU > 1.0e-20 ? lengthU : scale.x, 1.0e-20)),
                                log(max(lengthV > 1.0e-20 ? lengthV : scale.y, 1.0e-20)));
    // The third size and the base colour are the conversion's and stay: only
    // the two halves of this word that hold the scale are rewritten.
    shape[row + 2] = restShape[row + 2];
    shape[row + 3] = restShape[row + 3];
}
