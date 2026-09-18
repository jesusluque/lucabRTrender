// Copyright (c) 2026 lucabRTrender contributors.
//
// The mesh2splat effect's inputs and its answers, both on the device.
//
// The rule this file exists for: a test generates what it feeds the effect
// with a kernel and checks what came back with a kernel, and only counters
// cross to the processor. So the quad below is written here rather than in
// C++, and every claim about the gaussians -- how many, where, how wide, which
// way round, what colour -- is a counter of violations that must come back
// zero.
module mesh2splat_check;

struct CheckParams {
    uint  triangles;      ///< of the stream written or read
    uint  meshWidth;      ///< the mesh picture: entries a row
    uint  meshStride;     ///< pixels a row
    uint  recordPixels;   ///< float4 a splat in the records picture

    uint  splats;         ///< how many the effect said it wrote
    uint  dstWidth;
    uint  dstStride;
    uint  resolution;

    float sigma;
    float flatness;
    float tolerance;
    float opacity;

    float4 colour;        ///< what every splat's colour should be
    float4 axis;          ///< the plane's normal: the short axis of every frame

    uint  albedoWidth;    ///< a checker written into the albedo picture
    uint  albedoHeight;
    uint  albedoStride;
    uint  checker;        ///< cells across the checker, 0 for none
    float joint;          ///< the joint every gaussian should be carried by
    uint  pad0;
    uint  pad1;
};

RWStructuredBuffer<float4> mesh;      ///< the stream, written by `m2sQuad`
RWStructuredBuffer<float4> albedo;    ///< the checker, written by `m2sChecker`
StructuredBuffer<float4>   records;   ///< what the effect wrote
RWStructuredBuffer<uint>   counts;    ///< [outside, wide, turned, coloured, opaque]
ConstantBuffer<CheckParams> params;

uint meshAt(uint entry) {
    return (entry / params.meshWidth) * params.meshStride + entry % params.meshWidth;
}

uint recordAt(uint slot, uint component) {
    const uint entry = slot * params.recordPixels + component;
    return (entry / params.dstWidth) * params.dstStride + entry % params.dstWidth;
}

/// The unit quad in z = 0, as two triangles wound counter-clockwise, with
/// texture coordinates running 0 to 1 across it. Its box is exactly one by
/// one, so a cell of the projection grid is `1 / resolution` wide and a
/// gaussian on it is `sigma / resolution`: a closed form to check against.
[shader("compute")]
[numthreads(1, 1, 1)]
void m2sQuad(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const float3 corners[4] = {float3(0.0, 0.0, 0.0), float3(1.0, 0.0, 0.0), float3(1.0, 1.0, 0.0),
                               float3(0.0, 1.0, 0.0)};
    const uint3 faces[2] = {uint3(0, 1, 2), uint3(0, 2, 3)};
    for (uint t = 0; t < 2; ++t) {
        for (uint k = 0; k < 3; ++k) {
            const float3 p = corners[faces[t][k]];
            mesh[meshAt((t * 3 + k) * 2)] = float4(p, p.x);
            mesh[meshAt((t * 3 + k) * 2 + 1)] = float4(0.0, 0.0, 1.0, p.y);
        }
    }
}

/// A checker in the albedo picture: red where the cell is even, green where
/// it is odd, so a sample that lands in the wrong cell is not a near miss.
[shader("compute")]
[numthreads(8, 8, 1)]
void m2sChecker(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.albedoWidth || tid.y >= params.albedoHeight) {
        return;
    }
    const uint cellX = tid.x * params.checker / max(params.albedoWidth, 1u);
    const uint cellY = tid.y * params.checker / max(params.albedoHeight, 1u);
    const bool even = ((cellX + cellY) & 1u) == 0u;
    albedo[tid.y * params.albedoStride + tid.x] =
        even ? float4(1.0, 0.0, 0.0, 1.0) : float4(0.0, 1.0, 0.0, 1.0);
}

/// Turns `v` by the quaternion the records carry: w, x, y, z.
float3 turn(float4 q, float3 v) {
    const float3 u = float3(q.y, q.z, q.w);
    return v * (q.x * q.x - dot(u, u)) + u * (2.0 * dot(u, v)) + cross(u, v) * (2.0 * q.x);
}

/// Every claim the quad makes, counted.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCheckQuad(uint3 tid: SV_DispatchThreadID) {
    const uint splat = tid.x;
    if (splat >= params.splats) {
        return;
    }
    const float4 position = records[recordAt(splat, 0)];
    const float4 scale = records[recordAt(splat, 1)];
    const float4 rotation = records[recordAt(splat, 2)];
    const float4 colour = records[recordAt(splat, 3)];

    // On the quad, and inside it.
    if (abs(dot(position.xyz, params.axis.xyz)) > params.tolerance || position.x < -params.tolerance ||
        position.x > 1.0 + params.tolerance || position.y < -params.tolerance ||
        position.y > 1.0 + params.tolerance) {
        InterlockedAdd(counts[0], 1u);
    }
    // As wide as a cell of the grid: `|Ju| * sigma / resolution` with
    // `|Ju| = |Jv| = 1`, which is the whole of mesh2splat's sizing.
    const float want = params.sigma / float(max(params.resolution, 1u));
    // And as thick as the fraction of that the third size asks for.
    const float wantThird = want * params.flatness;
    if (abs(scale.x - want) > params.tolerance * want || abs(scale.y - want) > params.tolerance * want ||
        abs(scale.z - wantThird) > params.tolerance * max(wantThird, 1.0e-9)) {
        InterlockedAdd(counts[1], 1u);
    }
    // Flat against the quad: the short axis of the frame is the plane's own.
    const float3 shortAxis = turn(rotation, float3(0.0, 0.0, 1.0));
    if (abs(dot(shortAxis, params.axis.xyz)) < 1.0 - params.tolerance) {
        InterlockedAdd(counts[2], 1u);
    }
    if (any(abs(colour.xyz - params.colour.xyz) > params.tolerance)) {
        InterlockedAdd(counts[3], 1u);
    }
    if (abs(position.w - params.opacity) > params.tolerance) {
        InterlockedAdd(counts[4], 1u);
    }
}

/// The colour of every splat against the checker under its own texture
/// coordinate, which the sixth entry of a record carries.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCheckColour(uint3 tid: SV_DispatchThreadID) {
    const uint splat = tid.x;
    if (splat >= params.splats) {
        return;
    }
    const float4 colour = records[recordAt(splat, 3)];
    const float4 pbr = records[recordAt(splat, 5)];
    const float2 uv = pbr.zw;
    const uint cellX = uint(clamp(uv.x, 0.0, 0.999) * float(params.checker));
    const uint cellY = uint(clamp(uv.y, 0.0, 0.999) * float(params.checker));
    const bool even = ((cellX + cellY) & 1u) == 0u;
    const float3 want = even ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    // A gaussian that stands within half a texel of a cell's edge is sampled
    // across it, and bilinear filtering is what puts it between the two
    // colours. Only the ones well inside a cell are claims about the sampling.
    const float2 inCell = frac(uv * float(params.checker));
    const bool wellInside = all(inCell > 0.15) && all(inCell < 0.85);
    if (wellInside && any(abs(colour.xyz - want) > params.tolerance)) {
        InterlockedAdd(counts[3], 1u);
    }
}


/// The joints picture for the quad: every corner carried entirely by one
/// joint. Written by a kernel like the quad itself, so the test makes its
/// inputs the way it checks its outputs.
[shader("compute")]
[numthreads(1, 1, 1)]
void m2sInfluences(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    for (uint triangle = 0; triangle < 2; ++triangle) {
        for (uint corner = 0; corner < 3; ++corner) {
            const uint at = triangle * 6 + corner * 2;
            mesh[meshAt(at)] = float4(params.joint, 1.0, 0.0, 0.0);
            mesh[meshAt(at + 1)] = float4(0.0, 0.0, 0.0, 0.0);
        }
    }
}

/// THE JOINTS A GAUSSIAN CAME OUT CARRIED BY.
///
/// The conversion blends the three corners' influences by the barycentric
/// coordinates of the cell the gaussian stands in. Where every corner names
/// the same joint with all of its weight, every blend is that joint with all
/// of its weight whatever the coordinates are -- which is the case this
/// pins, because it is the one that catches the plumbing: a picture that did
/// not arrive, an entry addressed wrong, a weight left unnormalised.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCheckInfluences(uint3 tid: SV_DispatchThreadID) {
    const uint splat = tid.x;
    if (splat >= params.splats) {
        return;
    }
    const float4 low = records[recordAt(splat, 6)];
    const float4 high = records[recordAt(splat, 7)];
    // The first slot carries it all, and the other three carry nothing.
    if (abs(low.x - params.joint) > 1.0e-6 || abs(low.y - 1.0) > params.tolerance) {
        InterlockedAdd(counts[0], 1u);
    }
    if (low.w != 0.0 || high.y != 0.0 || high.w != 0.0) {
        InterlockedAdd(counts[1], 1u);
    }
    // And whatever the split, the weights are a whole.
    const float sum = low.y + low.w + high.y + high.w;
    if (abs(sum - 1.0) > params.tolerance) {
        InterlockedAdd(counts[2], 1u);
    }
}
