// Copyright (c) 2026 lucabRTrender contributors.
//
// A mesh's triangles as the stream an AOFX effect can be handed.
//
// An effect is given pictures, and a mesh is not one. So it travels as a
// picture of numbers: six `float4` entries a triangle, `(position, u)` and
// `(normal, v)` for each of its three corners, in world space. That is the
// whole of what mesh2splat's conversion reads, and writing it here rather
// than in the host is what keeps the rule: the triangulation is a kernel's,
// the transform is a kernel's, and the primvar lookup is a kernel's.
//
// One thread a triangle. Positions come from the mesh's own point buffer and
// the indices `geom::MeshBuilder` wrote; normals and texture coordinates come
// from primvars whose interpolation the host passes in, resolved exactly as
// `world/primvars.slang` resolves them -- per point, per corner, per authored
// face or one for the mesh.
module mesh_pack;

struct PackParams {
    uint  triangles;      ///< of this mesh
    uint  destFirst;      ///< where its entries start, in float4 entries
    uint  normalMode;     ///< an interpolation, or kNoPrimvar
    uint  normalCount;

    uint  uvMode;
    uint  uvCount;
    uint  points;
    uint  corners;

    uint  destWidth;      ///< entries a row of the destination picture
    uint  destStride;     ///< pixels a row of it, which is padded
    uint  chunkFirst;     ///< where this mesh's chunks start, for the fold
    uint  pad1;

    float4 toWorld0;      ///< object to world, three rows of four
    float4 toWorld1;
    float4 toWorld2;
    float4 normalTo0;     ///< its inverse transpose, for the normals
    float4 normalTo1;
    float4 normalTo2;

    uint  entries;        ///< float4 entries the whole stream holds
    uint  chunkSize;      ///< for the bounds fold below
    uint  chunkCount;
    uint  pad0;
};

static const uint kConstant = 0;
static const uint kUniform = 1;
static const uint kVarying = 2;
static const uint kVertex = 3;
static const uint kFaceVarying = 4;
static const uint kNoPrimvar = 0xFFFFFFFF;

StructuredBuffer<float4>   positions;         // float4 a point
StructuredBuffer<uint>     indices;           // 3 a triangle: points
StructuredBuffer<uint>     triangleCorners;   // 3 a triangle: face-vertex corners
StructuredBuffer<uint>     triangleFaces;     // 1 a triangle: the authored face
StructuredBuffer<float4>   normals;           // a primvar, float4 an element
StructuredBuffer<float4>   uvs;
RWStructuredBuffer<float4> stream;
RWStructuredBuffer<float4> extents;   // chunkCount * 2: low, high
ConstantBuffer<PackParams> pack;

/// Which element of a primvar one corner of a triangle reads. `kNoPrimvar`
/// back means there is nothing to read.
uint elementAt(uint mode, uint count, uint triangle, uint corner, uint point) {
    if (mode == kNoPrimvar || count == 0) {
        return kNoPrimvar;
    }
    if (mode == kConstant) {
        return 0;
    }
    if (mode == kUniform) {
        const uint face = triangleFaces[triangle];
        return face < count ? face : kNoPrimvar;
    }
    const uint at = mode == kFaceVarying ? triangleCorners[triangle * 3 + corner] : point;
    return at < count ? at : kNoPrimvar;
}

/// Where one entry of the stream lands. The destination is a picture, so an
/// entry's row is its own and the rows are padded -- the same address the
/// effect's own `meshEntry` computes from the other side.
uint streamAt(uint entry) {
    const uint x = entry % pack.destWidth;
    const uint y = entry / pack.destWidth;
    return y * pack.destStride + x;
}

float3 toWorld(float3 p) {
    return float3(dot(pack.toWorld0, float4(p, 1.0)), dot(pack.toWorld1, float4(p, 1.0)),
                  dot(pack.toWorld2, float4(p, 1.0)));
}

float3 normalToWorld(float3 n) {
    return float3(dot(pack.normalTo0.xyz, n), dot(pack.normalTo1.xyz, n), dot(pack.normalTo2.xyz, n));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void meshPack(uint3 tid: SV_DispatchThreadID) {
    const uint triangle = tid.x;
    if (triangle >= pack.triangles) {
        return;
    }
    const uint3 points = uint3(indices[triangle * 3], indices[triangle * 3 + 1], indices[triangle * 3 + 2]);
    float3 corner[3];
    for (uint k = 0; k < 3; ++k) {
        const uint point = points[k];
        corner[k] = toWorld(point < pack.points ? positions[point].xyz : float3(0.0));
    }
    // The geometric normal, for a mesh with no normals of its own and for a
    // corner whose primvar is missing. Wound as the triangles are.
    const float3 crossed = cross(corner[1] - corner[0], corner[2] - corner[0]);
    const float area2 = length(crossed);
    const float3 geometric = area2 > 1.0e-20 ? crossed / area2 : float3(0.0, 0.0, 1.0);

    for (uint k = 0; k < 3; ++k) {
        const uint point = points[k];
        float3 normal = geometric;
        const uint atNormal = elementAt(pack.normalMode, pack.normalCount, triangle, k, point);
        if (atNormal != kNoPrimvar) {
            const float3 authored = normalToWorld(normals[atNormal].xyz);
            if (dot(authored, authored) > 1.0e-20) {
                normal = normalize(authored);
            }
        }
        float2 uv = float2(0.0, 0.0);
        const uint atUv = elementAt(pack.uvMode, pack.uvCount, triangle, k, point);
        if (atUv != kNoPrimvar) {
            uv = uvs[atUv].xy;
        }
        const uint entry = pack.destFirst + triangle * 6 + k * 2;
        stream[streamAt(entry)] = float4(corner[k], uv.x);
        stream[streamAt(entry + 1)] = float4(normal, uv.y);
    }
}

/// The stream's own box, in world space, in the shape `lrt/scene/bounds_reduce`
/// folds: one low and one high corner a chunk.
///
/// It is read off the stream rather than off the mesh's points because the
/// stream is where the world transform has been applied -- and putting a box
/// through an affine map on the host would be arithmetic on scene data, which
/// this engine does not do. Only the even entries are positions; the odd ones
/// are normals, and a box drawn round those would be the unit sphere.
[shader("compute")]
[numthreads(64, 1, 1)]
void streamBoundsChunks(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= pack.chunkCount) {
        return;
    }
    const uint begin = c * pack.chunkSize;
    const uint end = min(begin + pack.chunkSize, pack.entries);
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    for (uint i = begin; i < end; ++i) {
        if ((i & 1u) == 0u) {
            const float3 p = stream[streamAt(i)].xyz;
            lo = min(lo, p);
            hi = max(hi, p);
        }
    }
    // Every mesh folds into the same buffer, at its own offset, so that one
    // reduce over the lot gives the model's box rather than each mesh's.
    const uint slot = pack.chunkFirst + c;
    extents[slot * 2] = float4(lo, 0.0);
    extents[slot * 2 + 1] = float4(hi, 0.0);
}
