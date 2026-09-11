// Copyright (c) 2026 lucabRTrender contributors.
//
// A USD mesh's polygons into triangles, on the device, in exactly the order
// Hydra's HdMeshUtil::ComputeTriangleIndices makes them (pxr/imaging/hd/
// meshUtil.cpp): a fan from each face's first vertex, faces with fewer than
// three vertices and hole faces skipped, a left-handed face's fan reversed and
// its first and last triangles rotated so the hidden edge stays hidden. The
// same order Storm and hdEmbree draw, so their images are an oracle.
//
//   meshPoints        point3f / point3h words -> float4 positions
//   meshHoles         a flag per hole face
//   meshFaceCounts    corners and triangles per face (prefix sums follow)
//   meshTriangulate   per face: its triangles' point indices, face-vertex
//                     corners (for faceVarying primvars) and face (uniform)
module mesh_topology;
import lrt.common.packing;

struct GeomParams {
    uint count;       // elements the dispatch covers
    uint faces;
    uint indices;     // faceVertexIndices length
    uint points;
    uint holes;
    uint flip;        // 1: leftHanded
    uint half;        // points as float16
    uint pad0;
};

StructuredBuffer<uint>     words;           // points: float or half words
RWStructuredBuffer<float4> positions;
StructuredBuffer<int>      faceVertexCounts;
StructuredBuffer<int>      faceVertexIndices;
StructuredBuffer<int>      holeIndices;
RWStructuredBuffer<uint>   holeFlags;       // per face
RWStructuredBuffer<uint>   cornerCounts;    // per face
RWStructuredBuffer<uint>   triangleCounts;  // per face
StructuredBuffer<uint>     faceStarts;      // per face: first corner (exclusive prefix sum)
StructuredBuffer<uint>     triangleStarts;  // per face: first triangle
RWStructuredBuffer<uint>   triangles;       // 3 per triangle: point indices
RWStructuredBuffer<uint>   triangleCorners; // 3 per triangle: face-vertex indices
RWStructuredBuffer<uint>   triangleFaces;   // per triangle: authored face
ConstantBuffer<GeomParams> params;

float pointValue(uint index) {
    if (params.half != 0) {
        const uint word = words[index / 2];
        return (index & 1u) == 0 ? lowHalf(word) : highHalf(word);
    }
    return bit_cast<float>(words[index]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshPoints(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    positions[i] = float4(pointValue(i * 3), pointValue(i * 3 + 1), pointValue(i * 3 + 2), 1.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshHoles(uint3 tid: SV_DispatchThreadID) {
    const uint h = tid.x;
    if (h >= params.holes) {
        return;
    }
    const int face = holeIndices[h];
    if (face >= 0 && uint(face) < params.faces) {
        holeFlags[uint(face)] = 1;   // a repeated hole writes the same 1
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshFaceCounts(uint3 tid: SV_DispatchThreadID) {
    const uint f = tid.x;
    if (f >= params.faces) {
        return;
    }
    const int nv = faceVertexCounts[f];
    cornerCounts[f] = uint(max(nv, 0));
    triangleCounts[f] = (nv >= 3 && holeFlags[f] == 0) ? uint(nv - 2) : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void meshTriangulate(uint3 tid: SV_DispatchThreadID) {
    const uint f = tid.x;
    if (f >= params.faces) {
        return;
    }
    const uint count = triangleCounts[f];
    const uint nv = count + 2;
    const uint start = faceStarts[f];
    const uint first = triangleStarts[f];
    const bool flip = params.flip != 0;
    for (uint j = 0; j < count; ++j) {
        const uint t = first + j;
        uint3 corner = flip ? uint3(start, start + j + 2, start + j + 1) : uint3(start, start + j + 1, start + j + 2);
        if (flip && nv > 3) {
            // HdMeshUtil's rotations: 021 -> 210 first, 043 -> 304 last.
            if (j == 0) {
                corner = uint3(corner.y, corner.z, corner.x);
            } else if (j == nv - 3) {
                corner = uint3(corner.z, corner.x, corner.y);
            }
        }
        const bool inside = start + j + 2 < params.indices;
        for (uint k = 0; k < 3; ++k) {
            // An overrun makes a triangle of point 0, as HdMeshUtil does.
            const int point = inside ? faceVertexIndices[corner[k]] : 0;
            triangles[t * 3 + k] = (point >= 0 && uint(point) < params.points) ? uint(point) : 0u;
            triangleCorners[t * 3 + k] = inside ? corner[k] : 0u;
        }
        triangleFaces[t] = f;
    }
}
