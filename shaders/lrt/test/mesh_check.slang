// Copyright (c) 2026 lucabRTrender contributors.
//
// Mesh tests' checks, each an independent way to the same answer:
//   areas         the triangles' total area against the faces' own (Newell's
//                 polygon area, straight from the authored topology)
//   corners       every triangle corner inside its face, pointing at the
//                 face-vertex index it names
//   facing        triangles facing +z (a planar mesh, either handedness)
//   bruteNormals  per point, the area-weighted normals of every triangle
//                 touching it, found by looking at every triangle
module mesh_check;

struct CheckParams {
    uint faces;
    uint indices;
    uint triangles;
    uint points;
    uint holes;
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<int>      faceVertexCounts;
StructuredBuffer<int>      faceVertexIndices;
StructuredBuffer<int>      holeIndices;
StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     triangles;
StructuredBuffer<uint>     triangleCorners;
StructuredBuffer<uint>     triangleFaces;
StructuredBuffer<float4>   normals;
RWStructuredBuffer<float>  result;   // areas: [triangles, faces]; normals: [worst angle]
RWStructuredBuffer<uint>   counts;   // corners: [bad]; facing: [+z, -z]
ConstantBuffer<CheckParams> params;

/// The angle between two unit vectors, precise when it is small (acos is
/// not: in float it cannot see below 6e-4 rad).
float angleBetween(float3 a, float3 b) {
    return 2.0 * asin(clamp(length(a - b) * 0.5, 0.0, 1.0));
}

bool isHole(uint f) {
    for (uint h = 0; h < params.holes; ++h) {
        if (holeIndices[h] == int(f)) {
            return true;
        }
    }
    return false;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void areas(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float triangleArea = 0.0;
    for (uint t = 0; t < params.triangles; ++t) {
        const float3 a = positions[triangles[t * 3]].xyz;
        const float3 b = positions[triangles[t * 3 + 1]].xyz;
        const float3 c = positions[triangles[t * 3 + 2]].xyz;
        triangleArea += 0.5 * length(cross(b - a, c - a));
    }
    float faceArea = 0.0;
    uint start = 0;
    for (uint f = 0; f < params.faces; ++f) {
        const int nv = faceVertexCounts[f];
        if (nv >= 3 && !isHole(f)) {
            float3 newell = float3(0.0);
            for (uint j = 0; j < uint(nv); ++j) {
                const float3 p = positions[uint(faceVertexIndices[start + j])].xyz;
                const float3 q = positions[uint(faceVertexIndices[start + (j + 1) % uint(nv)])].xyz;
                newell += cross(p, q);
            }
            faceArea += 0.5 * length(newell);
        }
        start += uint(max(nv, 0));
    }
    result[0] = triangleArea;
    result[1] = faceArea;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void corners(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint bad = 0;
    for (uint t = 0; t < params.triangles; ++t) {
        const uint f = triangleFaces[t];
        uint start = 0;
        for (uint g = 0; g < f; ++g) {
            start += uint(max(faceVertexCounts[g], 0));
        }
        const uint nv = uint(max(faceVertexCounts[f], 0));
        for (uint k = 0; k < 3; ++k) {
            const uint corner = triangleCorners[t * 3 + k];
            bad += (corner < start || corner >= start + nv) ? 1u : 0u;
            bad += triangles[t * 3 + k] != uint(faceVertexIndices[corner]) ? 1u : 0u;
        }
    }
    counts[0] = bad;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void facing(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint up = 0;
    uint down = 0;
    for (uint t = 0; t < params.triangles; ++t) {
        const float3 a = positions[triangles[t * 3]].xyz;
        const float3 b = positions[triangles[t * 3 + 1]].xyz;
        const float3 c = positions[triangles[t * 3 + 2]].xyz;
        const float z = cross(b - a, c - a).z;
        up += z > 0.0 ? 1u : 0u;
        down += z < 0.0 ? 1u : 0u;
    }
    counts[0] = up;
    counts[1] = down;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bruteNormals(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float worst = 0.0;
    for (uint p = 0; p < params.points; ++p) {
        float3 sum = float3(0.0);
        for (uint t = 0; t < params.triangles; ++t) {
            for (uint k = 0; k < 3; ++k) {
                if (triangles[t * 3 + k] == p) {
                    const float3 a = positions[triangles[t * 3]].xyz;
                    const float3 b = positions[triangles[t * 3 + 1]].xyz;
                    const float3 c = positions[triangles[t * 3 + 2]].xyz;
                    sum += cross(b - a, c - a);
                }
            }
        }
        const float3 expected = normalize(sum);
        const float3 got = normals[p].xyz;
        worst = max(worst, angleBetween(expected, got));
    }
    result[0] = worst;
}

/// The worst angle between each point's normal and its direction from the
/// origin: a sphere's analytic normal.
[shader("compute")]
[numthreads(1, 1, 1)]
void sphereNormals(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float worst = 0.0;
    for (uint p = 0; p < params.points; ++p) {
        const float3 expected = normalize(positions[p].xyz);
        worst = max(worst, angleBetween(expected, normals[p].xyz));
    }
    result[0] = worst;
}
