// Copyright (c) 2026 lucabRTrender contributors.
//
// GeomSubsets on the device: each face the subset it belongs to, then each
// triangle its face's -- 0 for none, k + 1 for the k-th subset -- so shading
// can pick a triangle's material. Subsets of one family are disjoint, as
// USD's materialBind family requires; a face listed twice keeps one of them.
module mesh_subsets;

struct SubsetParams {
    uint count;        // entries, faces or triangles, as the entry point needs
    uint faces;
    uint value;        // clear: what every entry becomes
    uint pad0;
};

StructuredBuffer<uint>       subsetFaces;       // pairs: face, subset + 1
RWStructuredBuffer<uint>     faceSubsets;       // per face
StructuredBuffer<uint>       triangleFaces;     // per triangle: its face
RWStructuredBuffer<uint>     triangleSubsets;   // per triangle
RWStructuredBuffer<uint>     cleared;
ConstantBuffer<SubsetParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void subsetClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        cleared[tid.x] = params.value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subsetScatter(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const uint face = subsetFaces[tid.x * 2];
    if (face < params.faces) {
        faceSubsets[face] = subsetFaces[tid.x * 2 + 1];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subsetTriangles(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        triangleSubsets[tid.x] = faceSubsets[triangleFaces[tid.x]];
    }
}
