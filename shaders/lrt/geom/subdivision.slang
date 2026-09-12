// Copyright (c) 2026 lucabRTrender contributors.
//
// One level of subdivision, the arithmetic of it: the host lays out the
// level's topology (which corners make which edges, what each vertex
// touches, which edges and corners are sharp) as tables, and these kernels
// place the new points -- a point per face, a point per edge, a moved point
// per vertex -- for Catmull-Clark, Loop and bilinear schemes, with creases
// and boundaries. A face-varying channel runs through the same kernels over
// its own topology. `subdivLimit` pushes a level's vertices onto the limit
// surface with the stencil their valence gives.
module subdivision;

static const uint kNone = 0xFFFFFFFF;

struct SubdivParams {
    uint  scheme;        // 0 bilinear, 1 catmullClark, 2 loop
    uint  vertices;
    uint  edges;
    uint  faces;
    uint  edgeBase;      // where the edge points start in the output: vertices
    uint  faceBase;      // and the face points: vertices + edges
    uint  pad0; uint pad1;
};

StructuredBuffer<float4>    values;         // per vertex of this level
StructuredBuffer<uint>      faceStarts;     // per face: first corner
StructuredBuffer<uint>      faceCounts;     // per face: corners
StructuredBuffer<uint>      corners;        // per corner: its vertex
StructuredBuffer<uint4>     edgeEnds;       // per edge: v0, v1, f0, f1 (kNone: none -- a boundary)
StructuredBuffer<uint2>     edgeOpposite;   // per edge: the third vertex of f0 and of f1 (loop)
StructuredBuffer<float>     edgeSharp;      // per edge
StructuredBuffer<uint>      vertexOffsets;  // per vertex + 1: into vertexEdges / vertexFaces
StructuredBuffer<uint>      vertexEdges;    // the edges at a vertex
StructuredBuffer<uint>      vertexFaces;    // the faces at a vertex (as many entries as edges; kNone where fewer)
StructuredBuffer<float>     vertexSharp;    // per vertex: its corner sharpness
RWStructuredBuffer<float4>  refined;        // [vertex points][edge points][face points]
ConstantBuffer<SubdivParams> params;

float4 facePoint(uint f) {
    const uint start = faceStarts[f];
    const uint n = faceCounts[f];
    float4 sum = float4(0.0);
    for (uint k = 0; k < n; ++k) {
        sum += values[corners[start + k]];
    }
    return n > 0 ? sum / float(n) : sum;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subdivFacePoints(uint3 tid: SV_DispatchThreadID) {
    const uint f = tid.x;
    if (f >= params.faces) {
        return;
    }
    refined[params.faceBase + f] = facePoint(f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subdivEdgePoints(uint3 tid: SV_DispatchThreadID) {
    const uint e = tid.x;
    if (e >= params.edges) {
        return;
    }
    const uint4 ends = edgeEnds[e];
    const float4 mid = 0.5 * (values[ends.x] + values[ends.y]);
    const bool boundary = ends.z == kNone || ends.w == kNone;
    const float s = clamp(edgeSharp[e], 0.0, 1.0);
    float4 smooth = mid;
    if (!boundary) {
        if (params.scheme == 1) {
            smooth = 0.25 * (values[ends.x] + values[ends.y] + refined[params.faceBase + ends.z] +
                             refined[params.faceBase + ends.w]);
        } else if (params.scheme == 2) {
            const uint2 opposite = edgeOpposite[e];
            smooth = 0.375 * (values[ends.x] + values[ends.y]) + 0.125 * (values[opposite.x] + values[opposite.y]);
        }
    }
    refined[params.edgeBase + e] = boundary ? mid : lerp(smooth, mid, s);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subdivVertexPoints(uint3 tid: SV_DispatchThreadID) {
    const uint v = tid.x;
    if (v >= params.vertices) {
        return;
    }
    const float4 p = values[v];
    const uint first = vertexOffsets[v];
    const uint last = vertexOffsets[v + 1];
    const uint n = last - first;
    if (params.scheme == 0 || n == 0) {
        refined[v] = p;
        return;
    }
    // The sharp edges at the vertex, and the boundary ones: up to two of
    // them make a crease through it, more make a corner.
    uint sharpCount = 0;
    float sharpness = 0.0;
    float4 creaseEnds = float4(0.0);
    float4 edgeMids = float4(0.0);
    float4 neighbours = float4(0.0);
    float4 facePoints = float4(0.0);
    uint faceCount = 0;
    bool boundary = false;
    for (uint k = first; k < last; ++k) {
        const uint e = vertexEdges[k];
        const uint4 ends = edgeEnds[e];
        const uint other = ends.x == v ? ends.y : ends.x;
        const bool edgeBoundary = ends.z == kNone || ends.w == kNone;
        const float s = edgeBoundary ? 1.0 : clamp(edgeSharp[e], 0.0, 1.0);
        boundary = boundary || edgeBoundary;
        if (s > 0.0) {
            if (sharpCount < 2) {
                creaseEnds += values[other];
            }
            ++sharpCount;
            sharpness = max(sharpness, s);
        }
        neighbours += values[other];
        edgeMids += 0.5 * (p + values[other]);
        const uint f = vertexFaces[k];
        if (f != kNone) {
            facePoints += refined[params.faceBase + f];
            ++faceCount;
        }
    }
    float4 smooth = p;
    if (params.scheme == 1) {
        // Catmull-Clark: (Q + 2 R + (n - 3) P) / n, Q the new face points'
        // mean, R the edge midpoints' mean.
        const float fn = float(n);
        const float4 q = faceCount > 0 ? facePoints / float(faceCount) : p;
        const float4 r = edgeMids / fn;
        smooth = (q + 2.0 * r + (fn - 3.0) * p) / fn;
    } else {
        // Loop: (1 - n beta) P + beta (sum of neighbours).
        const float beta = n == 3 ? 3.0 / 16.0 : 3.0 / (8.0 * float(n));
        smooth = (1.0 - float(n) * beta) * p + beta * neighbours;
    }
    const float4 crease = 0.125 * (creaseEnds + 6.0 * p);
    float4 out = smooth;
    if (sharpCount >= 3 || (boundary && n == 2)) {
        // A corner, or a boundary vertex that is nothing but its two
        // boundary edges: pinned (edgeAndCorner for points, cornersOnly for
        // a face-varying channel), or a square's corners would round off.
        out = p;
    } else if (sharpCount == 2) {
        out = lerp(smooth, crease, sharpness);
    }
    const float corner = clamp(vertexSharp[v], 0.0, 1.0);
    refined[v] = lerp(out, p, corner);
}

/// The limit position of every vertex of a level, from the level's own
/// neighbours: Catmull-Clark's (n^2 P + 4 sum of edge neighbours + sum of
/// the quads' diagonal vertices) / (n (n + 5)) (Halstead, Kass and DeRose;
/// exact once every face is a quad, which one level makes true), Loop's
/// (1 - n gamma) P + gamma sum of neighbours; a vertex on a crease or a
/// boundary takes (E1 + 4 P + E2) / 6 along it, and a corner stays. A first
/// version took edge midpoints and face centroids for the two sums and put
/// the cube's corner at 0.75 where deep subdivision put it at 0.5.
[shader("compute")]
[numthreads(256, 1, 1)]
void subdivLimit(uint3 tid: SV_DispatchThreadID) {
    const uint v = tid.x;
    if (v >= params.vertices) {
        return;
    }
    const float4 p = values[v];
    const uint first = vertexOffsets[v];
    const uint last = vertexOffsets[v + 1];
    const uint n = last - first;
    if (params.scheme == 0 || n == 0) {
        refined[v] = p;
        return;
    }
    uint sharpCount = 0;
    bool boundary = false;
    float4 creaseEnds = float4(0.0);
    float4 neighbours = float4(0.0);
    float4 diagonals = float4(0.0);
    uint faceCount = 0;
    for (uint k = first; k < last; ++k) {
        const uint e = vertexEdges[k];
        const uint4 ends = edgeEnds[e];
        const uint other = ends.x == v ? ends.y : ends.x;
        const bool edgeBoundary = ends.z == kNone || ends.w == kNone;
        boundary = boundary || edgeBoundary;
        if (edgeBoundary || edgeSharp[e] >= 1.0) {
            if (sharpCount < 2) {
                creaseEnds += values[other];
            }
            ++sharpCount;
        }
        neighbours += values[other];
        const uint f = vertexFaces[k];
        if (f != kNone) {
            // The quad's vertex across from v; a face of another size gives
            // its centroid instead, which is only reached at level zero.
            const uint start = faceStarts[f];
            const uint n = faceCounts[f];
            float4 diagonal = facePoint(f);
            if (n == 4) {
                for (uint j = 0; j < 4; ++j) {
                    if (corners[start + j] == v) {
                        diagonal = values[corners[start + (j + 2) % 4]];
                    }
                }
            }
            diagonals += diagonal;
            ++faceCount;
        }
    }
    float4 out;
    if (sharpCount >= 3 || vertexSharp[v] >= 1.0 || (boundary && n == 2)) {
        out = p;
    } else if (sharpCount == 2) {
        out = (creaseEnds + 4.0 * p) / 6.0;
    } else if (params.scheme == 1) {
        const float fn = float(n);
        out = (fn * fn * p + 4.0 * neighbours + diagonals) / (fn * (fn + 5.0));
    } else {
        const float beta = n == 3 ? 3.0 / 16.0 : 3.0 / (8.0 * float(n));
        const float gamma = 1.0 / (float(n) + 3.0 / (8.0 * beta));
        out = (1.0 - float(n) * gamma) * p + gamma * neighbours;
    }
    refined[v] = out;
}

/// A channel's values per corner, from its values per (channel) vertex.
RWStructuredBuffer<float4> gathered;
struct GatherParams { uint corners; uint pad0; uint pad1; uint pad2; };
ConstantBuffer<GatherParams> gather;

[shader("compute")]
[numthreads(256, 1, 1)]
void subdivGather(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= gather.corners) {
        return;
    }
    gathered[c] = values[corners[c]];
}
