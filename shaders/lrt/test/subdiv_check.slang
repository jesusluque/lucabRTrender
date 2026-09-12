// Copyright (c) 2026 lucabRTrender contributors.
//
// Subdivision against closed forms written a second time: a cube corner's
// Catmull-Clark limit from the coarse cube alone, an octahedron vertex's
// Loop limit, a crease holding a plane, and a square's face-varying values
// staying a grid.
module subdiv_check;

struct SubdivCheckParams {
    uint  count;         // refined points, or faces
    uint  vertex;        // the coarse vertex whose descendant is checked (its index stays)
    uint  valence;
    uint  scheme;        // 1 catmullClark, 2 loop
    float tolerance;
    float planeY;        // crease check: the plane's height
    uint  pad0; uint pad1;
};

StructuredBuffer<float4>   coarse;      // the coarse points
StructuredBuffer<uint>     neighbours;  // the checked vertex's neighbours (valence of them)
StructuredBuffer<float4>   centroids;   // the faces around it (valence of them), their centroids computed here? no: given as corner lists below
StructuredBuffer<uint>     faceCorners; // 4 * valence: the corners of the faces around the vertex (quads)
StructuredBuffer<float4>   refined;
StructuredBuffer<uint>     refinedCounts;   // per refined face: corners
StructuredBuffer<uint>     refinedStarts;
StructuredBuffer<float4>   fvar;        // per refined corner
RWStructuredBuffer<uint>   counts;
RWStructuredBuffer<float>  worst;
ConstantBuffer<SubdivCheckParams> check;

/// The limit of the coarse vertex, from the coarse mesh alone.
float3 coarseLimit() {
    const float3 p = coarse[check.vertex].xyz;
    const float n = float(check.valence);
    if (check.scheme == 1) {
        // (n^2 P + 4 sum of edge neighbours + sum of the quads' diagonal
        // vertices) / (n (n + 5)): the quad's diagonal is the corner two
        // steps from the vertex in each face's corner list.
        float3 edges = float3(0.0);
        float3 diagonals = float3(0.0);
        for (uint k = 0; k < check.valence; ++k) {
            edges += coarse[neighbours[k]].xyz;
            for (uint j = 0; j < 4; ++j) {
                if (faceCorners[k * 4 + j] == check.vertex) {
                    diagonals += coarse[faceCorners[k * 4 + (j + 2) % 4]].xyz;
                }
            }
        }
        return (n * n * p + 4.0 * edges + diagonals) / (n * (n + 5.0));
    }
    // Loop: (1 - n gamma) P + gamma sum of neighbours.
    const float beta = check.valence == 3 ? 3.0 / 16.0 : 3.0 / (8.0 * n);
    const float gamma = 1.0 / (n + 3.0 / (8.0 * beta));
    float3 sum = float3(0.0);
    for (uint k = 0; k < check.valence; ++k) {
        sum += coarse[neighbours[k]].xyz;
    }
    return (1.0 - n * gamma) * p + gamma * sum;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void subdivLimitCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const float3 expected = coarseLimit();
    const float3 got = refined[check.vertex].xyz;
    const float gap = length(got - expected);
    counts[0] = gap > check.tolerance ? 1u : 0u;
    worst[0] = gap;
    worst[1] = expected.x;
    worst[2] = expected.y;
    worst[3] = expected.z;
}

/// A crease around a plane: nothing rises above it, and the points on it
/// (within tolerance) are counted.
[shader("compute")]
[numthreads(1, 1, 1)]
void subdivPlaneCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint above = 0;
    uint on = 0;
    float highest = -1.0e30;
    for (uint i = 0; i < check.count; ++i) {
        const float y = refined[i].y;
        highest = max(highest, y);
        above += y > check.planeY + check.tolerance ? 1u : 0u;
        on += abs(y - check.planeY) <= check.tolerance ? 1u : 0u;
    }
    counts[0] = above;
    counts[1] = on;
    worst[0] = highest;
}

/// Every refined quad's face-varying values form an axis-aligned rectangle
/// (either orientation): what a square's grid stays under subdivision.
[shader("compute")]
[numthreads(1, 1, 1)]
void subdivRectCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint bad = 0;
    for (uint f = 0; f < check.count; ++f) {
        if (refinedCounts[f] != 4) {
            bad += 1;
            continue;
        }
        const uint s = refinedStarts[f];
        const float2 a = fvar[s].xy, b = fvar[s + 1].xy, c = fvar[s + 2].xy, d = fvar[s + 3].xy;
        const float t = check.tolerance;
        const bool one = abs(a.x - d.x) < t && abs(b.x - c.x) < t && abs(a.y - b.y) < t && abs(c.y - d.y) < t;
        const bool two = abs(a.x - b.x) < t && abs(c.x - d.x) < t && abs(a.y - d.y) < t && abs(b.y - c.y) < t;
        bad += (one || two) ? 0u : 1u;
    }
    counts[0] = bad;
}
