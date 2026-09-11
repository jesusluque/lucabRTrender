// Copyright (c) 2026 lucabRTrender contributors.
//
// Smooth normals as Hydra computes them (Hd_SmoothNormals over
// Hd_VertexAdjacency): at every face corner the cross product of the edges to
// its next and previous vertices -- swapped for a left-handed face -- summed
// per point and normalised. Every face counts, holes and all, as in Hydra.
//
//   cornerKeys     per corner: its point (the sort key) and its face
//   (radix sort of (point, corner) pairs)
//   pointRuns      where each point's corners start in the sorted order
//   pointNormals   per point: the sum over its run
module mesh_normals;

struct NormalParams {
    uint count;
    uint faces;
    uint indices;
    uint points;
    uint corners;
    uint flip;
    uint pad0;
    uint pad1;
};

StructuredBuffer<int>      faceVertexCounts;
StructuredBuffer<int>      faceVertexIndices;
StructuredBuffer<uint>     faceStarts;
StructuredBuffer<float4>   positions;
RWStructuredBuffer<uint>   keys;          // per corner: its point (sorted afterwards)
RWStructuredBuffer<uint>   values;        // per corner: itself (sorted afterwards)
RWStructuredBuffer<uint>   cornerFaces;   // per corner: its face
RWStructuredBuffer<uint>   runStarts;     // per point
RWStructuredBuffer<float4> normals;       // per point
ConstantBuffer<NormalParams> params;

static const uint kNone = 0xFFFFFFFF;

[shader("compute")]
[numthreads(256, 1, 1)]
void cornerKeys(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.corners) {
        return;
    }
    // The face holding corner c: the last whose first corner is at or before c.
    uint lo = 0;
    uint hi = params.faces;
    while (hi - lo > 1) {
        const uint middle = (lo + hi) / 2;
        if (faceStarts[middle] <= c) {
            lo = middle;
        } else {
            hi = middle;
        }
    }
    cornerFaces[c] = lo;
    const int point = c < params.indices ? faceVertexIndices[c] : -1;
    // Out of range: sorted past every real point and never summed.
    keys[c] = (point >= 0 && uint(point) < params.points) ? uint(point) : params.points;
    values[c] = c;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearRuns(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.points) {
        runStarts[tid.x] = kNone;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pointRuns(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.corners) {
        return;
    }
    const uint point = keys[i];
    if (point < params.points && (i == 0 || keys[i - 1] != point)) {
        runStarts[point] = i;
    }
}

float3 pointAt(int index) {
    return (index >= 0 && uint(index) < params.points) ? positions[uint(index)].xyz : float3(0.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pointNormals(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p >= params.points) {
        return;
    }
    float3 sum = float3(0.0);
    const float3 current = positions[p].xyz;
    for (uint i = runStarts[p]; i != kNone && i < params.corners && keys[i] == p; ++i) {
        const uint c = values[i];
        const uint f = cornerFaces[c];
        const uint start = faceStarts[f];
        const uint nv = uint(max(faceVertexCounts[f], 0));
        const uint j = c - start;
        const uint previousCorner = start + (j + nv - 1) % nv;
        const uint nextCorner = start + (j + 1) % nv;
        int previous = previousCorner < params.indices ? faceVertexIndices[previousCorner] : -1;
        int next = nextCorner < params.indices ? faceVertexIndices[nextCorner] : -1;
        if (params.flip != 0) {
            const int swap = previous;
            previous = next;
            next = swap;
        }
        sum += cross(pointAt(next) - current, pointAt(previous) - current);
    }
    const float length2 = dot(sum, sum);
    normals[p] = float4(length2 > 0.0 ? sum / sqrt(length2) : sum, 0.0);
}
