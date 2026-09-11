// Copyright (c) 2026 lucabRTrender contributors.
//
// Which groups of a level (or which splats) this frame draws. See lod_common.
module lod_cut;
import lrt.common.morton;
import lod_common;

StructuredBuffer<uint>    cells;      // per group: its cell code at `level`; per splat: its code
RWStructuredBuffer<uint>  selected;
ConstantBuffer<LodParams> params;     // count; level (of `cells`); threshold; eye; coarsest

/// The pixels a level-`level` cell `cell` spans at its nearest point to the eye.
float projectedEdge(uint cell, uint level) {
    const float edge = params.extent / float(1u << level);
    if (params.orthographic != 0) {
        return edge * params.pixelsPerUnit;
    }
    const float3 lo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ) + float3(cellIndex(cell)) * edge;
    const float3 eye = float3(params.eyeX, params.eyeY, params.eyeZ);
    const float3 nearest = clamp(eye, lo, lo + edge);
    // Inside the cell: as near as it gets, so as big as it gets.
    const float distance = max(length(nearest - eye), 1e-6);
    return edge * params.pixelsPerUnit / distance;
}

bool small(uint cell, uint level) {
    return params.threshold > 0.0 && projectedEdge(cell, level) <= params.threshold;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lodCutGroups(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint cell = cells[g];
    const bool here = small(cell, params.level);
    const bool above = params.coarsest == 0 && small(cell >> 3, params.level - 1);
    selected[g] = (here && !above) ? 1u : 0u;
}

/// Splats: drawn unless the finest merged level's cell holding them is small.
[shader("compute")]
[numthreads(256, 1, 1)]
void lodCutSplats(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    selected[i] = small(cells[i] >> levelShift(params.level), params.level) ? 0u : 1u;
}
