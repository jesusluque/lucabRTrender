// Copyright (c) 2026 lucabRTrender contributors.
//
// A converted grid against what it was made of: the active voxels
// counted through the leaf masks (a count known by construction from the
// boxes the fixture filled), and each leaf's maximum recomputed through
// the accessor, voxel by voxel, against the prepare kernel's.
import lrt.volume.nanovdb;

struct CheckParams {
    uint gridWord;
    uint leafCount;
    uint leafBase;
    uint pad0;
};

StructuredBuffer<uint>     words;
StructuredBuffer<float>    leafMax;
RWStructuredBuffer<uint>   counts;   // [active voxels, leaf maxima off, leaves checked]
ConstantBuffer<CheckParams> check;

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeCountActive(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.leafCount) return;
    const NvLeaf leaf = nvLeaf(words, nvGrid(check.gridWord), i);
    uint active = 0;
    for (uint n = 0; n < 512; ++n) {
        active += nvLeafActive(words, leaf, n) ? 1u : 0u;
    }
    InterlockedAdd(counts[0], active);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeLeafMaxCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.leafCount) return;
    const NvGrid grid = nvGrid(check.gridWord);
    const int3 origin = nvLeafOrigin(words, nvLeaf(words, grid, i));
    // Again, the other way: every voxel of the leaf through the accessor.
    float largest = 0.0;
    for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const int3 ijk = origin + int3(x, y, z);
                if (nvActiveAt(words, grid, ijk)) {
                    largest = max(largest, nvValueAt(words, grid, ijk));
                }
            }
        }
    }
    InterlockedAdd(counts[2], 1u);
    if (largest != leafMax[check.leafBase + i]) {
        InterlockedAdd(counts[1], 1u);
    }
}
