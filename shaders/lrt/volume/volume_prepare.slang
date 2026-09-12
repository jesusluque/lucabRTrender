// Copyright (c) 2026 lucabRTrender contributors.
//
// What a volume needs beyond its grid, computed on the device: the index
// bounds of its leaves (the statistics NanoVDB would have computed on the
// host are off), each leaf's largest value -- the majorant delta tracking
// takes through that leaf -- and the grid's largest value overall. One
// thread a leaf.
import lrt.volume.nanovdb;

struct PrepareParams {
    uint gridWord;    // the grid's word offset in `words`
    uint leafCount;
    uint leafBase;    // where this grid's leaf maxima start in `leafMax`
    uint slot;        // which grid: its six bounds entries and its majorant word
};

StructuredBuffer<uint>       words;
RWStructuredBuffer<int>      bounds;     // per grid: min xyz, max xyz (exclusive) in index space
RWStructuredBuffer<float>    leafMax;    // per leaf
RWStructuredBuffer<uint>     majorant;   // per grid: the largest leaf maximum, as float bits (non-negative floats order as uints)
ConstantBuffer<PrepareParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeBounds(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.leafCount) return;
    const NvGrid grid = nvGrid(params.gridWord);
    const int3 origin = nvLeafOrigin(words, nvLeaf(words, grid, i));
    const uint base = params.slot * 6u;
    InterlockedMin(bounds[base + 0], origin.x);
    InterlockedMin(bounds[base + 1], origin.y);
    InterlockedMin(bounds[base + 2], origin.z);
    InterlockedMax(bounds[base + 3], origin.x + 8);
    InterlockedMax(bounds[base + 4], origin.y + 8);
    InterlockedMax(bounds[base + 5], origin.z + 8);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeLeafMax(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.leafCount) return;
    const NvGrid grid = nvGrid(params.gridWord);
    const NvLeaf leaf = nvLeaf(words, grid, i);
    float largest = 0.0;
    for (uint n = 0; n < 512; ++n) {
        if (nvLeafActive(words, leaf, n)) {
            largest = max(largest, nvLeafValue(words, leaf, n));
        }
    }
    leafMax[params.leafBase + i] = largest;
    InterlockedMax(majorant[params.slot], asuint(max(largest, 0.0)));
}
