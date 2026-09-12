// Copyright (c) 2026 lucabRTrender contributors.
//
// What a volume needs beyond its grid, computed on the device: the index
// bounds of its leaves (the statistics NanoVDB would have computed on the
// host are off), each leaf's largest value -- the majorant delta tracking
// takes through that leaf -- and the grid's largest value overall. One
// thread a leaf; then one a grid to write what it found into the volume's
// record, in the same buffer of words the kernels read it from.
import lrt.volume.nanovdb;

struct PrepareParams {
    uint gridWord;    // the grid's word offset in `words`
    uint leafCount;
    uint leafBase;    // word offset in `words` of this grid's leaf maxima
    uint slot;        // which grid: its six bounds entries, its majorant word, its record
};

StructuredBuffer<uint>       words;      // the volumes: header, records, grids, leaf maxima
RWStructuredBuffer<uint>     wordsOut;   // the same buffer, for what these kernels write into it (leaf maxima, records)
RWStructuredBuffer<int>      bounds;     // per grid: min xyz, max xyz (exclusive) in index space
RWStructuredBuffer<uint>     majorant;   // per grid: the largest leaf maximum, as float bits (non-negative floats order as uints)
ConstantBuffer<PrepareParams> params;

static const uint kHeaderWords = 8;
static const uint kRecordWords = 32;

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
    wordsOut[params.leafBase + i] = asuint(largest);
    InterlockedMax(majorant[params.slot], asuint(max(largest, 0.0)));
}

/// The record's bounds and majorant, and where leaf 0 sits, once the
/// leaves have been walked.
[shader("compute")]
[numthreads(1, 1, 1)]
void volumeFinish(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const uint b = kHeaderWords + params.slot * kRecordWords;
    const uint base = params.slot * 6u;
    const bool any = params.leafCount > 0;
    for (uint k = 0; k < 3; ++k) {
        wordsOut[b + 16 + k] = asuint(any ? float(bounds[base + k]) : 0.0);
        wordsOut[b + 19 + k] = asuint(any ? float(bounds[base + 3 + k]) : 0.0);
    }
    wordsOut[b + 27] = majorant[params.slot];
    wordsOut[b + 3] = any ? nvLeaf(words, nvGrid(params.gridWord), 0u).byteOffset / 4u : 0u;
}
