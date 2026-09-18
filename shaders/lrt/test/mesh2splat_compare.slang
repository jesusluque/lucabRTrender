// Copyright (c) 2026 lucabRTrender contributors.
//
// Two runs of the conversion, entry by entry. The point is not that the
// numbers are close: it is that they are the same bits, because a gaussian is
// followed from one frame of an animation to the next by being the same
// element of the array. An emit that takes its slot from an atomic gives the
// same set in a different order, and this counts that.
module mesh2splat_compare;

struct CompareParams {
    uint entries;    ///< float4s to compare
    uint width;      ///< both pictures are laid out alike
    uint stride;
    uint pad;
};

StructuredBuffer<float4>   first;
StructuredBuffer<float4>   second;
RWStructuredBuffer<uint>   counts;   // [compared, differing]
ConstantBuffer<CompareParams> params;

uint entryAt(uint index) {
    const uint x = index % max(params.width, 1u);
    const uint y = index / max(params.width, 1u);
    return y * params.stride + x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCompare(uint3 tid: SV_DispatchThreadID) {
    const uint index = tid.x;
    if (index >= params.entries) {
        return;
    }
    const uint at = entryAt(index);
    const float4 a = first[at];
    const float4 b = second[at];
    InterlockedAdd(counts[0], 1u);
    const bool same = asuint(a.x) == asuint(b.x) && asuint(a.y) == asuint(b.y) &&
                      asuint(a.z) == asuint(b.z) && asuint(a.w) == asuint(b.w);
    if (!same) {
        InterlockedAdd(counts[1], 1u);
    }
}
