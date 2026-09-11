// Copyright (c) 2026 lucabRTrender contributors.
//
// Two visibility textures compared where the answer is not a matter of edge
// rules: a pixel counts when its 3x3 neighbourhood in `a` all saw the same
// triangle, so only pixels well inside a triangle are held to agreement.
module ids_compare;

struct IdsParams {
    uint width;
    uint height;
    uint pad0; uint pad1;
};

Texture2D<uint4>          a;
Texture2D<uint4>          b;
RWStructuredBuffer<uint>  counts;   // [interior mismatches, interior pixels, covered in a, covered in b]
ConstantBuffer<IdsParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void idsCompare(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint mismatches = 0;
    uint interior = 0;
    uint coveredA = 0;
    uint coveredB = 0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            const uint2 va = a.Load(int3(int(x), int(y), 0)).xy;
            const uint2 vb = b.Load(int3(int(x), int(y), 0)).xy;
            coveredA += va.x != 0 ? 1u : 0u;
            coveredB += vb.x != 0 ? 1u : 0u;
            if (x == 0 || y == 0 || x + 1 == params.width || y + 1 == params.height) {
                continue;
            }
            bool uniform = true;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uniform = uniform && all(a.Load(int3(int(x) + dx, int(y) + dy, 0)).xy == va);
                }
            }
            if (uniform) {
                ++interior;
                mismatches += any(va != vb) ? 1u : 0u;
            }
        }
    }
    counts[0] = mismatches;
    counts[1] = interior;
    counts[2] = coveredA;
    counts[3] = coveredB;
}
