// Copyright (c) 2026 lucabRTrender contributors.
//
// Invisible faces checked by what must not move: with the odd faces of a
// mesh hidden, every pixel where the full mesh showed an even face must
// still show exactly that (instance, triangle) -- hiding faces must not
// renumber the rest, and a face that was seen cannot become hidden -- and
// no drawn pixel may be an odd face's triangle. Where the full mesh showed
// an odd face, the hidden one shows whatever stood behind it, or nothing.
module hidden_check;

struct HiddenParams {
    uint width; uint height;
    uint firstTriangle;   // the mesh's, in the pool
    uint pad0;
};

Texture2D<uint4>         full;      // ids with every face drawn
Texture2D<uint4>         hidden;    // ids with the odd faces hidden
StructuredBuffer<uint>   triangleFaces;
RWStructuredBuffer<uint> counts;    // [drawn in hidden, of those differing from full, of those on an odd face, drawn in full]
ConstantBuffer<HiddenParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void hiddenCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint drawn = 0;
    uint differing = 0;
    uint odd = 0;
    uint drawnFull = 0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            const uint4 a = full.Load(int3(int(x), int(y), 0));
            const uint4 b = hidden.Load(int3(int(x), int(y), 0));
            drawnFull += a.x != 0 ? 1u : 0u;
            if (b.x == 0) {
                continue;
            }
            drawn += 1;
            const bool fullEven = a.x != 0 && (triangleFaces[params.firstTriangle + a.y] & 1u) == 0;
            differing += (fullEven && (a.x != b.x || a.y != b.y)) ? 1u : 0u;
            odd += (triangleFaces[params.firstTriangle + b.y] & 1u) != 0 ? 1u : 0u;
        }
    }
    counts[0] = drawn;
    counts[1] = differing;
    counts[2] = odd;
    counts[3] = drawnFull;
}
