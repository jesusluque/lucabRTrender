// Copyright (c) 2026 lucabRTrender contributors.
//
// An image against one colour it should be everywhere it is drawn, counted
// apart for the interior -- pixels whose neighbours are all drawn -- since a
// kernel's screen derivatives are only as good as its quad, and a quad that
// straddles a silhouette has nothing to difference against.
module constant_colour;

struct ConstantParams {
    float r; float g; float b; float tolerance;
    uint  width; uint height; uint pad0; uint pad1;
};

StructuredBuffer<float4>       colour;
StructuredBuffer<float>        depth;
RWStructuredBuffer<uint>       counts;   // [drawn, interior, differ (interior), differ (edge)]
ConstantBuffer<ConstantParams> constant;

bool drawnAt(uint x, uint y) {
    return depth[y * constant.width + x] > 0.0;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void constantColour(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint drawn = 0, interior = 0, differ = 0, edge = 0;
    for (uint y = 0; y < constant.height; ++y) {
        for (uint x = 0; x < constant.width; ++x) {
            if (!drawnAt(x, y)) {
                continue;
            }
            ++drawn;
            const float3 c = colour[y * constant.width + x].rgb;
            const bool off = any(abs(c - float3(constant.r, constant.g, constant.b)) > constant.tolerance);
            bool inside = x > 0 && y > 0 && x + 1 < constant.width && y + 1 < constant.height;
            for (int dy = -1; inside && dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    inside = inside && drawnAt(uint(int(x) + dx), uint(int(y) + dy));
                }
            }
            if (inside) {
                ++interior;
                differ += off ? 1u : 0u;
            } else {
                edge += off ? 1u : 0u;
            }
        }
    }
    counts[0] = drawn;
    counts[1] = interior;
    counts[2] = differ;
    counts[3] = edge;
}
