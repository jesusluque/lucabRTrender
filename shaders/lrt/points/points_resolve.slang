// Copyright (c) 2026 lucabRTrender contributors.
//
// The raster targets into the engine's buffers: bottom row first (a render
// target's row 0 is the top), colour normalised when it was accumulated.
module points_resolve;

struct ResolveParams {
    uint width;
    uint height;
    uint normalise;   // 1 after surface splatting: rgb /= weight
    uint pad0;
};

Texture2D<float4>             colourTarget;
Texture2D<float4>             depthTarget;
RWStructuredBuffer<float4>    colour;
RWStructuredBuffer<float>     depth;
ConstantBuffer<ResolveParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void pointsResolve(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const int3 texel = int3(int(tid.x), int(params.height - 1 - tid.y), 0);
    float4 c = colourTarget.Load(texel);
    const float4 d = depthTarget.Load(texel);
    if (params.normalise != 0) {
        c = c.w > 1e-6 ? float4(c.rgb / c.w, 1.0) : float4(0.0);
    }
    const uint at = tid.y * params.width + tid.x;
    colour[at] = c;
    depth[at] = c.w > 0.0 ? d.r : 0.0;
}
