// Copyright (c) 2026 lucabRTrender contributors.
//
// The camera's exposure, applied once over the composed frame: everything the
// camera sees -- meshes, splats, points, domes -- scaled by the same 2^stops,
// and nothing that is not a colour (the AOVs) touched. Alpha stays.
module exposure;

struct ExposureParams {
    float scale;   // 2^stops
    uint  pixels;
    uint  pad0; uint pad1;
};

RWStructuredBuffer<float4>     colour;
ConstantBuffer<ExposureParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void applyExposure(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.pixels) {
        const float4 c = colour[tid.x];
        colour[tid.x] = float4(c.rgb * params.scale, c.a);
    }
}
