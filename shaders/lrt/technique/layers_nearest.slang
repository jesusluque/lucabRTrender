// Copyright (c) 2026 lucabRTrender contributors.
//
// Two opaque layers (meshes, points) into one: per pixel the nearer of what
// each drew, by view z (0: nothing drawn).
module layers_nearest;

struct NearestParams {
    uint pixels;
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<float4>     colourA;
StructuredBuffer<float>      depthA;
StructuredBuffer<float4>     colourB;
StructuredBuffer<float>      depthB;
RWStructuredBuffer<float4>   colour;
RWStructuredBuffer<float>    depth;
ConstantBuffer<NearestParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void layersNearest(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.pixels) {
        return;
    }
    const float a = depthA[i];
    const float b = depthB[i];
    const bool takeA = a > 0.0 && (!(b > 0.0) || a <= b);
    colour[i] = takeA ? colourA[i] : colourB[i];
    depth[i] = takeA ? a : b;
}
