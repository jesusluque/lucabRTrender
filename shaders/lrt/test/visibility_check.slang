// Copyright (c) 2026 lucabRTrender contributors.
//
// A square of half size `half`, facing the camera at view depth `z`, checked
// pixel by pixel against what it analytically is: covered where the pixel's
// ray meets the square, at depth z, lit by the cosine to the eye.
module visibility_check;
import lrt.technique.camera;

struct PlaneParams {
    float z;
    float half;
    float colourR; float colourG; float colourB;
    float pad0; float pad1; float pad2;
};

StructuredBuffer<float4>     colour;
StructuredBuffer<float>      depth;
RWStructuredBuffer<uint>     counts;   // [coverage mismatches, colour mismatches, covered]
RWStructuredBuffer<float>    worst;    // [depth error]
ConstantBuffer<CameraParams> camera;
ConstantBuffer<PlaneParams>  plane;

[shader("compute")]
[numthreads(1, 1, 1)]
void planeCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint coverage = 0;
    uint colours = 0;
    uint covered = 0;
    float depthError = 0.0;
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            float3 origin;
            float3 direction;
            viewRay(camera, float2(x, y) + 0.5, origin, direction);
            const float3 at = origin + direction * plane.z;
            // Pixel centres exactly on an edge are the rasteriser's to decide.
            const float margin = max(abs(at.x), abs(at.y)) - plane.half;
            const uint i = y * camera.width + x;
            const bool drawn = depth[i] > 0.0;
            if (abs(margin) > 1e-4) {
                coverage += (drawn != (margin < 0.0)) ? 1u : 0u;
            }
            if (drawn) {
                ++covered;
                depthError = max(depthError, abs(depth[i] - plane.z));
                const float cosine = 1.0 / length(direction);
                const float3 expected = float3(plane.colourR, plane.colourG, plane.colourB) * cosine;
                colours += any(abs(colour[i].rgb - expected) > 1e-5) || abs(colour[i].w - 1.0) > 0.0 ? 1u : 0u;
            }
        }
    }
    counts[0] = coverage;
    counts[1] = colours;
    counts[2] = covered;
    worst[0] = depthError;
}
