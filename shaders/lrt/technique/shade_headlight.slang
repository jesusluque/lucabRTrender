// Copyright (c) 2026 lucabRTrender contributors.
//
// A visibility buffer into an image, lit from the eye: displayColor times the
// cosine between the surface normal and the direction to the eye, with
// displayOpacity. Depth is view z (0: nothing), as every engine image carries
// it. The surface itself is reconstructed as surface.slang says.
module shade_headlight;
import surface;

Texture2D<uint4>                  visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>        colour;
RWStructuredBuffer<float>         depth;
ConstantBuffer<CameraParams>      camera;

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeHeadlight(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        colour[at] = float4(0.0);
        depth[at] = 0.0;
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const float3 displayColor = surfacePrimvar(s, kSlotDisplayColor, s.instance.colour).rgb;
    const float opacity = saturate(surfacePrimvar(s, kSlotDisplayOpacity, float4(s.instance.colour.w)).x);
    const float cosine = abs(dot(s.normal, normalize(s.viewDirection)));
    colour[at] = float4(displayColor * cosine * opacity, opacity);
    depth[at] = s.depth;
}
