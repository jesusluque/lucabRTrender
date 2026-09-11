// Copyright (c) 2026 lucabRTrender contributors.
//
// What a frame shows where it drew nothing: the domes, along the pixel's own
// ray. A dome is not a layer -- it has no depth, and giving it one would make
// the background read as covered -- so it is painted at the end, over what
// the frame left empty, opaque because nothing lies behind a dome.
module dome_background;
import camera;
import lrt.light.lights;

struct BackgroundParams {
    // view to world, rows of a 3x4
    float v00; float v01; float v02; float v03;
    float v10; float v11; float v12; float v13;
    float v20; float v21; float v22; float v23;
};

StructuredBuffer<LightRecord>    lights;
uniform uint                     lightCount;
RWStructuredBuffer<float4>       colour;
StructuredBuffer<float>          depth;
ConstantBuffer<CameraParams>     camera;
ConstantBuffer<BackgroundParams> background;

[shader("compute")]
[numthreads(16, 16, 1)]
void domeBackground(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    if (depth[at] > 0.0) {
        return;   // something was drawn here
    }
    float3 origin;
    float3 direction;
    viewRay(camera, float2(tid.xy) + 0.5, origin, direction);
    const float3x4 toWorld =
        float3x4(background.v00, background.v01, background.v02, background.v03, background.v10, background.v11,
                 background.v12, background.v13, background.v20, background.v21, background.v22, background.v23);
    const float3 world = normalize(mul(toWorld, float4(direction, 0.0)));
    float3 sky = float3(0.0);
    bool any = false;
    for (uint k = 0; k < lightCount; ++k) {
        if (lights[k].kind != kLightDome) {
            continue;
        }
        sky += lightEmission(lights[k]) * domeImage(lights[k], world);
        any = true;
    }
    if (any) {
        colour[at] = float4(sky, 1.0);
    }
}
