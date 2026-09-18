// Copyright (c) 2026 lucabRTrender contributors.
//
// What a frame shows where it drew nothing: the domes, along the pixel's own
// ray. A dome is not a layer -- it has no depth, and giving it one would make
// the background read as covered -- so it is painted at the end, over what
// the frame left empty, opaque because nothing lies behind a dome.
module dome_background;
import camera;
import lrt.light.lights_image;

struct BackgroundParams {
    // view to world, rows of a 3x4
    float v00; float v01; float v02; float v03;
    float v10; float v11; float v12; float v13;
    float v20; float v21; float v22; float v23;
};

StructuredBuffer<LightRecord>    lights;
uniform uint                     lightCount;
RWStructuredBuffer<float4>       colour;
ConstantBuffer<CameraParams>     camera;
ConstantBuffer<BackgroundParams> background;

[shader("compute")]
[numthreads(16, 16, 1)]
void domeBackground(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
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
        sky += domeBackground(lights[k], world);
        any = true;
    }
    if (!any) {
        return;
    }
    // BEHIND WHAT IS THERE, NOT INSTEAD OF IT -- and behind a pixel that is
    // only half covered too.
    //
    // The sky used to be painted where the depth buffer said nothing had been
    // drawn, and written over the pixel. That is right for a mesh, which
    // covers a pixel or does not, and wrong for a cloud, whose silhouette is
    // a ramp of partial coverage five pixels wide: every one of those has a
    // depth, so none of them got any sky, and each kept the splat's colour
    // premultiplied by a fifth or a half over black. Measured on the pawn's
    // collar, the pixel outside its edge read 0.087 with an alpha of 0.199 --
    // 0.199 times the body, and nothing of the 0.8 of sky behind it -- so
    // every silhouette in the model wore a dark fringe, and at the
    // magnification a viewer reaches that fringe is a row of black splinters.
    //
    // The sky is opaque and it is behind everything, so it is composited
    // under, by the pixel's own coverage, and the pixel is opaque afterwards.
    // Where the coverage is one this adds nothing, which is what it did for a
    // mesh all along.
    const float4 front = colour[at];
    colour[at] = float4(front.rgb + sky * (1.0 - saturate(front.a)), 1.0);
}

/// The same background in the light groups' planes: a camera ray that meets a
/// dome is the light path C<L>, which C.*<L.'NAME'> matches with `.*` empty,
/// so a dome's background is its group's. Every plane takes the pixel's
/// opacity, as the colour's w; a group with no dome in it keeps no light.
RWStructuredBuffer<float4> groupPlanes;
uniform uint               groupBase;    // the first group's plane, in pixels
uniform uint               groupCount;

[shader("compute")]
[numthreads(16, 16, 1)]
void domeBackgroundGroups(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    float3 origin;
    float3 direction;
    viewRay(camera, float2(tid.xy) + 0.5, origin, direction);
    const float3x4 toWorld =
        float3x4(background.v00, background.v01, background.v02, background.v03, background.v10, background.v11,
                 background.v12, background.v13, background.v20, background.v21, background.v22, background.v23);
    const float3 world = normalize(mul(toWorld, float4(direction, 0.0)));
    const uint pixels = camera.width * camera.height;
    bool anyDome = false;
    for (uint k = 0; k < lightCount; ++k) {
        anyDome = anyDome || lights[k].kind == kLightDome;
    }
    if (!anyDome) {
        return;
    }
    // The same coverage the beauty composites by, so that the groups still sum
    // to it where a silhouette only half covers its pixel. Read before the
    // beauty pass writes it, which is why that pass is dispatched second.
    const float behind = 1.0 - saturate(colour[at].a);
    for (uint g = 0; g < groupCount; ++g) {
        float3 sky = float3(0.0);
        for (uint k = 0; k < lightCount; ++k) {
            if (lights[k].kind == kLightDome && lights[k].group == g + 1) {
                sky += domeBackground(lights[k], world);
            }
        }
        const uint plane = groupBase + g * pixels + at;
        groupPlanes[plane] = float4(groupPlanes[plane].rgb + sky * behind, 1.0);
    }
}
