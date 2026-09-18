// Copyright (c) 2026 lucabRTrender contributors.
//
// One instance's particle colours for this frame, into the colour buffer at
// `base`, and the opacity its SplatEdit leaves each particle (w). A particle's
// colour depends on the direction from the eye to its centre (the rasteriser's
// rule, and how 3DGS trained it), not on the ray, so it is evaluated once a
// frame here instead of once for every ray it reaches.
//
// A relit cloud is relit here, for the same reason and at the same moment: one
// answer a particle a frame. Until this it was relit only in the tile
// rasteriser, so `LrtSplatLightingAPI` meant one thing when a frame was
// rasterised and nothing at all when it was traced -- a converted mesh came
// back as its own albedo, unlit, which for a white material is 1.0 and for
// black marble is 0.003. What relighting is lives in splat_relight.slang, and
// both routes read it from there.
module rt_shade;
import lrt.common.packing;
import lrt.common.sh;
import lrt.scene.splat_encoding;
import lrt.splat.splat_relight;
import rt_common;

StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
StructuredBuffer<uint>     sh;
StructuredBuffer<uint>     pbr;      // metallic and roughness a splat, where the cloud carries them
StructuredBuffer<LightRecord> lights;
StructuredBuffer<float>    shadowFactors;   // unused here: a traced splat has no shadow pass of its own
RWStructuredBuffer<float4> colours;
ConstantBuffer<RtParams>   params;   // count, base, shWords, restPerColour, shLimit; v03/v13/v23: eye in cloud space

[shader("compute")]
[numthreads(256, 1, 1)]
void rtShade(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint row = i * 4;
    const float4 p = positions[i];
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    // The instance's SplatEdit decides the opacity this instance draws the
    // particle with, carried in w: zero is a particle the edit removed.
    const EditFate fate = editFate(params.edit, p.xyz, p.w, scale);
    if (fate.removed) {
        colours[params.base + i] = float4(0.0);
        return;
    }
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const uint degree = min(degreeOf(params.restPerColour), params.shLimit);
    const float3 eye = float3(params.v03, params.v13, params.v23);
    const float3 dir = normalize(p.xyz - eye);
    float3 colour = max(base + evaluateRest(sh, i * params.shWords, degree, dir), 0.0);
    if (params.relight != 0) {
        // In the world, where the lights are: the cloud's own space is where
        // its harmonics were trained, and a light stands somewhere else.
        const float4 w0 = float4(params.w00, params.w01, params.w02, params.w03);
        const float4 w1 = float4(params.w10, params.w11, params.w12, params.w13);
        const float4 w2 = float4(params.w20, params.w21, params.w22, params.w23);
        const float3x3 rotation = splatRotation(decodeQuaternion(shape[row + 0]));
        const float3 cloudNormal = splatNormal(rotation, scale, p.xyz, eye);
        const float3 eyeWorld = float3(params.eyeWorldX, params.eyeWorldY, params.eyeWorldZ);
        SplatSurface surface;
        surface.position = relightToWorld(w0, w1, w2, p.xyz);
        surface.normal = relightDirectionToWorld(w0, w1, w2, cloudNormal);
        surface.albedo = max(base, 0.0);
        surface.metallic = params.hasPbr != 0 ? pbrMetallic(pbr[i]) : 0.0;
        surface.roughness = params.hasPbr != 0 ? pbrRoughness(pbr[i]) : 1.0;
        // No shadow factors on this route: the rasteriser measures them in a
        // pass of its own before it projects, and a traced frame has no such
        // pass. Every light is taken whole.
        colour = relitSplat(surface, normalize(eyeWorld - surface.position), lights, params.lightCount,
                            params.categoriesLo, params.categoriesHi, shadowFactors, 0, 0);
    }
    if (fate.graded) {
        colour = editGrade(params.edit, colour);
    }
    colours[params.base + i] = float4(colour, fate.opacity);
}
