// Copyright (c) 2026 lucabRTrender contributors.
//
// Per splat: where it lands, its 2D shape, its colour from this eye, its depth
// key, and exactly how many tiles its footprint touches. See frame.slang.
module splat_project;
import lrt.common.packing;
import lrt.common.sh;
import frame;
import lrt.common.edit;
// The core of the light module, which reads no texture: a dome is its colour
// here, since this kernel lives below material and has no table to sample.
// What a relit splat does with those lights is splat_relight's, shared with
// the ray traced route so that the two cannot drift.
import splat_relight;
import lrt.common.color;
import lrt.scene.splat_encoding;
// A shadow ray against the cloud's own proxies, where the caller built them.

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
StructuredBuffer<uint>      sh;
StructuredBuffer<uint>      pbr;      // metallic and roughness a splat, where the cloud carries them
StructuredBuffer<LightRecord> lights;
RWStructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<uint>    visible;      // 1 when the splat touches a tile
RWStructuredBuffer<uint>    tileRects;    // 4 per splat
RWStructuredBuffer<uint>    tilesTouched;
RWStructuredBuffer<uint>    depthKeys;
ConstantBuffer<FrameParams> params;

// Relighting's shadow, as splat_shadow.slang measured it: one float per
// (splat, light), the transmittance between that splat and that light's
// centre. It is a separate kernel because an inline ray is a feature a device
// may not have -- putting the RayQuery here made this kernel, and with it the
// whole splat rasteriser, unavailable on CUDA (measured: "entrypoint
// splatProject uses features that are not available").
StructuredBuffer<float>     shadowFactors;

void cull(uint slot) {
    // Every early return zeroes the record first: a slot keeps last frame's
    // splat otherwise (openFXplayer learned this one the hard way).
    tilesTouched[slot] = 0;
    visible[slot] = 0;
    depthKeys[slot] = 0x00FFFFFFu;
    tileRects[slot * 4] = 1;
    tileRects[slot * 4 + 1] = 1;
    tileRects[slot * 4 + 2] = 0;
    tileRects[slot * 4 + 3] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatProject(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;

    float4 posOpacity = positions[i];
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]),
                                    lowHalf(shape[row + 2])));
    // The SplatEdit, in the cloud's own space, before anything is projected.
    const EditFate fate = editFate(params.edit, posOpacity.xyz, posOpacity.w, scale);
    if (fate.removed) {
        cull(slot);
        return;
    }
    posOpacity.w = fate.opacity;

    const float3 t = toView(params, posOpacity.xyz);
    if (!(t.z > params.nearZ) || t.z > params.farZ) {
        cull(slot);
        return;
    }

    // 3D covariance in the cloud's space: R S S^T R^T.
    const float4 q = decodeQuaternion(shape[row]);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float3x3 r = float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
    const float3x3 m = float3x3(r[0] * scale, r[1] * scale, r[2] * scale);

    // T = J W, then the 2D covariance is (T M)(T M)^T.
    const bool ortho = params.orthographic != 0;
    float3 jx, jy;
    float2 centre;
    if (ortho) {
        jx = float3(params.focalX, 0.0, 0.0);
        jy = float3(0.0, params.focalY, 0.0);
        centre = float2(params.focalX * t.x + params.centreX, params.focalY * t.y + params.centreY);
    } else {
        const float invZ = 1.0 / t.z;
        centre = float2(params.focalX * t.x * invZ + params.centreX,
                        params.focalY * t.y * invZ + params.centreY);
        // The Jacobian is linearised at a point clamped to 1.3x the frustum:
        // outside it the affine approximation explodes and a splat far off
        // screen smears across the whole picture (3DGS does the same).
        const float limLoX = (-0.15 * float(params.width) - params.centreX) / params.focalX;
        const float limHiX = (1.15 * float(params.width) - params.centreX) / params.focalX;
        const float limLoY = (-0.15 * float(params.height) - params.centreY) / params.focalY;
        const float limHiY = (1.15 * float(params.height) - params.centreY) / params.focalY;
        const float tx = clamp(t.x * invZ, limLoX, limHiX) * t.z;
        const float ty = clamp(t.y * invZ, limLoY, limHiY) * t.z;
        jx = float3(params.focalX * invZ, 0.0, -params.focalX * tx * invZ * invZ);
        jy = float3(0.0, params.focalY * invZ, -params.focalY * ty * invZ * invZ);
    }
    const float3x3 wv = viewLinear(params);
    const float3 tx3 = mul(jx, wv);   // row vectors: J rows times W
    const float3 ty3 = mul(jy, wv);
    const float3 ax = mul(tx3, m);
    const float3 ay = mul(ty3, m);
    float a = dot(ax, ax);
    float b = dot(ax, ay);
    float c = dot(ay, ay);

    // A third of a pixel of dilation, and the Mip-Splatting 2D filter paying
    // its energy back as opacity (openFXplayer splat.slang; gsplat 'antialiased').
    const float detBefore = a * c - b * b;
    a += 0.3;
    c += 0.3;
    const float det = a * c - b * b;
    if (!(det > 1e-12)) {
        cull(slot);
        return;
    }
    float alpha = posOpacity.w;
    if (params.antialias != 0) {
        alpha *= sqrt(max(detBefore, 0.0) / det);
    }
    if (!(alpha >= 1.0 / 255.0)) {
        cull(slot);
        return;
    }
    const float3 conic = float3(c / det, -b / det, a / det);
    const float cutoff = cutoffPower(alpha);
    const float2 extent = footprintExtent(conic, cutoff);
    uint4 rect;
    tileRect(params, centre, extent, rect);
    if (rect.x > rect.z || rect.y > rect.w) {
        cull(slot);
        return;
    }

    // The exact count, with the same test the emit repeats.
    uint touched = 0;
    for (uint ty = rect.y; ty <= rect.w; ++ty) {
        for (uint tx = rect.x; tx <= rect.z; ++tx) {
            if (tileTouched(params, centre, conic, cutoff, tx, ty)) {
                touched += 1;
            }
        }
    }
    if (touched == 0) {
        cull(slot);
        return;
    }

    // Colour from this eye, in the cloud's space: the harmonics were trained
    // there, and a camera left in world space makes colours swim when the
    // cloud is turned.
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]),
                               highHalf(shape[row + 3]));
    const uint degree = min(degreeOf(params.restPerColour), params.shLimit);
    const float3 dir = normalize(posOpacity.xyz - float3(params.eyeX, params.eyeY, params.eyeZ));
    float3 colour = max(base + evaluateRest(sh, i * params.shWords, degree, dir), 0.0);
    if (params.relight != 0) {
        // LrtSplatLightingAPI: the radiance a capture was baked with replaced
        // by what this scene's lights put on it. What that means is in
        // splat_relight.slang, which the ray traced route reads too.
        // In the world, where the lights are: the cloud's own space is where
        // its harmonics were trained, and a light stands somewhere else.
        const float4 w0 = float4(params.w00, params.w01, params.w02, params.w03);
        const float4 w1 = float4(params.w10, params.w11, params.w12, params.w13);
        const float4 w2 = float4(params.w20, params.w21, params.w22, params.w23);
        const float3 eye = float3(params.eyeWorldX, params.eyeWorldY, params.eyeWorldZ);
        const float3 cloudNormal = splatNormal(r, scale, posOpacity.xyz,
                                               float3(params.eyeX, params.eyeY, params.eyeZ));
        SplatSurface surface;
        surface.position = relightToWorld(w0, w1, w2, posOpacity.xyz);
        surface.normal = relightDirectionToWorld(w0, w1, w2, cloudNormal);
        // The stored colour is in the space the cloud is blended in; an
        // albedo is light. Every cloud needs this, not only a converted one:
        // a capture's trained colour is sRGB too.
        surface.albedo = params.linearise != 0 ? srgbToLinear(base) : max(base, 0.0);
        surface.metallic = params.hasPbr != 0 ? pbrMetallic(pbr[i]) : 0.0;
        surface.roughness = params.hasPbr != 0 ? pbrRoughness(pbr[i]) : 1.0;
        surface.transmission = params.hasPbr != 0 ? pbrTransmission(pbr[i]) : 0.0;
        surface.lit = params.litBody != 0;
        const uint shadowLights = params.shadowRays != 0 ? params.shadowLights : 0;
        colour = relitForBlend(relitSplat(surface, normalize(eye - surface.position), lights,
                                          params.lightCount, params.categoriesLo, params.categoriesHi,
                                          shadowFactors, slot * params.shadowLights, shadowLights),
                               params.linearise != 0);
    }
    if (fate.graded) {
        colour = editGrade(params.edit, colour);
    }

    ProjRecord record;
    record.screen = float4(centre, t.z, cutoff);
    record.conic = float4(conic, alpha);
    record.colour = float4(colour, 0.0);
    proj[slot] = record;
    tileRects[slot * 4] = rect.x;
    tileRects[slot * 4 + 1] = rect.y;
    tileRects[slot * 4 + 2] = rect.z;
    tileRects[slot * 4 + 3] = rect.w;
    tilesTouched[slot] = touched;
    visible[slot] = 1;
    // Positive floats order as their bits do. The top 24 bits -- exponent and
    // 15 bits of mantissa, a relative depth step of 3e-5 -- sort in three
    // radix passes instead of four; splats nearer each other than that keep
    // index order, as exact ties always did.
    depthKeys[slot] = bit_cast<uint>(t.z) >> 7;
}
