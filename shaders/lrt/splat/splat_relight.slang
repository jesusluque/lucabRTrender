// Copyright (c) 2026 lucabRTrender contributors.
//
// What this scene's lights put on a splat, for both routes that draw one.
//
// WHY IT LIVES IN A MODULE OF ITS OWN
//
// Relighting used to be written into `splat_project.slang`, which is the tile
// rasteriser's -- so a cloud that asked to be relit was relit when rasterised
// and shown exactly as it was baked when path traced. The flag meant two
// things, and with a cloud converted from a mesh (whose colours are an albedo
// and not radiance anybody captured) the traced frame was the albedo itself:
// a white material came back at 1.0 and a black marble at 0.003, both with no
// light on them at all. This engine's rule everywhere else is that the routes
// agree; here they now share the function rather than a description of it.
//
// WHAT IT IS, AND WHAT IT IS NOT
//
// A splat is not a surface. The normal is its shortest axis -- the direction a
// flattened Gaussian faces -- turned towards the eye, and its albedo is the
// harmonics' constant term, the colour it shows on average. On top of that
// there is now a specular lobe, because without one a dark glossy material is
// black: the chess set's marble is `0.003` of diffuse albedo and everything
// you see of it in a render of the mesh is reflection. The lobe is GGX with
// Smith's height-correlated shadowing and Schlick's Fresnel, the same shape
// the surface lobes use, driven by a metallic and a roughness the splat
// carries where its cloud was converted from a mesh.
//
// One sample at the middle of each light, no second sample, and a shadow only
// where the caller measured one beforehand. That is what a splat gets: it has
// no BSDF to importance sample and no ray of its own in the rasteriser.
module splat_relight;
__exported import lrt.light.lights;
import lrt.common.color;

/// Relit light put where the blend expects a splat's colour.
///
/// A cloud is blended in the encoded space it was trained in and the finished
/// pixel is linearised (frame.slang). Relighting does not produce a trained
/// colour, it produces light -- so it goes in encoded, and the blend's own
/// linearise gives back exactly this. Without it every relit cloud was put
/// through the sRGB curve a second time: 0.5 came out 0.214 and a dark
/// material, where the curve is steepest, came out ten times too dark. That
/// is what made a converted mesh black.
public float3 relitForBlend(float3 lit, bool linearising) {
    return linearising ? linearToSrgb(lit) : lit;
}

/// What a splat reflects with.
public struct SplatSurface {
    public float3 position;
    public float3 normal;      ///< already turned towards the eye
    public float3 albedo;
    public float  metallic;    ///< 0 for a capture, which carries no such thing
    public float  roughness;   ///< 1 is the Lambert-only answer this had before
    /// What the material lets through, and the reason a splat can be
    /// translucent at all. A gaussian cannot refract; what it can do is send
    /// on the light that arrives from **behind** it, which is what glass lit
    /// from the far side does and what a purely reflective lobe never does.
    /// So the body of the material is split: `1 - transmission` of it faces
    /// the light, and `transmission` of it faces away. What stands behind the
    /// splat is seen through its own opacity, which the conversion lowers for
    /// the same reason.
    public float  transmission;
};

/// GGX, as the surface lobes write it: D, Smith's height-correlated V, and
/// Schlick's F folded into one weight for a light in direction `wi`.
float3 specularLobe(SplatSurface s, float3 wo, float3 wi) {
    const float3 h = normalize(wo + wi);
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float nDotL = max(dot(s.normal, wi), 0.0);
    const float nDotH = max(dot(s.normal, h), 0.0);
    const float vDotH = max(dot(wo, h), 0.0);
    if (nDotL <= 0.0) {
        return float3(0.0);
    }
    // A perfectly smooth splat would be a mirror the size of a texel, which
    // at one sample a light is a spike nothing resolves; the floor is the
    // same one an interactive frame puts on a surface.
    const float alpha = max(s.roughness * s.roughness, 1.0e-3);
    const float a2 = alpha * alpha;
    const float d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    const float distribution = a2 / max(3.14159265358979 * d * d, 1.0e-9);
    const float lambdaV = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
    const float lambdaL = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
    const float visibility = 0.5 / max(lambdaV + lambdaL, 1.0e-9);
    // A dielectric reflects 4% head on; a metal reflects its own colour.
    const float3 f0 = lerp(float3(0.04), s.albedo, s.metallic);
    const float3 fresnel = f0 + (float3(1.0) - f0) * pow(1.0 - vDotH, 5.0);
    return distribution * visibility * fresnel;
}

/// What a lobe of this roughness reflects of everything around it, as
/// Lazarov's analytic fit of the split-sum environment term gives it. It is
/// what a dome needs: a dome is not somewhere, so one sample of it is not a
/// highlight but a fraction of a whole sky, and the fraction is this.
float3 environmentBrdf(float3 f0, float roughness, float nDotV) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    const float4 r = roughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
    const float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

/// A dome, answered whole rather than sampled.
///
/// Every other light is somewhere: one sample at its centre is a direction
/// worth taking, and the lobe evaluated there is the highlight. A dome is
/// everywhere, and a single sample of it gives a splat one spike where a
/// surface would show a broad sheen -- which is exactly what the chess set's
/// polished marble is made of. So it is integrated instead: uniform radiance
/// over the hemisphere is `albedo * L` of diffuse, and `environmentBrdf * L`
/// along the mirror direction of specular, both closed forms.
float3 relitByDome(SplatSurface s, float3 wo, LightRecord light) {
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float3 mirror = reflect(-wo, s.normal);
    const LightHit above = lightHit(light, s.position, s.normal);
    const LightHit reflected = lightHit(light, s.position, mirror);
    const float3 f0 = lerp(float3(0.04), s.albedo, s.metallic);
    // A dome is on both sides of a splat at once, so the translucent half
    // sees exactly what the facing half sees and the two sum to the whole
    // body: a glass ball under a sky is its own colour, not a dark shell.
    const LightHit behind = lightHit(light, s.position, -s.normal);
    const float3 body = s.albedo * (1.0 - s.metallic);
    return body * ((1.0 - s.transmission) * above.radiance + s.transmission * behind.radiance) +
           environmentBrdf(f0, s.roughness, nDotV) * reflected.radiance;
}

/// The light `light` puts on `s`, seen from `wo`, with `through` of it
/// arriving.
///
/// Three terms: the body of the material facing the light, the body of it
/// facing away -- which is the translucent half, and is why a light behind a
/// glass splat lights it instead of leaving it black -- and the reflection,
/// which is on the lit side only.
public float3 relitByOne(SplatSurface s, float3 wo, LightSample ls, float through) {
    if (ls.pdf <= 0.0) {
        return float3(0.0);
    }
    const float nDotL = dot(s.normal, ls.wi);
    const float front = max(nDotL, 0.0);
    const float back = -nDotL;   // negative in front of the surface; the wrap below clamps it
    const float3 body = s.albedo * (1.0 - s.metallic) * (1.0 / 3.14159265358979);
    // Translucency is not transparency. What comes through a translucent
    // material has been scattered on the way, so it does not stop at the
    // terminator the way a reflection does and it does not draw what stands
    // behind: it wraps around the body and leaves softened. `kWrap` is how
    // far round it reaches -- half a hemisphere, which is what reads as jade
    // or wax rather than as a window.
    const float kWrap = 0.5;
    const float wrapped = saturate((back + kWrap) / (1.0 + kWrap));
    const float3 diffuse = body * ((1.0 - s.transmission) * front + s.transmission * wrapped);
    const float3 specular = specularLobe(s, wo, ls.wi) * front;
    return through * (diffuse + specular) * ls.radiance / ls.pdf;
}

/// Every light this splat is linked to, one sample each at its centre.
///
/// `shadowLights` of 0 means nothing measured a shadow, and each light is
/// taken whole; otherwise `shadowFactors[shadowBase + k]` is what reaches the
/// splat from light k (splat_shadow.slang).
public float3 relitSplat(SplatSurface s, float3 wo, StructuredBuffer<LightRecord> lights, uint lightCount,
                         uint categoriesLo, uint categoriesHi, StructuredBuffer<float> shadowFactors,
                         uint shadowBase, uint shadowLights) {
    float3 lit = float3(0.0);
    for (uint k = 0; k < lightCount; ++k) {
        const LightRecord light = lights[k];
        // Light linking reaches a cloud as it reaches a mesh: by one bit.
        if (!lightLinked(light.lightCategory, categoriesLo, categoriesHi)) {
            continue;
        }
        const float through = k < shadowLights ? shadowFactors[shadowBase + k] : 1.0;
        if (light.kind == kLightDome) {
            lit += through * relitByDome(s, wo, light);
            continue;
        }
        const LightSample ls = sampleLight(light, s.position, s.normal, float2(0.5, 0.5));
        if (!ls.valid) {
            continue;
        }
        lit += relitByOne(s, wo, ls, through);
    }
    return lit;
}

/// The rotation a packed quaternion stands for, columns in the frame's order.
public float3x3 splatRotation(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

/// The normal a splat stands for: its shortest axis, turned towards the eye.
/// `rotation` is the frame's three columns, `scale` its three sizes.
public float3 splatNormal(float3x3 rotation, float3 scale, float3 position, float3 eye) {
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    float3 normal = normalize(float3(rotation[0][shortest], rotation[1][shortest], rotation[2][shortest]));
    if (dot(normal, normalize(eye - position)) < 0.0) {
        normal = -normal;
    }
    return normal;
}

/// A point of the cloud in the world, by the rows a caller passes.
public float3 relightToWorld(float4 row0, float4 row1, float4 row2, float3 p) {
    return float3(dot(row0, float4(p, 1.0)), dot(row1, float4(p, 1.0)), dot(row2, float4(p, 1.0)));
}

/// A direction of the cloud in the world. The same rows, which is exact for a
/// rigid or uniformly scaled prim and skewed for any other -- the
/// approximation the shadow kernel already makes, and a splat's normal is an
/// approximation before this one touches it.
public float3 relightDirectionToWorld(float4 row0, float4 row1, float4 row2, float3 v) {
    const float3 turned = float3(dot(row0.xyz, v), dot(row1.xyz, v), dot(row2.xyz, v));
    return dot(turned, turned) > 1.0e-20 ? normalize(turned) : float3(0.0, 0.0, 1.0);
}
