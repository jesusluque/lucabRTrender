// Copyright (c) 2026 lucabRTrender contributors.
//
// The scene's lights as the device has them, in world space, and how a
// shading point samples one: a direction, the distance to shadow along, the
// radiance arriving and the solid-angle density it was drawn with. A light's
// colour is derived here, not on the host -- exposure, and the blackbody of a
// colour temperature -- so a record carries what USD authored.
module lights;
// gTextures lives here: a dome reads its image through the same table the
// materials sample, and the kernels that light a surface import both anyway.
__exported import lrt.material.material_inputs;

public static const uint kLightDistant = 0;
public static const uint kLightSphere  = 1;
public static const uint kLightDisk    = 2;
public static const uint kLightRect    = 3;
public static const uint kLightDome    = 4;

/// A light with no collection reaches every prim.
public static const uint kLightUnlinked = 0xFFFFFFFF;

public static const uint kLightShadow      = 1;   // casts one
public static const uint kLightNormalize   = 2;   // intensity is power, not radiance: divide by area
public static const uint kLightTemperature = 4;   // tint by the blackbody of `temperature`

public struct LightRecord {
    public uint   kind;
    public uint   flags;
    public float  sizeX;         // sphere and disk radius, rect width, distant angular diameter (radians)
    public float  sizeY;         // rect height
    public float3 colour;        // colour * intensity
    public float  exposure;
    public float  temperature;   // kelvin
    public float  coneCos;       // cosine of the shaping cone's half angle; -1: none
    public float  coneSoftness;
    public uint   texture;       // a dome's lat-long image; 0xFFFFFFFF: its colour alone
    public uint   sampler;
    public uint   lightCategory;    // the bit a prim needs for this light to light it
    public uint   shadowCategory;   // and for it to cast this light's shadow
    public float  cumulative;       // its share of the frame's power, accumulated
    public float4 row0;          // light to world, rows of a 3x4
    public float4 row1;
    public float4 row2;
};

/// Whether `l` reaches a surface in `categories`: USD resolves a light's
/// collection into a category the geometry it lights carries, so the test is
/// one bit -- and a light with no collection carries none and lights all.
public bool lightLinked(uint category, uint categoriesLo, uint categoriesHi) {
    if (category == kLightUnlinked) {
        return true;
    }
    return category < 32 ? (categoriesLo & (1u << category)) != 0 : (categoriesHi & (1u << (category - 32))) != 0;
}

/// One light out of many, in proportion to its power: a binary search over
/// the accumulated shares, so a pixel's cost stops growing with the number of
/// lights. The density it was chosen with comes back, to be divided out.
public struct LightChoice {
    public uint  index;
    public float probability;
    public bool  valid;
};

public LightChoice chooseLight(StructuredBuffer<LightRecord> lights, uint count, float power, float u) {
    LightChoice choice;
    choice.index = 0;
    choice.probability = 0.0;
    choice.valid = false;
    if (count == 0 || !(power > 0.0)) {
        return choice;
    }
    const float target = clamp(u, 0.0, 0.999999) * power;
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (lights[mid].cumulative <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const float before = lo == 0 ? 0.0 : lights[lo - 1].cumulative;
    choice.index = lo;
    choice.probability = max(lights[lo].cumulative - before, 1.0e-9) / power;
    choice.valid = true;
    return choice;
}

public struct LightSample {
    public float3 wi;         // from the shading point towards the light, unit
    public float  distance;   // how far to shadow along wi
    public float3 radiance;   // arriving at the point
    public float  pdf;        // per solid angle; 0 for a delta light
    public bool   delta;
    public bool   valid;
};

static const float kLightInfinity = 1.0e30;

float3 lightApply(LightRecord l, float3 v, float w) {
    return float3(dot(l.row0.xyz, v) + l.row0.w * w, dot(l.row1.xyz, v) + l.row1.w * w,
                  dot(l.row2.xyz, v) + l.row2.w * w);
}

/// The light's axes in world space. UsdLux shines down -Z and its shapes lie
/// in the xy plane.
public float3 lightAxisX(LightRecord l) { return normalize(float3(l.row0.x, l.row1.x, l.row2.x)); }
public float3 lightAxisY(LightRecord l) { return normalize(float3(l.row0.y, l.row1.y, l.row2.y)); }
public float3 lightAxisZ(LightRecord l) { return normalize(float3(l.row0.z, l.row1.z, l.row2.z)); }
public float3 lightOrigin(LightRecord l) { return float3(l.row0.w, l.row1.w, l.row2.w); }

/// Planck's law through the CIE curves, as a colour normalised to luminance
/// 1: the analytic fit USD's own renderers use (Krystek's uv, then sRGB).
public float3 blackbody(float kelvin) {
    const float t = clamp(kelvin, 1000.0, 15000.0);
    const float t2 = t * t;
    // Krystek's rational fit of the Planckian locus in CIE 1960 uv.
    const float u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t2) /
                    (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t2);
    const float v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t2) /
                    (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t2);
    const float denominator = 2.0 * u - 8.0 * v + 4.0;
    const float x = 3.0 * u / denominator;
    const float y = 2.0 * v / denominator;
    const float z = 1.0 - x - y;
    const float3 xyz = float3(x / max(y, 1e-6), 1.0, z / max(y, 1e-6));
    // XYZ to linear sRGB.
    const float3 rgb = float3(dot(float3(3.2404542, -1.5371385, -0.4985314), xyz),
                              dot(float3(-0.9692660, 1.8760108, 0.0415560), xyz),
                              dot(float3(0.0556434, -0.2040259, 1.0572252), xyz));
    const float3 positive = max(rgb, float3(0.0));
    const float luminance = dot(float3(0.2126, 0.7152, 0.0722), positive);
    return luminance > 0.0 ? positive / luminance : float3(1.0);
}

/// Where a direction lands in a lat-long image: u around the light's own
/// -Z, v from its +Y down, as UsdLux maps a dome.
public float2 domeUv(LightRecord l, float3 wi) {
    const float3 d = normalize(float3(dot(lightAxisX(l), wi), dot(lightAxisY(l), wi), dot(lightAxisZ(l), wi)));
    const float u = atan2(d.x, -d.z) * (0.5 / 3.14159265358979) + 0.5;
    const float v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / 3.14159265358979);
    return float2(u, v);
}

/// A dome's image along `wi`, white where it has none.
public float3 domeImage(LightRecord l, float3 wi) {
    if (l.texture == 0xFFFFFFFF) {
        return float3(1.0);
    }
    const float2 uv = domeUv(l, wi);
    bool found;
    const float4 value = sampleTexture(gTextures, l.texture, l.sampler, uv, float2(0.0), float2(0.0), found);
    return found ? value.rgb : float3(1.0);
}

/// The area a light's `normalize` divides by, 0 where it has none.
float lightArea(LightRecord l) {
    switch (l.kind) {
    case kLightSphere: return 4.0 * 3.14159265358979 * l.sizeX * l.sizeX;
    case kLightDisk:   return 3.14159265358979 * l.sizeX * l.sizeX;
    case kLightRect:   return l.sizeX * l.sizeY;
    default:           return 0.0;
    }
}

/// What the light emits, before any shaping: colour, intensity, exposure,
/// colour temperature, and `normalize` where it applies.
public float3 lightEmission(LightRecord l) {
    float3 emission = l.colour * exp2(l.exposure);
    if ((l.flags & kLightTemperature) != 0) {
        emission *= blackbody(l.temperature);
    }
    const float area = lightArea(l);
    if ((l.flags & kLightNormalize) != 0 && area > 0.0) {
        emission /= area;
    }
    return emission;
}

/// The shaping cone's falloff along `wi` (the direction the light travels is
/// -wi), 1 where the light has no cone.
float lightCone(LightRecord l, float3 wi) {
    if (l.coneCos <= -1.0) {
        return 1.0;
    }
    const float cosine = dot(-wi, lightAxisZ(l) * -1.0);   // angle from the light's -Z
    if (cosine < l.coneCos) {
        return 0.0;
    }
    if (l.coneSoftness <= 0.0) {
        return 1.0;
    }
    // Softness eats into the cone from its edge, as UsdLux describes it.
    const float inner = lerp(1.0, l.coneCos, 1.0 - saturate(l.coneSoftness));
    return inner > l.coneCos ? smoothstep(l.coneCos, inner, cosine) : 1.0;
}

float2 sampleDisk(float2 u) {
    const float r = sqrt(u.x);
    const float phi = 2.0 * 3.14159265358979 * u.y;
    return float2(r * cos(phi), r * sin(phi));
}

/// A direction in the cone of half angle acos(cosMax) around +z.
float3 sampleCone(float cosMax, float2 u) {
    const float cosTheta = lerp(cosMax, 1.0, u.x);
    const float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    const float phi = 2.0 * 3.14159265358979 * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

/// An orthonormal frame around `z`.
public void lightFrame(float3 z, out float3 x, out float3 y) {
    const float sign = z.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (sign + z.z);
    const float b = z.x * z.y * a;
    x = float3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    y = float3(b, sign + z.y * z.y * a, -z.y);
}

/// One texel of one level, read exactly. Filtering would blend neighbours,
/// and then the four children of a cell would not sum to their parent: the
/// descent's probabilities would stop telescoping and the closed-form density
/// would no longer be the one the samples were drawn with.
public float domeTexelOf(LightRecord l, int2 xy, uint level) {
    const TextureRecord record = gTextures.records[l.texture];
    const int2 size = int2(max(int(record.width) >> level, 1), max(int(record.height) >> level, 1));
    const int2 at = clamp(xy, int2(0, 0), size - 1);
    const float3 rgb = gTextures.textures[record.slot].Load(int3(at, int(level))).rgb;
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

/// The same by uv, for a density that is asked about a direction.
float domeLuminance(LightRecord l, float2 uv, float level) {
    const TextureRecord record = gTextures.records[l.texture];
    const uint mip = uint(max(level, 0.0));
    const int2 size = int2(max(int(record.width) >> mip, 1), max(int(record.height) >> mip, 1));
    return domeTexelOf(l, int2(int(uv.x * float(size.x)), int(uv.y * float(size.y))), mip);
}

/// A dome's image, importance sampled.
///
/// A lat-long texel covers 2 pi^2 sin(theta) du dv of solid angle, and the
/// probability of landing on one is its luminance over the image's total --
/// which the 1x1 level already holds as an average. So the density needs no
/// walk at all:
///
///     pdf(wi) = luminance(wi) / (average * 2 pi^2 * sin(theta))
///
/// and only the sampling descends the mip pyramid, choosing between the four
/// children of a cell by their luminance. Nothing is precomputed: the chain
/// the texture store already built is the distribution.
static const float kTwoPiSquared = 2.0 * 3.14159265358979 * 3.14159265358979;

/// The image's mean luminance, from the top of its chain.
float domeAverage(LightRecord l) {
    const TextureRecord record = gTextures.records[l.texture];
    const float top = log2(max(max(record.width, record.height), 1.0));
    return domeLuminance(l, float2(0.5, 0.5), ceil(top));
}

/// Whether a dome is worth sampling by its image rather than around the
/// surface. An image that barely varies is better served by the cosine: the
/// warp is uniform in uv, which crowds the poles and drops the cosine
/// altogether, and a plane under a flat dome came out sixteen times noisier
/// that way. One with a sun in it is the opposite. The 1x1 level gives the
/// mean and a middle level the spread, so the choice costs two samples.
public bool domeHasImage(LightRecord l) {
    // Off until the descent is right. It draws directions its own density
    // does not describe: a chi-square binned in the image itself -- where the
    // warp works, so no grid artefact can explain it -- reads z 61543 over a
    // million samples, and 127965 with a square image, so the asymmetry of a
    // lat-long is not the cause either. The chain telescopes to 0.7%, the uv
    // round trip is exact, and every sample agrees with lightPdf, which only
    // shows that the per-sample test compares the density with itself. What
    // is left is the split: left/right then top/bottom off one level does not
    // give the four children their own probabilities. A cosine around the
    // surface is unbiased meanwhile, and for a flat sky it is the better
    // estimator anyway (0.08% against 20% at the same count).
    if (l.texture == 0xFFFFFFFF) {
        return false;
    }
    if ((gTextures.records[l.texture].flags & 1u) == 0) {
        return false;
    }
    const TextureRecord record = gTextures.records[l.texture];
    const float top = ceil(log2(max(max(record.width, record.height), 1.0)));
    const float average = domeLuminance(l, float2(0.5, 0.5), top);
    if (!(average > 0.0)) {
        return false;
    }
    // Four quarters of the image against its mean: enough to tell a sky with
    // a sun from a sky without one, and it reads the chain already built.
    const float level = max(top - 1.0, 0.0);
    float spread = 0.0;
    for (uint k = 0; k < 4; ++k) {
        const float2 uv = float2(k % 2 == 0 ? 0.25 : 0.75, k < 2 ? 0.25 : 0.75);
        spread = max(spread, abs(domeLuminance(l, uv, level) - average));
    }
    return spread > 0.25 * average;
}

/// Where a direction lands in the image, and back again.
public float3 domeDirection(LightRecord l, float2 uv) {
    const float phi = (uv.x - 0.5) * 2.0 * 3.14159265358979;
    const float theta = uv.y * 3.14159265358979;
    const float sinTheta = sin(theta);
    // The inverse of domeUv: x and -z around the light's own axes, y up.
    const float3 local = float3(sinTheta * sin(phi), cos(theta), -sinTheta * cos(phi));
    return normalize(lightAxisX(l) * local.x + lightAxisY(l) * local.y + lightAxisZ(l) * local.z);
}

public float domeImagePdf(LightRecord l, float3 wi) {
    if (!domeHasImage(l)) {
        return 0.0;
    }
    const float average = domeAverage(l);
    if (!(average > 0.0)) {
        return 0.0;
    }
    const float2 uv = domeUv(l, wi);
    const float sinTheta = sin(uv.y * 3.14159265358979);
    if (sinTheta <= 1.0e-6) {
        return 0.0;
    }
    return domeLuminance(l, uv, 0.0) / (average * kTwoPiSquared * sinTheta);
}

/// Down the pyramid, one level at a time: within the current cell, split left
/// from right by luminance and take u.x with it, then top from bottom with
/// u.y. What comes out is a texel drawn in proportion to its luminance, and
/// the two numbers are spent rather than merely consumed.
public LightSample domeImageSample(LightRecord l, float2 u) {
    LightSample s;
    s.wi = float3(0.0, 0.0, 1.0);
    s.distance = 1.0e30;
    s.radiance = float3(0.0);
    s.pdf = 0.0;
    s.delta = false;
    s.valid = false;
    const TextureRecord record = gTextures.records[l.texture];
    const float levels = ceil(log2(max(max(record.width, record.height), 1.0)));
    float2 lo = float2(0.0, 0.0);
    float2 size = float2(1.0, 1.0);
    // Down the pyramid, one level at a time. A chain over a lat-long is not
    // square: 64 x 32 reaches one row while it still has columns, and from
    // there a cell has two children rather than four -- splitting an axis
    // that has no resolution left hands probability to texels that are not
    // there. So each level splits only the axes that still divide.
    const int2 full = int2(max(int(record.width), 1), max(int(record.height), 1));
    int2 cell = int2(0, 0);
    for (float level = levels - 1.0; level >= 0.0; level -= 1.0) {
        const uint mip = uint(level);
        const int2 size2 = int2(max(full.x >> mip, 1), max(full.y >> mip, 1));
        const int2 parent = int2(max(full.x >> (mip + 1), 1), max(full.y >> (mip + 1), 1));
        const bool splitX = size2.x > parent.x;
        const bool splitY = size2.y > parent.y;
        cell = int2(splitX ? cell.x * 2 : cell.x, splitY ? cell.y * 2 : cell.y);
        const uint children = (splitX ? 2u : 1u) * (splitY ? 2u : 1u);
        float w[4];
        float total = 0.0;
        for (uint k = 0; k < children; ++k) {
            const int2 step = int2(splitX ? int(k & 1u) : 0, splitY ? int(splitX ? (k >> 1) : k) : 0);
            w[k] = domeTexelOf(l, cell + step, mip);
            total += w[k];
        }
        uint chosen = children - 1;
        float before = 0.0;
        if (total > 0.0) {
            float target = u.x * total;
            for (uint k = 0; k + 1 < children; ++k) {
                if (target < w[k]) {
                    chosen = k;
                    break;
                }
                target -= w[k];
                before += w[k];
            }
            if (chosen == children - 1) {
                before = total - w[chosen];
            }
            // What is left of the number, inside the child it chose.
            u.x = saturate((u.x * total - before) / max(w[chosen], 1.0e-9));
        } else {
            chosen = min(uint(u.x * float(children)), children - 1);
            u.x = saturate(u.x * float(children) - float(chosen));
        }
        const int2 step = int2(splitX ? int(chosen & 1u) : 0, splitY ? int(splitX ? (chosen >> 1) : chosen) : 0);
        cell += step;
        size *= float2(splitX ? 0.5 : 1.0, splitY ? 0.5 : 1.0);
        lo += size * float2(float(step.x), float(step.y));
    }
    const float2 uv = lo + size * u;
    s.wi = domeDirection(l, uv);
    s.pdf = domeImagePdf(l, s.wi);
    s.radiance = lightEmission(l) * domeImage(l, s.wi);
    s.valid = s.pdf > 0.0 && any(s.radiance > float3(0.0));
    return s;
}

/// The density `sampleLight` draws `wi` with, as a function of direction:
/// what a chi-square asks for, and what MIS will weigh a hit by. Zero where
/// the light cannot be reached that way, and zero for a delta light, which
/// has no density at all.
public float lightPdf(LightRecord l, float3 p, float3 n, float3 wi) {
    switch (l.kind) {
    case kLightDistant: {
        const float cosMax = cos(0.5 * max(l.sizeX, 0.0));
        if (cosMax >= 1.0) {
            return 0.0;   // a delta direction
        }
        return dot(wi, lightAxisZ(l)) >= cosMax ? 1.0 / (2.0 * 3.14159265358979 * (1.0 - cosMax)) : 0.0;
    }
    case kLightDome:
        // By the image where it has one, and cosine weighted around the
        // surface where it does not.
        return domeHasImage(l) ? domeImagePdf(l, wi) : max(dot(n, wi), 0.0) / 3.14159265358979;
    case kLightSphere: {
        const float3 offset = lightOrigin(l) - p;
        const float distance2 = dot(offset, offset);
        const float radius = max(l.sizeX, 0.0);
        if (radius <= 0.0 || distance2 <= radius * radius) {
            return 0.0;   // a point, or inside the light
        }
        const float cosMax = sqrt(max(0.0, 1.0 - radius * radius / distance2));
        return dot(wi, normalize(offset)) >= cosMax ? 1.0 / (2.0 * 3.14159265358979 * (1.0 - cosMax)) : 0.0;
    }
    case kLightDisk:
    case kLightRect: {
        // Where the ray meets the light's plane, if it does so in front and
        // inside the shape: the area density turned into a solid angle one.
        const float3 normal = -lightAxisZ(l);
        const float facing = dot(normal, -wi);
        if (facing <= 0.0) {
            return 0.0;
        }
        const float3 centre = lightOrigin(l);
        const float denominator = dot(wi, -normal);
        if (denominator <= 0.0) {
            return 0.0;
        }
        const float t = dot(centre - p, -normal) / denominator;
        if (!(t > 0.0)) {
            return 0.0;
        }
        const float3 at = p + wi * t - centre;
        const float x = dot(at, lightAxisX(l));
        const float y = dot(at, lightAxisY(l));
        float area;
        if (l.kind == kLightDisk) {
            if (x * x + y * y > l.sizeX * l.sizeX) {
                return 0.0;
            }
            area = 3.14159265358979 * l.sizeX * l.sizeX;
        } else {
            if (abs(x) > 0.5 * l.sizeX || abs(y) > 0.5 * l.sizeY) {
                return 0.0;
            }
            area = l.sizeX * l.sizeY;
        }
        return area > 0.0 ? (t * t) / (facing * area) : 0.0;
    }
    default:
        return 0.0;
    }
}

/// One sample of `l` as seen from `p`, whose surface faces `n`. Only a dome
/// reads the normal: the rest are somewhere rather than everywhere.
public LightSample sampleLight(LightRecord l, float3 p, float3 n, float2 u) {
    LightSample s;
    s.wi = float3(0.0, 0.0, 1.0);
    s.distance = kLightInfinity;
    s.radiance = float3(0.0);
    s.pdf = 0.0;
    s.delta = false;
    s.valid = false;
    const float3 emission = lightEmission(l);
    switch (l.kind) {
    case kLightDistant: {
        // UsdLux shines down -Z, so the light is towards +Z, spread over a
        // disc of angular diameter sizeX.
        const float3 towards = lightAxisZ(l);
        const float cosMax = cos(0.5 * max(l.sizeX, 0.0));
        if (cosMax >= 1.0) {
            s.wi = towards;
            s.delta = true;
            s.pdf = 1.0;
        } else {
            float3 x, y;
            lightFrame(towards, x, y);
            const float3 local = sampleCone(cosMax, u);
            s.wi = normalize(x * local.x + y * local.y + towards * local.z);
            s.pdf = 1.0 / (2.0 * 3.14159265358979 * (1.0 - cosMax));
        }
        s.radiance = emission;
        break;
    }
    case kLightDome: {
        if (domeHasImage(l)) {
            return domeImageSample(l, u);
        }
        // Cosine weighted around the surface, which is where a dome's light
        // reaches it from: uniform over the sphere throws half its samples
        // below the horizon, and a plane under a constant dome came out with
        // 7% of noise at 4096 of them. The warp that follows the image's own
        // brightness (the plan's mip warp) is still to come, and is what a
        // dome with a sun in it needs.
        float3 x, y;
        lightFrame(n, x, y);
        const float2 d = sampleDisk(u);
        const float z = sqrt(max(0.0, 1.0 - dot(d, d)));
        s.wi = normalize(x * d.x + y * d.y + n * z);
        // The density lightPdf reports for this direction, to the letter: a
        // sample along the horizon carries no light and is not drawn at all,
        // rather than being given a floor its pdf does not have.
        s.pdf = max(dot(n, s.wi), 0.0) / 3.14159265358979;
        s.radiance = emission * domeImage(l, s.wi);
        break;
    }
    case kLightSphere: {
        const float3 centre = lightOrigin(l);
        const float3 offset = centre - p;
        const float distance2 = dot(offset, offset);
        const float radius = max(l.sizeX, 0.0);
        if (distance2 <= radius * radius) {
            return s;   // inside the light
        }
        const float distance = sqrt(distance2);
        if (radius <= 0.0) {
            s.wi = offset / distance;
            s.distance = distance;
            s.radiance = emission / distance2;
            s.delta = true;
            s.pdf = 1.0;
            break;
        }
        // The cone the sphere subtends: uniform in solid angle, which is the
        // density a plane's closed-form irradiance is written against.
        const float cosMax = sqrt(max(0.0, 1.0 - radius * radius / distance2));
        const float3 towards = offset / distance;
        float3 x, y;
        lightFrame(towards, x, y);
        const float3 local = sampleCone(cosMax, u);
        s.wi = normalize(x * local.x + y * local.y + towards * local.z);
        // Where the cone's ray meets the sphere.
        const float b = dot(s.wi, offset);
        const float c = distance2 - radius * radius;
        const float discriminant = max(0.0, b * b - c);
        s.distance = b - sqrt(discriminant);
        s.radiance = emission;
        s.pdf = 1.0 / (2.0 * 3.14159265358979 * (1.0 - cosMax));
        break;
    }
    case kLightDisk:
    case kLightRect: {
        const float3 normal = -lightAxisZ(l);   // it shines down -Z
        float3 at;
        float area;
        if (l.kind == kLightDisk) {
            const float2 d = sampleDisk(u) * max(l.sizeX, 0.0);
            at = lightOrigin(l) + lightAxisX(l) * d.x + lightAxisY(l) * d.y;
            area = 3.14159265358979 * l.sizeX * l.sizeX;
        } else {
            at = lightOrigin(l) + lightAxisX(l) * (u.x - 0.5) * l.sizeX +
                 lightAxisY(l) * (u.y - 0.5) * l.sizeY;
            area = l.sizeX * l.sizeY;
        }
        const float3 offset = at - p;
        const float distance2 = dot(offset, offset);
        if (distance2 <= 0.0 || area <= 0.0) {
            return s;
        }
        const float distance = sqrt(distance2);
        s.wi = offset / distance;
        const float cosine = dot(normal, -s.wi);
        if (cosine <= 0.0) {
            return s;   // the back of a one-sided light
        }
        s.distance = distance;
        s.radiance = emission;
        s.pdf = distance2 / (cosine * area);
        break;
    }
    default:
        return s;
    }
    s.radiance *= lightCone(l, s.wi);
    s.valid = s.pdf > 0.0 && any(s.radiance > float3(0.0));
    return s;
}
