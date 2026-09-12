// Copyright (c) 2026 lucabRTrender contributors.
//
// A dome's image, and the warp that samples by it. This is the half of the
// light module that needs a texture table, kept apart so that a kernel
// without one can still light a surface: the splat projection lives below
// material in the module order and cannot have it.
//
// sampleLightImaged and lightPdfImaged are what a kernel with a table calls;
// they use the image where a dome has one and defer to the core where it does
// not.
module lights_image;
__exported import lrt.light.lights;
__exported import lrt.material.material_inputs;

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
/// One sample of `l`, by its image where it has one.
public LightSample sampleLightImaged(LightRecord l, float3 p, float3 n, float2 u) {
    if (l.kind == kLightDome) {
        if (domeHasImage(l)) {
            return domeImageSample(l, u);
        }
        LightSample s = sampleLight(l, p, n, u);
        s.radiance *= domeImage(l, s.wi);
        return s;
    }
    return sampleLight(l, p, n, u);
}

/// The density that sample was drawn with.
public float lightPdfImaged(LightRecord l, float3 p, float3 n, float3 wi) {
    if (l.kind == kLightDome && domeHasImage(l)) {
        return domeImagePdf(l, wi);
    }
    return lightPdf(l, p, n, wi);
}

/// What a dome puts in a direction nothing was drawn in.
public float3 domeBackground(LightRecord l, float3 wi) {
    return lightEmission(l) * domeImage(l, wi);
}
