// Copyright (c) 2026 lucabRTrender contributors.
//
// The lobes MaterialX's BSDF nodes become, and the stack a material pushes
// them onto. Each lobe evaluates as its pbrlib node does in genglsl (the
// response, f times the cosine) and adds what a path tracer needs: an
// importance sample and its density. The stack samples one lobe chosen by its
// weighted directional albedo and weighs the sample by every lobe's density
// (one-sample MIS).
//
// Conventions: world-space unit directions, `wo` towards the viewer and `wi`
// towards the light; eval returns f * |cos theta_i|; a lobe faces its normal
// towards `wo`, as MaterialX's mx_forward_facing_normal does.
module lobes;
__exported import sampling;
__exported import microfacet;

public static const uint kLobeNone = 0;
public static const uint kLobeOrenNayar = 1;
public static const uint kLobeBurley = 2;
public static const uint kLobeTranslucent = 3;
public static const uint kLobeDielectric = 4;
public static const uint kLobeConductor = 5;
public static const uint kLobeSchlick = 6;     // generalized_schlick_bsdf
public static const uint kLobeSheen = 7;
public static const uint kLobeHair = 8;        // chiang_hair_bsdf: R, TT, TRT and the rest, on a fibre

public static const uint kFlagEnergyCompensation = 1;   // Oren-Nayar: OpenPBR's compensated form
public static const uint kFlagInside = 2;               // the surface is seen from inside: the ior inverts
public static const uint kFlagZeltner = 4;              // sheen: Zeltner's LTC instead of Imageworks'

public static const uint kScatterReflect = 0;
public static const uint kScatterTransmit = 1;
public static const uint kScatterBoth = 2;

public struct Lobe {
    public uint   kind;
    public uint   flags;
    public uint   scatter;
    public uint   pad0;
    public float3 weight;       // the node's weight, times what mixing and layering made of it
    public float  roughness;    // diffuse and sheen
    public float3 colour0;      // diffuse colour, dielectric tint, conductor n, Schlick F0
    public float  ior;          // dielectric
    public float3 colour1;      // conductor k, Schlick F90
    public float  exponent;     // Schlick
    public float3 colour2;      // Schlick F82
    public float  pad1;
    public float2 alpha;        // GGX
    public float2 pad2;
    public float3 normal;
    public float  pad3;
    public float3 tangent;      // anisotropy's, or the fibre's direction
    public float  pad4;
};

/// Hair keeps the Lobe at 128 bytes: colour0/1/2 are the R, TT and TRT
/// tints, alpha the R roughness (longitudinal variance v, azimuthal s),
/// roughness the cuticle angle (0..1, remapped to -pi/2..pi/2), ior the
/// fibre's; the absorption coefficient sits in pad1 and pad2, the TT
/// roughness in pad3 and pad4, the TRT roughness in exponent (Schlick's,
/// which hair has no use for) and pad0 (float bits in the uint). Growing the
/// struct by two float4 made the shadowed shading kernel draw the umbra
/// wrong on Metal (2583 pixels off a closed form that had been exact), and
/// putting the fields here made it exact again; the cause is unexplained.
public float3 hairAbsorption(Lobe lobe) { return float3(lobe.pad1, lobe.pad2.x, lobe.pad2.y); }
public float2 hairRoughnessTT(Lobe lobe) { return float2(lobe.pad3, lobe.pad4); }
public float2 hairRoughnessTRT(Lobe lobe) { return float2(lobe.exponent, asfloat(lobe.pad0)); }
public void setHair(inout Lobe lobe, float3 absorption, float2 roughnessTT, float2 roughnessTRT) {
    lobe.pad1 = absorption.x;
    lobe.pad2 = absorption.yz;
    lobe.pad3 = roughnessTT.x;
    lobe.pad4 = roughnessTT.y;
    lobe.exponent = roughnessTRT.x;
    lobe.pad0 = asuint(roughnessTRT.y);
}

public struct LobeSample {
    public float3 wi;
    public float3 weight;   // f * |cos| / pdf, lobe weights included
    public float  pdf;      // solid angle; for a delta lobe, its discrete probability
    public bool   delta;
    public bool   valid;
};

static const float3 kLuminance = float3(0.2126, 0.7152, 0.0722);

bool smooth(float2 alpha) {
    return max(alpha.x, alpha.y) < 1e-3;
}

Frame frameOf(Lobe lobe, float3 wo) {
    if (lobe.kind == kLobeHair) {
        return Frame(lobe.normal, lobe.tangent);   // the fibre's normal as the hit gave it
    }
    const float3 n = dot(lobe.normal, wo) < 0.0 ? -lobe.normal : lobe.normal;
    return Frame(n, lobe.tangent);
}

float relativeIor(Lobe lobe) {
    const float ior = max(lobe.ior, 1e-4);
    return (lobe.flags & kFlagInside) != 0 ? 1.0 / ior : ior;
}

float3 schlickF(Lobe lobe, float cosTheta) {
    return fresnelHoffmanSchlick(cosTheta, max(lobe.colour0, 0.0), max(lobe.colour2, 0.0), max(lobe.colour1, 0.0),
                                 lobe.exponent);
}

/// The transmission microfacet normal for wo (outside, z > 0) and wi (below),
/// relative ior eta; false where no refraction joins them.
bool transmissionNormal(float3 wo, float3 wi, float eta, out float3 h) {
    h = -(wo + eta * wi);
    const float length2 = dot(h, h);
    if (length2 < 1e-20) {
        return false;
    }
    h /= sqrt(length2);
    if (h.z < 0.0) {
        h = -h;
    }
    return dot(wo, h) > 0.0 && dot(wi, h) < 0.0;
}

// ---- hair (Chiang et al. 2016, as pbrt-v3 and MaterialX write it) ---------------
//
// Local frame: x the fibre's direction, z its normal at the hit, y = z cross x.
// theta is the angle off the normal plane (sin theta = w.x), phi the azimuth
// in that plane about the normal; h, the offset across the fibre, is what the
// hit's normal says of the outgoing direction: the fibre is a tube here, so
// the normal is the radial one and the offset follows. Lobes p = 0 (R), 1
// (TT), 2 (TRT) and 3 (the rest, as a geometric tail); each is a
// longitudinal density M_p over theta_i times an azimuthal one N_p over phi,
// times the attenuation A_p from the Fresnel terms and the absorption on the
// way through. With no absorption the A_p sum to 1, and so does the albedo:
// the white furnace holds exactly. The model is not reciprocal (A_p is the
// outgoing side's), as its authors' is not.

float hairI0(float x) {
    float v = 1.0;
    float n = 1.0;
    float d = 1.0;
    float f = 1.0;
    const float x2 = x * x;
    for (int i = 0; i < 9; ++i) {
        d *= 4.0 * (f * f);
        n *= x2;
        v += n / d;
        f += 1.0;
    }
    return v;
}

float hairLogI0(float x) {
    return x > 12.0 ? x + 0.5 * (-log(2.0 * kPi) + log(1.0 / x) + 1.0 / (8.0 * x)) : log(hairI0(x));
}

/// M_p: the longitudinal density over theta_i for variance v.
float hairMp(float sinThetaI, float cosThetaI, float sinThetaO, float cosThetaO, float v) {
    const float a = cosThetaO * cosThetaI / v;
    const float b = sinThetaO * sinThetaI / v;
    return v < 0.1 ? exp(hairLogI0(a) - b - 1.0 / v + 0.6931 + log(0.5 / v))
                   : exp(-b) * hairI0(a) / (2.0 * v * sinh(1.0 / v));
}

float hairLogistic(float x, float s) {
    x = -abs(x);
    const float f = exp(x / s);
    return f / (s * (1.0 + f) * (1.0 + f));
}

float hairLogisticCdf(float x, float s) {
    return 1.0 / (1.0 + exp(-x / s));
}

static const float kHairLogisticScale = 0.626657;   // sqrt(pi / 8): Chiang, appendix A

float hairTrimmedLogistic(float x, float s) {
    s *= kHairLogisticScale;
    return hairLogistic(x, s) / (hairLogisticCdf(kPi, s) - hairLogisticCdf(-kPi, s));
}

float hairSampleTrimmedLogistic(float u, float s) {
    s *= kHairLogisticScale;
    const float a = hairLogisticCdf(-kPi, s);
    const float k = hairLogisticCdf(kPi, s) - a;
    const float x = -s * log(1.0 / (u * k + a) - 1.0);
    return clamp(x, -kPi, kPi);
}

float hairPhi(int p, float gammaO, float gammaT) {
    return 2.0 * float(p) * gammaT - 2.0 * gammaO + float(p) * kPi;
}

/// N_p: the azimuthal density over phi = phi_i - phi_o.
float hairNp(float phi, int p, float s, float gammaO, float gammaT) {
    if (p >= 3) {
        return 0.5 / kPi;
    }
    float dphi = phi - hairPhi(p, gammaO, gammaT);
    dphi -= 2.0 * kPi * floor((dphi + kPi) / (2.0 * kPi));   // into (-pi, pi]
    return hairTrimmedLogistic(dphi, s);
}

/// What the outgoing direction fixes: theta_o, gamma_o and gamma_t, the
/// attenuations A_p and the roughnesses; shared by eval, pdf and sample.
struct HairState {
    float sinThetaO, cosThetaO;
    float gammaO, gammaT;
    float phiO;          // wo's azimuth in the normal plane
    float3 ap[4];        // the attenuations, tints folded in
    float apPdf[4];      // each lobe's share, for choosing one
    float v[4];          // longitudinal variances
    float s[4];          // azimuthal roughnesses
    float alpha;         // the cuticle's tilt, radians
};

HairState hairState(Lobe lobe, float3 wo) {
    HairState h;
    h.sinThetaO = clamp(wo.x, -1.0, 1.0);
    h.cosThetaO = sqrt(max(1.0 - h.sinThetaO * h.sinThetaO, 0.0));
    h.phiO = atan2(wo.z, wo.y);
    // The offset across the fibre from the radial normal: the outgoing
    // direction projected into the normal plane against it.
    const float2 perp = float2(wo.y, wo.z);
    const float perpLength = length(perp);
    const float cosGammaO = perpLength > 1.0e-8 ? clamp(wo.z / perpLength, -1.0, 1.0) : 1.0;
    float sinGammaO = sqrt(max(1.0 - cosGammaO * cosGammaO, 0.0));
    if (wo.y > 0.0) {
        sinGammaO = -sinGammaO;
    }
    h.gammaO = asin(clamp(sinGammaO, -1.0, 1.0));
    const float ior = max(lobe.ior, 1.0001);
    const float sinThetaT = h.sinThetaO / ior;
    const float cosThetaT = sqrt(max(1.0 - sinThetaT * sinThetaT, 0.0));
    const float etaP = sqrt(max(ior * ior - h.sinThetaO * h.sinThetaO, 0.0)) / max(h.cosThetaO, kEpsilon);
    const float sinGammaT = clamp(sinGammaO / etaP, -1.0, 1.0);
    const float cosGammaT = sqrt(1.0 - sinGammaT * sinGammaT);
    h.gammaT = asin(sinGammaT);
    const float f = fresnelDielectric(h.cosThetaO * cosGammaO, ior);
    const float3 T = exp(-max(hairAbsorption(lobe), 0.0) * (2.0 * cosGammaT / max(cosThetaT, kEpsilon)));
    const float3 tintR = max(lobe.colour0, 0.0);
    const float3 tintTT = max(lobe.colour1, 0.0);
    const float3 tintTRT = max(lobe.colour2, 0.0);
    h.ap[0] = float3(f);
    h.ap[1] = (1.0 - f) * (1.0 - f) * T;
    h.ap[2] = h.ap[1] * T * f;
    h.ap[3] = h.ap[2] * T * f / max(float3(1.0) - T * f, kEpsilon);
    h.ap[0] *= tintR;
    h.ap[1] *= tintTT;
    h.ap[2] *= tintTRT;
    h.ap[3] *= tintTRT;
    float total = 0.0;
    for (int p = 0; p < 4; ++p) {
        h.apPdf[p] = dot(h.ap[p], kLuminance);
        total += h.apPdf[p];
    }
    for (int p = 0; p < 4; ++p) {
        h.apPdf[p] = total > 0.0 ? h.apPdf[p] / total : (p == 0 ? 1.0 : 0.0);
    }
    const float2 rR = clamp(lobe.alpha, 0.001, 1.0);
    const float2 rTT = clamp(hairRoughnessTT(lobe), 0.001, 1.0);
    const float2 rTRT = clamp(hairRoughnessTRT(lobe), 0.001, 1.0);
    h.v[0] = rR.x;   h.s[0] = rR.y;
    h.v[1] = rTT.x;  h.s[1] = rTT.y;
    h.v[2] = rTRT.x; h.s[2] = rTRT.y;
    h.v[3] = rTRT.x; h.s[3] = rTRT.y;
    h.alpha = lobe.roughness * kPi - 0.5 * kPi;
    return h;
}

/// theta_o tilted by the cuticle for lobe p: 2 alpha, -alpha, -4 alpha, none.
/// The tilt goes on the outgoing angle, as pbrt puts it, and not on the
/// incoming one as MaterialX's eval does: shifting theta_i changes the
/// measure cos(theta_i) d theta_i the density is normalised against, and a
/// sampler drawing from the shifted density no longer matches the pdf (a
/// chi-square of z 126 said so); shifting theta_o leaves the measure alone.
/// The lobes' peaks are the same either way.
void hairTilt(HairState h, int p, out float sinTilted, out float cosTilted) {
    if (p >= 3 || h.alpha == 0.0) {
        sinTilted = h.sinThetaO;
        cosTilted = h.cosThetaO;
        return;
    }
    const float m = 2.0 - float(p) * 3.0;
    const float sa = sin(m * h.alpha);
    const float ca = cos(m * h.alpha);
    sinTilted = clamp(h.sinThetaO * ca + h.cosThetaO * sa, -1.0, 1.0);
    cosTilted = sqrt(max(1.0 - sinTilted * sinTilted, 0.0));
}

/// f |cos theta_i| for the pair (what evalLocal returns for every lobe), and
/// the density of drawing wi: the sums over the lobes. M_p is normalised
/// against cos(theta_i) d theta_i and N_p against d phi, so the product is a
/// density over solid angle -- the pdf as it stands, and f times the cosine.
void hairEvalPdf(HairState h, float3 wi, out float3 f, out float pdf) {
    const float sinThetaI = clamp(wi.x, -1.0, 1.0);
    const float cosThetaI = sqrt(max(1.0 - sinThetaI * sinThetaI, 0.0));
    const float phi = atan2(wi.z, wi.y) - h.phiO;
    f = float3(0.0);
    pdf = 0.0;
    for (int p = 0; p < 4; ++p) {
        float sinTilted, cosTilted;
        hairTilt(h, p, sinTilted, cosTilted);
        const float mp = hairMp(sinThetaI, cosThetaI, sinTilted, cosTilted, h.v[p]);
        const float np = hairNp(phi, p, h.s[p], h.gammaO, h.gammaT);
        f += mp * np * h.ap[p];
        pdf += mp * np * h.apPdf[p];
    }
}

float3 hairSampleDirection(HairState h, float3 u, out int chosen) {
    // The lobe, by its share of the attenuation.
    int p = 0;
    float pick = u.z;
    for (; p < 3; ++p) {
        if (pick < h.apPdf[p]) {
            break;
        }
        pick -= h.apPdf[p];
    }
    chosen = p;
    // What is left of the choice, rescaled: a fourth number, independent of
    // the other two. (Reusing u.y's low bits for it tied theta_i to phi_i
    // and showed as a chi-square of z 4.8.)
    const float u3 = clamp(pick / max(h.apPdf[p], 1.0e-6), 0.0, 0.999999);
    // theta_i from M_p about the lobe's tilted theta_o (pbrt's draw).
    float sinThetaOp, cosThetaOp;
    hairTilt(h, p, sinThetaOp, cosThetaOp);
    const float v = h.v[p];
    const float u0 = max(u.x, 1.0e-5);
    const float cosTheta = 1.0 + v * log(u0 + (1.0 - u0) * exp(-2.0 / v));
    const float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    const float cosPhi = cos(2.0 * kPi * u.y);
    const float sinThetaI = clamp(-cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp, -1.0, 1.0);
    const float cosThetaI = sqrt(max(1.0 - sinThetaI * sinThetaI, 0.0));
    // phi_i about phi_o: the lobe's centre plus a trimmed-logistic draw.
    float dphi;
    if (p < 3) {
        dphi = hairPhi(p, h.gammaO, h.gammaT) + hairSampleTrimmedLogistic(u3, h.s[p]);
    } else {
        dphi = 2.0 * kPi * u3;
    }
    const float phiI = h.phiO + dphi;
    return float3(sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI));
}

// ---- one lobe, local directions ----------------------------------------------

float3 microfacetReflection(Lobe lobe, float3 wo, float3 wi, float3 F) {
    const float3 h = normalize(wo + wi);
    const float avg = averageAlpha(lobe.alpha);
    const float D = ggxD(h, lobe.alpha);
    const float G = ggxG2(max(wi.z, kEpsilon), wo.z, avg);
    const float3 comp = ggxEnergyCompensation(wo.z, avg, F);
    return D * F * G * comp / (4.0 * wo.z);
}

float3 microfacetTransmission(Lobe lobe, float3 wo, float3 wi, float eta, float F) {
    float3 h;
    if (!transmissionNormal(wo, wi, eta, h)) {
        return float3(0.0);
    }
    const float oh = dot(wo, h);
    const float ih = dot(wi, h);
    const float denominator = oh + eta * ih;
    const float D = ggxD(h, lobe.alpha);
    const float G = ggxG2(max(-wi.z, kEpsilon), wo.z, averageAlpha(lobe.alpha));
    return float3(abs(oh) * abs(ih) * eta * eta * (1.0 - F) * D * G / (wo.z * denominator * denominator));
}

float reflectionPdf(Lobe lobe, float3 wo, float3 wi) {
    const float3 h = normalize(wo + wi);
    return ggxVisibleNormalPdf(wo, h, lobe.alpha) / (4.0 * max(dot(wo, h), kEpsilon));
}

float transmissionPdf(Lobe lobe, float3 wo, float3 wi, float eta, out float F) {
    float3 h;
    F = 1.0;
    if (!transmissionNormal(wo, wi, eta, h)) {
        return 0.0;
    }
    const float oh = dot(wo, h);
    const float ih = dot(wi, h);
    const float denominator = oh + eta * ih;
    return ggxVisibleNormalPdf(wo, h, lobe.alpha) * eta * eta * abs(ih) / (denominator * denominator);
}

/// How likely a dielectric or Schlick lobe is to reflect at microfacet normal h.
float reflectChance(Lobe lobe, float cosine) {
    if (lobe.scatter == kScatterReflect) {
        return 1.0;
    }
    if (lobe.scatter == kScatterTransmit) {
        return 0.0;
    }
    if (lobe.kind == kLobeDielectric) {
        return fresnelDielectric(cosine, relativeIor(lobe));
    }
    return saturate(dot(schlickF(lobe, cosine), float3(1.0 / 3.0)));
}

float schlickIor(Lobe lobe) {
    return f0ToIor(dot(max(lobe.colour0, 0.0), float3(1.0 / 3.0)));
}

float3 evalLocal(Lobe lobe, float3 wo, float3 wi, float3 woWorld, float3 wiWorld) {
    if (lobe.kind == kLobeHair) {
        float3 f;
        float pdf;
        hairEvalPdf(hairState(lobe, wo), wi, f, pdf);
        return f;
    }
    switch (lobe.kind) {
    case kLobeOrenNayar: {
        if (wi.z <= 0.0) {
            return float3(0.0);
        }
        const float LdotV = clamp(dot(wiWorld, woWorld), kEpsilon, 1.0);
        const float3 diffuse = (lobe.flags & kFlagEnergyCompensation) != 0
                                   ? orenNayarCompensated(wo.z, wi.z, LdotV, lobe.roughness, lobe.colour0)
                                   : orenNayar(wo.z, wi.z, LdotV, lobe.roughness) * lobe.colour0;
        return diffuse * wi.z * kInvPi;
    }
    case kLobeBurley: {
        if (wi.z <= 0.0) {
            return float3(0.0);
        }
        const float LdotH = clamp(dot(wi, normalize(wi + wo)), kEpsilon, 1.0);
        return lobe.colour0 * burley(wo.z, wi.z, LdotH, lobe.roughness) * wi.z * kInvPi;
    }
    case kLobeTranslucent:
        return wi.z < 0.0 ? lobe.colour0 * (-wi.z) * kInvPi : float3(0.0);
    case kLobeConductor: {
        if (wi.z <= 0.0 || smooth(lobe.alpha)) {
            return float3(0.0);
        }
        const float VdotH = clamp(dot(wo, normalize(wo + wi)), kEpsilon, 1.0);
        return microfacetReflection(lobe, wo, wi, fresnelConductor(VdotH, lobe.colour0, lobe.colour1));
    }
    case kLobeDielectric:
    case kLobeSchlick: {
        if (smooth(lobe.alpha)) {
            return float3(0.0);
        }
        const bool dielectric = lobe.kind == kLobeDielectric;
        const float3 tint = dielectric ? max(lobe.colour0, 0.0) : float3(1.0);
        if (wi.z > 0.0) {
            if (lobe.scatter == kScatterTransmit) {
                return float3(0.0);
            }
            const float VdotH = clamp(dot(wo, normalize(wo + wi)), kEpsilon, 1.0);
            const float3 F = dielectric ? float3(fresnelDielectric(VdotH, relativeIor(lobe))) : schlickF(lobe, VdotH);
            return tint * microfacetReflection(lobe, wo, wi, F);
        }
        if (lobe.scatter == kScatterReflect) {
            return float3(0.0);
        }
        const float eta = dielectric ? relativeIor(lobe) : schlickIor(lobe);
        float3 h;
        if (!transmissionNormal(wo, wi, eta, h)) {
            return float3(0.0);
        }
        const float F = dielectric ? fresnelDielectric(dot(wo, h), eta)
                                   : saturate(dot(schlickF(lobe, dot(wo, h)), float3(1.0 / 3.0)));
        return tint * microfacetTransmission(lobe, wo, wi, eta, F);
    }
    case kLobeSheen: {
        if (wi.z <= 0.0) {
            return float3(0.0);
        }
        if ((lobe.flags & kFlagZeltner) == 0) {
            const float3 h = normalize(wi + wo);
            return lobe.colour0 * imageworksSheen(wi.z, wo.z, h.z, lobe.roughness) * wi.z;
        }
        const float r = clamp(lobe.roughness, 0.01, 1.0);
        const float3 x = abs(wo.x) + abs(wo.y) > 1e-7 ? normalize(float3(wo.xy, 0.0)) : float3(1.0, 0.0, 0.0);
        const float3 y = cross(float3(0.0, 0.0, 1.0), x);
        const float3 w = float3(dot(wi, x), dot(wi, y), wi.z);
        const float aInv = zeltnerLtcAInv(wo.z, r);
        const float bInv = zeltnerLtcBInv(wo.z, r);
        const float3 original = float3(aInv * w.x + bInv * w.z, aInv * w.y, w.z);
        const float length2 = dot(original, original);
        const float D = max(original.z, 0.0) * kInvPi * square(aInv / length2);
        return lobe.colour0 * zeltnerSheenDirectionalAlbedo(wo.z, r) * D;
    }
    default:
        return float3(0.0);
    }
}

float pdfLocal(Lobe lobe, float3 wo, float3 wi) {
    if (lobe.kind == kLobeHair) {
        float3 f;
        float pdf;
        hairEvalPdf(hairState(lobe, wo), wi, f, pdf);
        return pdf;
    }
    switch (lobe.kind) {
    case kLobeOrenNayar:
    case kLobeBurley:
        return wi.z > 0.0 ? wi.z * kInvPi : 0.0;
    case kLobeTranslucent:
        return wi.z < 0.0 ? -wi.z * kInvPi : 0.0;
    case kLobeConductor:
        return wi.z > 0.0 && !smooth(lobe.alpha) ? reflectionPdf(lobe, wo, wi) : 0.0;
    case kLobeDielectric:
    case kLobeSchlick: {
        if (smooth(lobe.alpha)) {
            return 0.0;
        }
        if (wi.z > 0.0) {
            if (lobe.scatter == kScatterTransmit) {
                return 0.0;
            }
            const float3 h = normalize(wo + wi);
            return reflectionPdf(lobe, wo, wi) * reflectChance(lobe, dot(wo, h));
        }
        if (lobe.scatter == kScatterReflect) {
            return 0.0;
        }
        const float eta = lobe.kind == kLobeDielectric ? relativeIor(lobe) : schlickIor(lobe);
        float3 h;
        if (!transmissionNormal(wo, wi, eta, h)) {
            return 0.0;
        }
        float unused;
        return transmissionPdf(lobe, wo, wi, eta, unused) * (1.0 - reflectChance(lobe, dot(wo, h)));
    }
    case kLobeSheen: {
        if (wi.z <= 0.0) {
            return 0.0;
        }
        if ((lobe.flags & kFlagZeltner) == 0) {
            return wi.z * kInvPi;
        }
        const float r = clamp(lobe.roughness, 0.01, 1.0);
        const float3 x = abs(wo.x) + abs(wo.y) > 1e-7 ? normalize(float3(wo.xy, 0.0)) : float3(1.0, 0.0, 0.0);
        const float3 y = cross(float3(0.0, 0.0, 1.0), x);
        const float3 w = float3(dot(wi, x), dot(wi, y), wi.z);
        const float aInv = zeltnerLtcAInv(wo.z, r);
        const float bInv = zeltnerLtcBInv(wo.z, r);
        const float3 original = float3(aInv * w.x + bInv * w.z, aInv * w.y, w.z);
        const float length2 = dot(original, original);
        return max(original.z, 0.0) * kInvPi * square(aInv / length2);
    }
    default:
        return 0.0;
    }
}

// ---- one lobe, world directions ---------------------------------------------

public float3 lobeEval(Lobe lobe, float3 wo, float3 wi) {
    const Frame frame = frameOf(lobe, wo);
    float3 o = frame.toLocal(wo);
    o.z = max(o.z, kEpsilon);
    return evalLocal(lobe, o, frame.toLocal(wi), wo, wi);
}

public float lobePdf(Lobe lobe, float3 wo, float3 wi) {
    const Frame frame = frameOf(lobe, wo);
    float3 o = frame.toLocal(wo);
    o.z = max(o.z, kEpsilon);
    return pdfLocal(lobe, o, frame.toLocal(wi));
}

/// Without the lobe's weight: the fraction of light from `wo` it returns,
/// as MaterialX's directional albedo fits have it.
public float3 lobeAlbedo(Lobe lobe, float3 wo) {
    const float NdotV = clamp(abs(dot(lobe.normal, wo)), kEpsilon, 1.0);
    const float avg = averageAlpha(max(lobe.alpha, kEpsilon));
    switch (lobe.kind) {
    case kLobeOrenNayar:
        return (lobe.flags & kFlagEnergyCompensation) != 0
                   ? orenNayarCompensatedDirectionalAlbedo(NdotV, lobe.roughness, lobe.colour0)
                   : orenNayarDirectionalAlbedo(NdotV, lobe.roughness) * lobe.colour0;
    case kLobeBurley:
        return burleyDirectionalAlbedo(NdotV, lobe.roughness) * lobe.colour0;
    case kLobeTranslucent:
        return lobe.colour0;
    case kLobeConductor: {
        const float3 F = fresnelConductor(NdotV, lobe.colour0, lobe.colour1);
        return ggxDirectionalAlbedo(NdotV, avg, F, F) * ggxEnergyCompensation(NdotV, avg, F);
    }
    case kLobeDielectric: {
        const float F0 = iorToF0(relativeIor(lobe));
        const float3 F = float3(fresnelDielectric(NdotV, relativeIor(lobe)));
        const float3 reflected = ggxDirectionalAlbedo(NdotV, avg, float3(F0), float3(1.0)) *
                                 ggxEnergyCompensation(NdotV, avg, F);
        const float3 tint = max(lobe.colour0, 0.0);
        if (lobe.scatter == kScatterReflect) {
            return tint * reflected;
        }
        if (lobe.scatter == kScatterTransmit) {
            return tint * saturate(1.0 - reflected);
        }
        return tint;
    }
    case kLobeSchlick: {
        const float3 F = schlickF(lobe, NdotV);
        const float3 reflected = ggxDirectionalAlbedo(NdotV, avg, max(lobe.colour0, 0.0), max(lobe.colour1, 0.0)) *
                                 ggxEnergyCompensation(NdotV, avg, F);
        if (lobe.scatter == kScatterReflect) {
            return reflected;
        }
        if (lobe.scatter == kScatterTransmit) {
            return saturate(1.0 - reflected);
        }
        return float3(1.0);
    }
    case kLobeSheen:
        return lobe.colour0 * ((lobe.flags & kFlagZeltner) != 0
                                   ? zeltnerSheenDirectionalAlbedo(NdotV, clamp(lobe.roughness, 0.01, 1.0))
                                   : imageworksSheenDirectionalAlbedo(NdotV, lobe.roughness));
    case kLobeHair: {
        // The attenuations' sum: what the fibre returns of what reaches it,
        // the lobes' densities integrating to one each.
        const Frame frame = frameOf(lobe, wo);
        const HairState h = hairState(lobe, frame.toLocal(wo));
        return h.ap[0] + h.ap[1] + h.ap[2] + h.ap[3];
    }
    default:
        return float3(0.0);
    }
}

/// One direction from the lobe alone; its weight excludes the lobe's weight.
public LobeSample lobeSample(Lobe lobe, float3 woWorld, float3 u) {
    LobeSample s;
    s.wi = float3(0.0);
    s.weight = float3(0.0);
    s.pdf = 0.0;
    s.delta = false;
    s.valid = false;
    const Frame frame = frameOf(lobe, woWorld);
    float3 wo = frame.toLocal(woWorld);
    wo.z = max(wo.z, kEpsilon);
    float3 wi = float3(0.0);
    if (lobe.kind == kLobeHair) {
        int chosen;
        wi = hairSampleDirection(hairState(lobe, wo), u, chosen);
        s.wi = frame.toWorld(wi);
        s.pdf = pdfLocal(lobe, wo, wi);
        if (s.pdf <= 0.0) {
            return s;
        }
        s.weight = evalLocal(lobe, wo, wi, woWorld, s.wi) / s.pdf;
        s.valid = true;
        return s;
    }
    switch (lobe.kind) {
    case kLobeOrenNayar:
    case kLobeBurley:
    case kLobeSheen:
        if (lobe.kind == kLobeSheen && (lobe.flags & kFlagZeltner) != 0) {
            const float r = clamp(lobe.roughness, 0.01, 1.0);
            const float3 original = sampleCosineHemisphere(u.xy);
            const float aInv = zeltnerLtcAInv(wo.z, r);
            const float bInv = zeltnerLtcBInv(wo.z, r);
            const float3 w = normalize(float3(original.x / aInv - original.z * bInv / aInv, original.y / aInv,
                                              original.z));
            const float3 x = abs(wo.x) + abs(wo.y) > 1e-7 ? normalize(float3(wo.xy, 0.0)) : float3(1.0, 0.0, 0.0);
            const float3 y = cross(float3(0.0, 0.0, 1.0), x);
            wi = x * w.x + y * w.y + float3(0.0, 0.0, w.z);
        } else {
            wi = sampleCosineHemisphere(u.xy);
        }
        break;
    case kLobeTranslucent:
        wi = sampleCosineHemisphere(u.xy);
        wi.z = -wi.z;
        break;
    case kLobeConductor:
    case kLobeDielectric:
    case kLobeSchlick: {
        const bool isSmooth = smooth(lobe.alpha);
        const float3 h = isSmooth ? float3(0.0, 0.0, 1.0) : ggxSampleVisibleNormal(wo, lobe.alpha, u.xy);
        const float oh = dot(wo, h);
        bool reflect = true;
        float eta = 1.0;
        float chance = 1.0;
        if (lobe.kind != kLobeConductor) {
            chance = reflectChance(lobe, oh);
            eta = lobe.kind == kLobeDielectric ? relativeIor(lobe) : schlickIor(lobe);
            reflect = u.z < chance;
        }
        if (reflect) {
            wi = 2.0 * oh * h - wo;
            if (wi.z <= 0.0) {
                return s;
            }
        } else {
            const float cos2 = 1.0 - (1.0 - oh * oh) / (eta * eta);
            if (cos2 <= 0.0) {
                return s;   // total internal reflection, which the reflection branch took with chance 1
            }
            wi = (oh / eta - sqrt(cos2)) * h - wo / eta;
            if (wi.z >= 0.0) {
                return s;
            }
        }
        if (isSmooth) {
            const float3 tint = lobe.kind == kLobeDielectric ? max(lobe.colour0, 0.0) : float3(1.0);
            float3 F = float3(0.0);
            if (lobe.kind == kLobeConductor) {
                F = fresnelConductor(wo.z, lobe.colour0, lobe.colour1);
            } else if (lobe.kind == kLobeDielectric) {
                F = float3(fresnelDielectric(wo.z, eta));
            } else {
                F = schlickF(lobe, wo.z);
            }
            const float3 carried = reflect ? F : 1.0 - saturate(float3(dot(F, float3(1.0 / 3.0))));
            const float probability = reflect ? chance : 1.0 - chance;
            s.wi = frame.toWorld(wi);
            s.weight = tint * carried / max(probability, kEpsilon);
            s.pdf = probability;
            s.delta = true;
            s.valid = probability > 0.0;
            return s;
        }
        break;
    }
    default:
        return s;
    }
    s.wi = frame.toWorld(wi);
    s.pdf = pdfLocal(lobe, wo, wi);
    if (s.pdf <= 0.0) {
        return s;
    }
    s.weight = evalLocal(lobe, wo, wi, woWorld, s.wi) / s.pdf;
    s.valid = true;
    return s;
}

// ---- the stack ----------------------------------------------------------------

public static const uint kMaxLobes = 8;

public struct LobeStack {
    public Lobe   lobes[kMaxLobes];
    public uint   count;
    public float3 emission;
    public float  opacity;
};

public LobeStack emptyStack() {
    LobeStack stack;
    stack.count = 0;
    stack.emission = float3(0.0);
    stack.opacity = 1.0;
    for (uint k = 0; k < kMaxLobes; ++k) {
        stack.lobes[k].kind = kLobeNone;
    }
    return stack;
}

float selectionWeight(Lobe lobe, float3 wo) {
    return max(dot(lobe.weight * lobeAlbedo(lobe, wo), kLuminance), 0.0);
}

public float3 stackEval(LobeStack stack, float3 wo, float3 wi) {
    float3 sum = float3(0.0);
    for (uint k = 0; k < stack.count; ++k) {
        sum += stack.lobes[k].weight * lobeEval(stack.lobes[k], wo, wi);
    }
    return sum;
}

/// The stack's directional albedo at `wo`: what fraction of the light
/// arriving from there it returns, summed over the lobes with their weights.
/// The denoiser's albedo input, and what a layered material's throughput is
/// derived from.
public float3 stackAlbedo(LobeStack stack, float3 wo) {
    float3 sum = float3(0.0);
    for (uint k = 0; k < stack.count; ++k) {
        sum += stack.lobes[k].weight * lobeAlbedo(stack.lobes[k], wo);
    }
    return sum;
}

public float stackPdf(LobeStack stack, float3 wo, float3 wi) {
    float total = 0.0;
    float weighted = 0.0;
    for (uint k = 0; k < stack.count; ++k) {
        const float w = selectionWeight(stack.lobes[k], wo);
        total += w;
        weighted += w * lobePdf(stack.lobes[k], wo, wi);
    }
    return total > 0.0 ? weighted / total : 0.0;
}

public LobeSample stackSample(LobeStack stack, float3 wo, float3 u) {
    LobeSample none;
    none.wi = float3(0.0);
    none.weight = float3(0.0);
    none.pdf = 0.0;
    none.delta = false;
    none.valid = false;
    float weights[kMaxLobes];
    float total = 0.0;
    for (uint k = 0; k < stack.count; ++k) {
        weights[k] = selectionWeight(stack.lobes[k], wo);
        total += weights[k];
    }
    if (total <= 0.0) {
        return none;
    }
    // Choose a lobe with u.z, and reuse what is left of u.z inside the choice.
    float target = u.z * total;
    uint chosen = stack.count - 1;
    for (uint k = 0; k < stack.count; ++k) {
        if (target < weights[k] || k + 1 == stack.count) {
            chosen = k;
            break;
        }
        target -= weights[k];
    }
    const float probability = weights[chosen] / total;
    const float uz = saturate(target / max(weights[chosen], kEpsilon));
    LobeSample s = lobeSample(stack.lobes[chosen], wo, float3(u.xy, min(uz, 0.99999994)));
    if (!s.valid) {
        return none;
    }
    if (s.delta) {
        s.weight *= stack.lobes[chosen].weight / probability;
        s.pdf *= probability;
        return s;
    }
    s.pdf = stackPdf(stack, wo, s.wi);
    if (s.pdf <= 0.0) {
        return none;
    }
    s.weight = stackEval(stack, wo, s.wi) / s.pdf;
    return s;
}
