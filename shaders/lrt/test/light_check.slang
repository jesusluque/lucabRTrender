// Copyright (c) 2026 lucabRTrender contributors.
//
// Each light against itself on the device: directions drawn by sampleLight
// from one shading point, binned over the sphere, against lightPdf integrated
// over each bin. A light whose samples do not follow the density it reports
// would light a path tracer wrongly however right its radiance looked.
module light_check;
import lrt.light.lights_image;

struct LightCheckParams {
    uint   thetaBins;
    uint   phiBins;
    uint   samples;
    uint   pad0;
    float4 point;    // xyz: where the surface is
    float4 normal;   // xyz: the way it faces
};

StructuredBuffer<LightRecord>    lights;
RWStructuredBuffer<uint>         bins;       // per sample: its bin, or thetaBins * phiBins where none
RWStructuredBuffer<float>        observed;   // per bin
RWStructuredBuffer<float>        expected;   // per bin
RWStructuredBuffer<float4>       result;
RWStructuredBuffer<uint>         mismatches;   // [samples whose reported pdf differs from lightPdf, whose direction lightHit does not find as sampled]
RWStructuredBuffer<uint>         roundTrip;    // [uv that do not survive a turn through a direction]
RWStructuredBuffer<float>        pyramid;      // [worst relative gap between a parent and its four children]
StructuredBuffer<float4>         image;        // a rendered frame, for the link counters
StructuredBuffer<float>          imageDepth;
RWStructuredBuffer<uint>         halves;       // [drawn left, lit left, drawn right, lit right]
RWStructuredBuffer<float>        worst;        // [the worst relative difference]
ConstantBuffer<LightCheckParams> params;

static const float kPiCheck = 3.14159265358979;

/// Two numbers for sample i, spread rather than stratified: the statistic
/// asks for independence, not for low discrepancy.
float2 sampleAt(uint i) {
    uint h = i * 2654435761u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const uint g = h * 1664525u + 1013904223u;
    return float2(float(h >> 8) * (1.0 / 16777216.0), float(g >> 8) * (1.0 / 16777216.0));
}

/// A cone light's axis and the cosine of its half angle, or false where the
/// light is not a cone.
bool coneOf(LightRecord l, out float3 axis, out float cosMax) {
    axis = float3(0.0, 0.0, 1.0);
    cosMax = -1.0;
    if (l.kind == kLightDistant) {
        axis = lightAxisZ(l);
        cosMax = cos(0.5 * max(l.sizeX, 0.0));
        return cosMax < 1.0;
    }
    if (l.kind == kLightSphere) {
        const float3 offset = lightOrigin(l) - params.point.xyz;
        const float distance2 = dot(offset, offset);
        const float radius = max(l.sizeX, 0.0);
        if (radius <= 0.0 || distance2 <= radius * radius) {
            return false;
        }
        axis = normalize(offset);
        cosMax = sqrt(max(0.0, 1.0 - radius * radius / distance2));
        return true;
    }
    return false;
}

/// Bins over a cone's own solid angle: cosine from its axis, then around it.
/// A grid in world axes cuts a small cone badly, and then the statistic
/// measures the grid rather than the light.
uint coneBinOf(float3 axis, float cosMax, float3 w) {
    const float cosTheta = dot(axis, w);
    if (cosTheta < cosMax) {
        return params.thetaBins * params.phiBins;   // outside the cone
    }
    float3 x, y;
    lightFrame(axis, x, y);
    float phi = atan2(dot(y, w), dot(x, w));
    if (phi < 0.0) {
        phi += 2.0 * kPiCheck;
    }
    const float span = max(1.0 - cosMax, 1.0e-9);
    const uint t = min(uint((1.0 - cosTheta) / span * float(params.thetaBins)), params.thetaBins - 1);
    const uint p = min(uint(phi / (2.0 * kPiCheck) * float(params.phiBins)), params.phiBins - 1);
    return t * params.phiBins + p;
}

/// Bins over a cylinder's lateral surface: the angle about its axis, then
/// the height along it, at the point where the direction meets it. A cell
/// on the far side never receives a sample; its expectation says so.
uint cylinderBinOf(LightRecord l, float3 w) {
    float t;
    float3 normal;
    float along;
    if (!cylinderHit(l, params.point.xyz, w, t, normal, along)) {
        return params.thetaBins * params.phiBins;
    }
    float phi = atan2(dot(normal, lightAxisZ(l)), dot(normal, lightAxisY(l)));
    if (phi < 0.0) {
        phi += 2.0 * kPiCheck;
    }
    const uint u = min(uint(phi / (2.0 * kPiCheck) * float(params.thetaBins)), params.thetaBins - 1);
    const uint v = min(uint((along / l.sizeY + 0.5) * float(params.phiBins)), params.phiBins - 1);
    return u * params.phiBins + v;
}

/// Whether the light is an area one, and its half extents in its own axes.
bool areaOf(LightRecord l, out float2 half) {
    half = float2(0.0, 0.0);
    if (l.kind == kLightDisk) {
        half = float2(l.sizeX, l.sizeX);
        return l.sizeX > 0.0;
    }
    if (l.kind == kLightRect) {
        half = float2(0.5 * l.sizeX, 0.5 * l.sizeY);
        return l.sizeX > 0.0 && l.sizeY > 0.0;
    }
    return false;
}

/// A point in the light's own axes, or false where the ray misses its plane.
bool areaHit(LightRecord l, float3 w, out float2 at) {
    at = float2(0.0, 0.0);
    const float3 normal = -lightAxisZ(l);
    const float denominator = dot(w, -normal);
    if (denominator <= 0.0) {
        return false;
    }
    const float t = dot(lightOrigin(l) - params.point.xyz, -normal) / denominator;
    if (!(t > 0.0)) {
        return false;
    }
    const float3 offset = params.point.xyz + w * t - lightOrigin(l);
    at = float2(dot(offset, lightAxisX(l)), dot(offset, lightAxisY(l)));
    return true;
}

/// Bins over the light's own surface, where a sampler uniform in area must
/// fill the histogram uniformly. A grid in world angles cuts a straight edge
/// diagonally, and then the statistic measures the grid.
uint areaBinOf(LightRecord l, float2 half, float3 w) {
    float2 at;
    if (!areaHit(l, w, at)) {
        return params.thetaBins * params.phiBins;
    }
    if (l.kind == kLightDisk ? dot(at, at) > half.x * half.x
                             : (abs(at.x) > half.x || abs(at.y) > half.y)) {
        return params.thetaBins * params.phiBins;
    }
    const uint u = min(uint((at.x + half.x) / (2.0 * half.x) * float(params.thetaBins)), params.thetaBins - 1);
    const uint v = min(uint((at.y + half.y) / (2.0 * half.y) * float(params.phiBins)), params.phiBins - 1);
    return u * params.phiBins + v;
}

/// A dome with an image is binned in the image, where the warp does its
/// work: a sampler following luminance must fill a bin in proportion to the
/// luminance in it. Angles in world axes cannot resolve a sun a few texels
/// wide, which is the artefact that made a cone's statistic meaningless.
uint domeBinOf(LightRecord l, float3 w) {
    const float2 uv = domeUv(l, w);
    const uint u = min(uint(uv.x * float(params.phiBins)), params.phiBins - 1);
    const uint v = min(uint(uv.y * float(params.thetaBins)), params.thetaBins - 1);
    return v * params.phiBins + u;
}

uint binOf(float3 w) {
    const float theta = acos(clamp(w.z, -1.0, 1.0));
    float phi = atan2(w.y, w.x);
    if (phi < 0.0) {
        phi += 2.0 * kPiCheck;
    }
    const uint t = min(uint(theta / kPiCheck * float(params.thetaBins)), params.thetaBins - 1);
    const uint p = min(uint(phi / (2.0 * kPiCheck) * float(params.phiBins)), params.phiBins - 1);
    return t * params.phiBins + p;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lightDraw(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.samples) {
        return;
    }
    const LightSample s = sampleLightImaged(lights[0], params.point.xyz, params.normal.xyz, sampleAt(i));
    float3 axis;
    float cosMax;
    float2 half;
    const bool cone = coneOf(lights[0], axis, cosMax);
    const bool area = areaOf(lights[0], half);
    const bool warped = lights[0].kind == kLightDome && domeHasImage(lights[0]);
    const bool cylinder = lights[0].kind == kLightCylinder;
    uint bin = params.thetaBins * params.phiBins;
    if (s.valid && !s.delta) {
        bin = warped   ? domeBinOf(lights[0], s.wi)
              : cone   ? coneBinOf(axis, cosMax, s.wi)
              : area   ? areaBinOf(lights[0], half, s.wi)
              : cylinder ? cylinderBinOf(lights[0], s.wi)
                       : binOf(s.wi);
    }
    bins[i] = bin;
}

/// What a link let through, counted by halves: two prims of different
/// categories, side by side, and a light that reaches one of them. Drawn says
/// the geometry is there; lit says the light arrived.
[shader("compute")]
[numthreads(1, 1, 1)]
void lightLinkCounters(uint3 tid: SV_DispatchThreadID) {
    uint drawnLeft = 0, litLeft = 0, drawnRight = 0, litRight = 0;
    for (uint y = 0; y < params.thetaBins; ++y) {
        for (uint x = 0; x < params.phiBins; ++x) {
            const uint i = y * params.phiBins + x;
            if (!(imageDepth[i] > 0.0)) {
                continue;
            }
            const float3 c = image[i].rgb;
            const bool lit = max(max(c.r, c.g), c.b) > 1.0e-4;
            if (x < params.phiBins / 2) {
                ++drawnLeft;
                litLeft += lit ? 1u : 0u;
            } else {
                ++drawnRight;
                litRight += lit ? 1u : 0u;
            }
        }
    }
    halves[0] = drawnLeft;
    halves[1] = litLeft;
    halves[2] = drawnRight;
    halves[3] = litRight;
}

/// Does the chain telescope? A descent that splits by luminance is only
/// unbiased if a parent is the sum of its four children. These mips are
/// filtered in uv and decoded as sRGB, neither of which preserves a sum, so
/// this measures the gap rather than assuming it away.
[shader("compute")]
[numthreads(1, 1, 1)]
void lightPyramid(uint3 tid: SV_DispatchThreadID) {
    const LightRecord l = lights[0];
    const TextureRecord record = gTextures.records[l.texture];
    const float levels = ceil(log2(max(max(record.width, record.height), 1.0)));
    float worstGap = 0.0;
    for (float level = 1.0; level <= levels; level += 1.0) {
        const uint mip = uint(level);
        const int2 size = int2(max(int(record.width) >> mip, 1), max(int(record.height) >> mip, 1));
        for (int y = 0; y < size.y; ++y) {
            for (int x = 0; x < size.x; ++x) {
                const float parent = domeTexelOf(l, int2(x, y), mip) * 4.0;
                const float children = domeTexelOf(l, int2(x * 2, y * 2), mip - 1) +
                                       domeTexelOf(l, int2(x * 2 + 1, y * 2), mip - 1) +
                                       domeTexelOf(l, int2(x * 2, y * 2 + 1), mip - 1) +
                                       domeTexelOf(l, int2(x * 2 + 1, y * 2 + 1), mip - 1);
                const float scale = max(max(parent, children), 1.0e-6);
                worstGap = max(worstGap, abs(parent - children) / scale);
            }
        }
    }
    pyramid[0] = worstGap;
}

/// A direction is where the image says it is: uv into a direction and back
/// again must land where it started. A statistic would fold a mapping error
/// into its noise; this cannot.
[shader("compute")]
[numthreads(1, 1, 1)]
void lightRoundTrip(uint3 tid: SV_DispatchThreadID) {
    uint differ = 0;
    const uint steps = 64;
    for (uint j = 1; j < steps; ++j) {
        for (uint i = 0; i < steps; ++i) {
            const float2 uv = float2((float(i) + 0.5) / float(steps), (float(j) + 0.5) / float(steps));
            const float2 back = domeUv(lights[0], domeDirection(lights[0], uv));
            // u wraps, so a difference of one whole turn is no difference.
            float du = abs(back.x - uv.x);
            du = min(du, 1.0 - du);
            differ += (du > 1.0e-4 || abs(back.y - uv.y) > 1.0e-4) ? 1u : 0u;
        }
    }
    roundTrip[0] = differ;
}

/// The density a sample was drawn with, against the density the same
/// direction has: a disagreement here is a sampler and its pdf parting ways,
/// which no amount of binning would show and which MIS would weigh wrongly.
/// One thread a sample, each writing what it found: a million samples of an
/// imaged dome looped in one invocation ran past the Vulkan driver's patience
/// on an NVIDIA L4 (VK_ERROR_DEVICE_LOST), while Metal took it.
RWStructuredBuffer<uint>  sampleFlags;   // per sample: bit 0 pdf differs, bit 1 lightHit does not find it
RWStructuredBuffer<float> sampleGaps;    // per sample: its pdf's relative difference

[shader("compute")]
[numthreads(256, 1, 1)]
void lightConsistent(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.samples) {
        return;
    }
    sampleFlags[i] = 0;
    sampleGaps[i] = 0.0;
    const LightSample s = sampleLightImaged(lights[0], params.point.xyz, params.normal.xyz, sampleAt(i));
    if (!s.valid || s.delta) {
        return;
    }
    // An area light's pdf divides by the emitter's cosine, so beside its
    // silhouette the comparison measures conditioning, not agreement.
    // For a curved emitter the hit is found through sqrt(h^2 / a) with
    // h^2 = r^2 - |c|^2 vanishing at a tangent ray: h^2 carries an
    // absolute error of about 1e-8, the root turns that into 5e-6 in t,
    // the normal into 1.7e-5, and the pdf into 1.7e-5 / cos -- past 1e-3
    // until cos reaches 0.017. The light arriving from that band is of
    // order cos^2 / 2 of the whole, 0.02%. Measured on a cylinder before
    // the guard: 7826 of a million samples with the b^2 - 4ac form, 2378
    // with the closest-approach one, 2026 above a guard of 1e-3.
    if (s.emitterCosine < 2.0e-2) {
        return;
    }
    const float density = lightPdfImaged(lights[0], params.point.xyz, params.normal.xyz, s.wi);
    const float relative = abs(density - s.pdf) / max(s.pdf, 1.0e-9);
    sampleGaps[i] = relative;
    uint flags = relative > 1.0e-3 ? 1u : 0u;
    // And the direction found again by the ray: the light met, at the
    // distance the sample was drawn at, with the radiance it reported.
    const LightHit h = lightHitImaged(lights[0], params.point.xyz, s.wi);
    bool agrees = h.valid;
    if (agrees) {
        const float3 gap = abs(h.radiance - s.radiance) / max(abs(s.radiance), float3(1.0e-9));
        agrees = all(gap <= float3(1.0e-3));
        if (s.distance < 1.0e29) {
            agrees = agrees && abs(h.t - s.distance) <= 1.0e-3 * max(s.distance, 1.0e-3);
        } else {
            agrees = agrees && h.t >= 1.0e29;
        }
    }
    flags |= agrees ? 0u : 2u;
    sampleFlags[i] = flags;
}

/// The samples' findings summed: a loop of reads and additions.
[shader("compute")]
[numthreads(1, 1, 1)]
void lightConsistentReduce(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint differ = 0;
    uint hitsDiffer = 0;
    float worstRelative = 0.0;
    for (uint i = 0; i < params.samples; ++i) {
        const uint flags = sampleFlags[i];
        differ += flags & 1u;
        hitsDiffer += (flags >> 1) & 1u;
        worstRelative = max(worstRelative, sampleGaps[i]);
    }
    mismatches[0] = differ;
    mismatches[1] = hitsDiffer;
    worst[0] = worstRelative;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lightCount(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.thetaBins * params.phiBins;
    for (uint b = 0; b <= n; ++b) {
        observed[b] = 0.0;
    }
    for (uint i = 0; i < params.samples; ++i) {
        observed[bins[i]] += 1.0;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lightExpect(uint3 tid: SV_DispatchThreadID) {
    const LightRecord light = lights[0];
    const uint sub = 8;
    float3 axis;
    float cosMax;
    const bool cone = coneOf(light, axis, cosMax);
    float2 half;
    const bool area = areaOf(light, half);
    const bool warped = light.kind == kLightDome && domeHasImage(light);
    const bool cylinder = light.kind == kLightCylinder;
    float3 coneX, coneY;
    lightFrame(axis, coneX, coneY);
    // Over the same parametrisation the samples were binned in, so the
    // measure is dcos dphi for a cone and sin(theta) dtheta dphi otherwise.
    const float dCos = cone ? (1.0 - cosMax) / float(params.thetaBins * sub) : 0.0;
    const float dTheta = kPiCheck / float(params.thetaBins * sub);
    const float dPhi = 2.0 * kPiCheck / float(params.phiBins * sub);
    float integral = 0.0;
    for (uint t = 0; t < params.thetaBins; ++t) {
        for (uint p = 0; p < params.phiBins; ++p) {
            float sum = 0.0;
            if (warped) {
                // In the image's own measure: du dv, with the solid angle the
                // density is written against folded back in.
                const float du = 1.0 / float(params.phiBins * sub);
                const float dv = 1.0 / float(params.thetaBins * sub);
                for (uint j = 0; j < sub; ++j) {
                    const float v = (float(t * sub + j) + 0.5) * dv;
                    const float sinTheta = sin(v * kPiCheck);
                    for (uint k = 0; k < sub; ++k) {
                        const float u = (float(p * sub + k) + 0.5) * du;
                        const float3 w = domeDirection(light, float2(u, v));
                        sum += lightPdfImaged(light, params.point.xyz, params.normal.xyz, w) * kPiCheck * 2.0 * kPiCheck *
                               sinTheta * du * dv;
                    }
                }
                expected[t * params.phiBins + p] = sum * float(params.samples);
                integral += sum;
                continue;
            }
            if (cylinder) {
                // Uniform on the lateral surface, but only where it faces
                // the point: a cell whose outward normal looks away is never
                // drawn, and expects nothing.
                const float3 axisX = lightAxisX(light);
                const float3 ay = lightAxisY(light);
                const float3 az = lightAxisZ(light);
                const float cellPhi = 2.0 * kPiCheck / float(params.thetaBins * sub);
                const float cellX = light.sizeY / float(params.phiBins * sub);
                const float total = float(params.thetaBins * params.phiBins * sub * sub);
                for (uint j = 0; j < sub; ++j) {
                    const float phi = (float(t * sub + j) + 0.5) * cellPhi;
                    const float3 normal = ay * cos(phi) + az * sin(phi);
                    for (uint k = 0; k < sub; ++k) {
                        const float along = -0.5 * light.sizeY + (float(p * sub + k) + 0.5) * cellX;
                        const float3 at = lightOrigin(light) + axisX * along + normal * light.sizeX;
                        const bool facing = dot(normal, params.point.xyz - at) > 0.0;
                        sum += facing ? 1.0 / total : 0.0;
                    }
                }
                expected[t * params.phiBins + p] = sum * float(params.samples);
                integral += sum;
                continue;
            }
            if (area) {
                // Uniform in area: what a bin should hold is the part of it
                // that lies on the light, over the light's whole area.
                const float cellX = 2.0 * half.x / float(params.thetaBins * sub);
                const float cellY = 2.0 * half.y / float(params.phiBins * sub);
                const float total = light.kind == kLightDisk ? 3.14159265358979 * half.x * half.x
                                                             : 4.0 * half.x * half.y;
                for (uint j = 0; j < sub; ++j) {
                    const float x = -half.x + (float(t * sub + j) + 0.5) * cellX;
                    for (uint k = 0; k < sub; ++k) {
                        const float y = -half.y + (float(p * sub + k) + 0.5) * cellY;
                        const bool inside = light.kind == kLightDisk ? (x * x + y * y <= half.x * half.x) : true;
                        sum += inside ? cellX * cellY / total : 0.0;
                    }
                }
                expected[t * params.phiBins + p] = sum * float(params.samples);
                integral += sum;
                continue;
            }
            for (uint j = 0; j < sub; ++j) {
                const float theta = (float(t * sub + j) + 0.5) * dTheta;
                const float sinTheta = sin(theta);
                const float cosTheta = 1.0 - (float(t * sub + j) + 0.5) * dCos;
                const float coneSin = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
                for (uint k = 0; k < sub; ++k) {
                    const float phi = (float(p * sub + k) + 0.5) * dPhi;
                    if (cone) {
                        const float3 w = normalize(axis * cosTheta + coneX * (coneSin * cos(phi)) +
                                                   coneY * (coneSin * sin(phi)));
                        sum += lightPdfImaged(light, params.point.xyz, params.normal.xyz, w) * dCos * dPhi;
                    } else {
                        const float3 w = float3(sinTheta * cos(phi), sinTheta * sin(phi), cos(theta));
                        sum += lightPdfImaged(light, params.point.xyz, params.normal.xyz, w) * sinTheta * dTheta * dPhi;
                    }
                }
            }
            expected[t * params.phiBins + p] = sum * float(params.samples);
            integral += sum;
        }
    }
    result[0] = float4(integral, 0.0, 0.0, 0.0);
}

/// Pearson's statistic over bins expecting at least 5, the rest pooled.
[shader("compute")]
[numthreads(1, 1, 1)]
void lightStatistic(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.thetaBins * params.phiBins;
    float statistic = 0.0;
    float dof = 0.0;
    float pooledObserved = 0.0;
    float pooledExpected = 0.0;
    float drawn = 0.0;
    for (uint b = 0; b < n; ++b) {
        drawn += observed[b];
        if (expected[b] < 5.0) {
            pooledObserved += observed[b];
            pooledExpected += expected[b];
            continue;
        }
        statistic += (observed[b] - expected[b]) * (observed[b] - expected[b]) / expected[b];
        dof += 1.0;
    }
    if (pooledExpected >= 5.0) {
        statistic += (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected) / pooledExpected;
        dof += 1.0;
    }
    result[1] = float4(statistic, max(dof - 1.0, 1.0), drawn / float(params.samples), 0.0);
}

/// The irradiance factor at params.point from the sampler alone: the mean
/// over params.samples of max(n . wi, 0) / pdf (unit radiance), invalid
/// samples counting as zero -- exactly the estimator shading uses, with
/// nothing of shading around it. Beside it, the same integral by quadrature
/// over a cylinder's lateral surface, so a sampler can be told from a shader.
RWStructuredBuffer<float> irradiance;   // [by the sampler, by quadrature]

[shader("compute")]
[numthreads(1, 1, 1)]
void lightIrradianceAt(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const LightRecord l = lights[0];
    const float3 p = params.point.xyz;
    const float3 n = params.normal.xyz;
    float sum = 0.0;
    for (uint i = 0; i < params.samples; ++i) {
        const LightSample s = sampleLight(l, p, n, sampleAt(i));
        if (!s.valid || s.pdf <= 0.0) {
            continue;
        }
        sum += max(dot(n, s.wi), 0.0) / s.pdf;
    }
    irradiance[0] = sum / float(params.samples);
    // Quadrature, cylinder only.
    float quad = 0.0;
    if (l.kind == kLightCylinder) {
        const float3 axis = lightAxisX(l);
        const float3 ay = lightAxisY(l);
        const float3 az = lightAxisZ(l);
        const uint cells = 512;
        const float dPhi = 2.0 * kPiCheck / float(cells);
        const float dX = l.sizeY / float(cells);
        const float dA = l.sizeX * dPhi * dX;
        for (uint i = 0; i < cells; ++i) {
            const float phi = (float(i) + 0.5) * dPhi;
            const float3 normal = ay * cos(phi) + az * sin(phi);
            for (uint j = 0; j < cells; ++j) {
                const float x = -0.5 * l.sizeY + (float(j) + 0.5) * dX;
                const float3 q = lightOrigin(l) + axis * x + normal * l.sizeX;
                const float3 d = q - p;
                const float d2 = dot(d, d);
                const float3 w = d / sqrt(d2);
                const float cosP = dot(n, w);
                const float cosQ = dot(normal, -w);
                if (cosP > 0.0 && cosQ > 0.0) {
                    quad += cosP * cosQ / d2 * dA;
                }
            }
        }
    }
    irradiance[1] = quad;
}

/// The cumulative shares light_prefix wrote, against the power written a
/// second time here from the record alone: monotone, each step the light's
/// own power, the last the total. Counts the records that miss.
RWStructuredBuffer<uint> prefixMisses;   // [not monotone or wrong step, over lightCount records]
uniform uint             prefixCount;

[shader("compute")]
[numthreads(1, 1, 1)]
void lightPrefixCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint misses = 0;
    float running = 0.0;
    for (uint k = 0; k < prefixCount; ++k) {
        const LightRecord l = lights[k];
        float power = max(0.2126 * l.colour.r + 0.7152 * l.colour.g + 0.0722 * l.colour.b, 0.0) * exp2(l.exposure);
        if ((l.flags & kLightNormalize) == 0) {
            if (l.kind == kLightSphere) power *= 4.0 * kPiCheck * l.sizeX * l.sizeX;
            else if (l.kind == kLightDisk) power *= kPiCheck * l.sizeX * l.sizeX;
            else if (l.kind == kLightRect) power *= l.sizeX * l.sizeY;
            else if (l.kind == kLightCylinder) power *= 2.0 * kPiCheck * l.sizeX * l.sizeY;
        }
        power = max(power, 1.0e-6);
        const float before = running;
        running += power;
        const bool monotone = l.cumulative > before;
        const bool step = abs(l.cumulative - running) <= 1.0e-5 * max(running, 1.0e-6);
        misses += (monotone && step) ? 0u : 1u;
    }
    prefixMisses[0] = misses;
}


// Light instancing: a light's copies, placed on the device from an
// instancer's rows, against the same lights authored one by one. Every
// record must match its authored twin -- rows, size, power share -- and the
// tables must be the same length.
StructuredBuffer<LightRecord> authored;
RWStructuredBuffer<uint>      instanceMisses;   // [records differing, records compared]
uniform uint                  authoredCount;

[shader("compute")]
[numthreads(1, 1, 1)]
void lightInstanceCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint misses = prefixCount == authoredCount ? 0u : 1u;
    const uint n = min(prefixCount, authoredCount);
    for (uint k = 0; k < n; ++k) {
        const LightRecord a = lights[k];
        const LightRecord b = authored[k];
        const float rows = max(max(maxElement(abs(a.row0 - b.row0)), maxElement(abs(a.row1 - b.row1))),
                               maxElement(abs(a.row2 - b.row2)));
        const bool same = rows <= 1.0e-5 && a.kind == b.kind && a.sizeX == b.sizeX && a.sizeY == b.sizeY &&
                          abs(a.cumulative - b.cumulative) <= 1.0e-5 * max(b.cumulative, 1.0e-6);
        misses += same ? 0u : 1u;
    }
    instanceMisses[0] = misses;
    instanceMisses[1] = n;
}

float maxElement(float4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}
