// Copyright (c) 2026 lucabRTrender contributors.
//
// The ray tracer's ground truth, second half: per pixel, every particle
// evaluated at its peak along the pixel's ray, sorted exactly by that peak
// (then by index), blended front to back. What the ray tracer's segments and
// carry approximate, computed without them.
module reference_peak_blend;
import lrt.common.color;
import lrt.splat.frame;

static const uint kMaxHits = 1024;

StructuredBuffer<float4>    peakRow0;
StructuredBuffer<float4>    peakRow1;
StructuredBuffer<float4>    peakRow2;
StructuredBuffer<float4>    peakColour;
RWStructuredBuffer<float4>  colour;
RWStructuredBuffer<float>   depth;
RWStructuredBuffer<uint>    overflow;
ConstantBuffer<FrameParams> params;     // count: total particles

[shader("compute")]
[numthreads(8, 8, 1)]
void referencePeakBlend(uint3 tid: SV_DispatchThreadID) {
    const uint px = tid.x;
    const uint py = tid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    // View space: +z forward. A perspective ray leaves the eye with z = 1 per
    // unit of t; an orthographic one leaves the film plane along +z.
    const float2 film = (float2(float(px) + 0.5, float(py) + 0.5) - float2(params.centreX, params.centreY)) /
                        float2(params.focalX, params.focalY);
    const float3 origin = params.orthographic != 0 ? float3(film, 0.0) : float3(0.0);
    const float3 direction = params.orthographic != 0 ? float3(0.0, 0.0, 1.0) : float3(film, 1.0);

    float hitT[kMaxHits] = {};
    uint hitIndex[kMaxHits] = {};
    float hitAlpha[kMaxHits] = {};
    uint hits = 0;
    uint over = 0;
    for (uint s = 0; s < params.count; ++s) {
        const float4 c = peakColour[s];
        if (c.w < 0.0) {
            continue;
        }
        const float3x3 m = float3x3(peakRow0[s].xyz, peakRow1[s].xyz, peakRow2[s].xyz);
        const float3 centre = float3(peakRow0[s].w, peakRow1[s].w, peakRow2[s].w);
        const float3 o = mul(m, origin - centre);
        const float3 d = mul(m, direction);
        const float t = -dot(o, d) / max(dot(d, d), 1e-30);
        const float3 closest = o + t * d;
        const float power = dot(closest, closest);
        if (power > cutoffPower(c.w) || t < params.nearZ || t > params.farZ) {
            continue;
        }
        const float alpha = min(0.99, c.w * exp(-0.5 * power));
        if (alpha < 1.0 / 255.0) {
            continue;
        }
        if (hits == kMaxHits) {
            over = 1;
            continue;
        }
        uint at = hits;
        while (at > 0 && hitT[at - 1] > t) {
            hitT[at] = hitT[at - 1];
            hitIndex[at] = hitIndex[at - 1];
            hitAlpha[at] = hitAlpha[at - 1];
            at -= 1;
        }
        hitT[at] = t;
        hitIndex[at] = s;
        hitAlpha[at] = alpha;
        hits += 1;
    }

    float3 rgb = float3(0.0);
    float transmittance = 1.0;
    float depthSum = 0.0;
    float depthAt = -1.0;
    for (uint k = 0; k < hits && transmittance >= 1e-4; ++k) {
        const float weight = transmittance * hitAlpha[k];
        rgb += weight * peakColour[hitIndex[k]].xyz;
        depthSum += weight * hitT[k];
        transmittance *= 1.0 - hitAlpha[k];
        if (depthAt < 0.0 && 1.0 - transmittance >= params.depthThreshold) {
            depthAt = hitT[k];
        }
    }
    const float coverage = 1.0 - transmittance;
    float3 linearRgb = rgb;
    if (params.linearise != 0 && coverage > 1e-6) {
        linearRgb = srgbToLinear(rgb / coverage) * coverage;
    }
    const uint at = py * params.width + px;
    colour[at] = float4(linearRgb + transmittance * float3(params.bgR, params.bgG, params.bgB),
                        coverage + transmittance * params.bgA);
    depth[at] = params.depthMode == 0 ? (coverage > 1e-6 ? depthSum / coverage : 0.0) : max(depthAt, 0.0);
    overflow[at] = over;
}
