// Copyright (c) 2026 lucabRTrender contributors.
//
// The ground truth, per pixel: every splat is considered, the ones that reach
// this pixel are sorted exactly (depth, then index -- the rasteriser's own tie
// rule), and blended front to back. No tiles, no ranges, no pairs. Slow, and
// meant for scenes of thousands of splats in tests.
//
// `overflow` counts pixels with more than kMaxHits contributors, which a test
// must see as zero for the comparison to mean anything.
module reference_blend;
import lrt.common.color;
import lrt.splat.frame;

static const uint kMaxHits = 1024;

StructuredBuffer<float4>    refScreen;
StructuredBuffer<float4>    refConic;
StructuredBuffer<float4>    refColour;
RWStructuredBuffer<float4>  colour;
RWStructuredBuffer<float>   depth;
RWStructuredBuffer<uint>    overflow;   // per pixel: 1 when kMaxHits was exceeded
ConstantBuffer<FrameParams> params;     // count: total splats

[shader("compute")]
[numthreads(8, 8, 1)]
void referenceBlend(uint3 tid: SV_DispatchThreadID) {
    const uint px = tid.x;
    const uint py = tid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    const float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);
    float hitDepth[kMaxHits] = {};
    uint hitIndex[kMaxHits] = {};
    float hitAlpha[kMaxHits] = {};
    uint hitDisc[kMaxHits] = {};
    uint hits = 0;
    uint over = 0;
    for (uint s = 0; s < params.count; ++s) {
        const float4 screen = refScreen[s];
        if (screen.z < 0.0) {
            continue;
        }
        const float4 conic = refConic[s];
        const float2 d = pixel - screen.xy;
        const float power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
        if (power > screen.w) {
            continue;
        }
        const bool disc = refColour[s].w > 0.5;
        const float alpha = disc ? 1.0 : min(0.99, conic.w * exp(-0.5 * power));
        if (alpha < 1.0 / 255.0) {
            continue;
        }
        if (hits == kMaxHits) {
            over = 1;
            continue;
        }
        // Insertion sort by (depth, index). Splats arrive in index order, so
        // an equal depth goes after what is already there.
        uint at = hits;
        while (at > 0 && hitDepth[at - 1] > screen.z) {
            hitDepth[at] = hitDepth[at - 1];
            hitIndex[at] = hitIndex[at - 1];
            hitAlpha[at] = hitAlpha[at - 1];
            hitDisc[at] = hitDisc[at - 1];
            at -= 1;
        }
        hitDepth[at] = screen.z;
        hitIndex[at] = s;
        hitAlpha[at] = alpha;
        hitDisc[at] = disc ? 1 : 0;
        hits += 1;
    }

    float3 rgb = float3(0.0);
    float transmittance = 1.0;
    float depthSum = 0.0;
    float depthAt = -1.0;
    bool discHit = false;
    float3 discColour = float3(0.0);
    float discZ = 0.0;
    for (uint k = 0; k < hits; ++k) {
        if (hitDisc[k] != 0) {
            discHit = true;
            discColour = refColour[hitIndex[k]].xyz;
            discZ = hitDepth[k];
            break;
        }
        const float weight = transmittance * hitAlpha[k];
        rgb += weight * refColour[hitIndex[k]].xyz;
        depthSum += weight * hitDepth[k];
        transmittance *= 1.0 - hitAlpha[k];
        if (depthAt < 0.0 && 1.0 - transmittance >= params.depthThreshold) {
            depthAt = hitDepth[k];
        }
        if (transmittance < 1e-4) {
            break;
        }
    }
    const float coverage = 1.0 - transmittance;
    float3 linearRgb = rgb;
    if (params.linearise != 0 && coverage > 1e-6) {
        linearRgb = srgbToLinear(rgb / coverage) * coverage;
    }
    const uint at = py * params.width + px;
    float4 behind = float4(params.bgR, params.bgG, params.bgB, params.bgA);
    if (discHit) {
        behind = float4(discColour, 1.0);
        depthSum += transmittance * discZ;
        if (depthAt < 0.0) {
            depthAt = discZ;
        }
    }
    const float total = discHit ? 1.0 : coverage;
    colour[at] = float4(linearRgb + transmittance * behind.rgb, coverage + transmittance * behind.w);
    depth[at] = params.depthMode == 0 ? (total > 1e-6 ? depthSum / total : 0.0) : max(depthAt, 0.0);
    overflow[at] = over;
}
