// Copyright (c) 2026 lucabRTrender contributors.
//
// WHAT A CLOUD CASTS ON THE SPACE AROUND IT, BAKED BY PART, READ BY PRODUCT.
//
// A skinned cloud cannot carry one baked visibility: the wing moves and takes
// its shadow with it. But a single part of a body changes shape very little
// between poses, so each part gets a field of its own, baked in the pose it
// was bound in: over a grid of probes, in every direction, how much of a ray
// leaving that probe this part's gaussians stop. At render a gaussian asks
// each part in that part's own current frame -- the position and the light's
// direction taken back through the part's skinning transform -- and multiplies
// what they answer. The wing's field turns with the wing; the body's stays
// with the body. (Lin et al. 2024 factor visibility this way with a network
// a part; DNF-Avatar bakes it this way as a scalar. This is the table, kept
// directional.)
//
// Directions live in an octahedral map, not in harmonics: a shadow has an
// edge, and harmonics of any published order do not (RGCA, PRTGaussian,
// GUS-IR all say so).
//
// The bake starts each ray one probe cell out from the probe, so the crowd a
// receiving gaussian is part of does not shadow it -- the same reason the
// rasteriser's shadow ray starts offset, but in units of the field, which is
// what does not depend on how fine the cloud was converted.
module splat_visibility;
import lrt.common.packing;
import lrt.common.octahedral;
import lrt.rt.rt_shadow;
import lrt.rt.rt_common;
import lrt.splat.splat_visibility_common;

// ----------------------------------------------------------------------------
// 1. Which part each gaussian is: its heaviest joint, through a table.

struct PartOfParams {
    uint count;       // gaussians
    uint perSplat;    // influences each carries
    uint pad0;
    uint pad1;
};

StructuredBuffer<float2>      influences;    // (joint, weight), perSplat a gaussian
StructuredBuffer<uint>        jointToPart;   // a part index a joint
RWStructuredBuffer<uint>      partOf;        // a part a gaussian
ConstantBuffer<PartOfParams>  partParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void visPartOf(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= partParams.count) {
        return;
    }
    float best = -1.0;
    uint joint = 0;
    for (uint k = 0; k < partParams.perSplat; ++k) {
        const float2 influence = influences[i * partParams.perSplat + k];
        if (influence.y > best) {
            best = influence.y;
            joint = uint(influence.x);
        }
    }
    partOf[i] = jointToPart[joint];
}

// ----------------------------------------------------------------------------
// 2. The bake: one thread a (probe, direction), through one part's proxies.

struct BakeParams {
    uint  part;         // which VisibilityPart
    uint  threads;      // grid^3 * octave^2
    float cut;          // stop once transmittance is under this
    float start;        // the ray's TMin, in the space of the cloud: one probe cell
    uint  pad0; uint pad1; uint pad2; uint pad3;
};

StructuredBuffer<VisibilityPart> parts;
StructuredBuffer<float4>         shadowFrames;
StructuredBuffer<float4>         shadowColours;
StructuredBuffer<float4>         shadowInstanceData;
StructuredBuffer<uint>           shadowInstanceIndices;
RWStructuredBuffer<uint>         texels;     // two f16 a word: [texel/2]
ConstantBuffer<BakeParams>       bakeParams;
uniform RaytracingAccelerationStructure scene;

/// Where texel `t` of a field lives, as a word and a half of it.
uint texelWord(uint t) { return t >> 1; }

void storeHalf(uint t, float v) {
    // Two texels share a word; only this thread's half is changed. The
    // dispatch is one thread a texel, so the two halves of a word are
    // written by two threads: an atomic or-after-clear keeps them apart.
    const uint bits = f32tof16(v) & 0xFFFFu;
    const uint shifted = (t & 1u) != 0u ? (bits << 16) : bits;
    InterlockedOr(texels[texelWord(t)], shifted);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void visBake(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= bakeParams.threads) {
        return;
    }
    const VisibilityPart part = parts[bakeParams.part];
    const uint dirs = part.octave * part.octave;
    const uint probe = t / dirs;
    const uint dir = t - probe * dirs;
    const uint g = part.grid;
    const uint x = probe % g;
    const uint y = (probe / g) % g;
    const uint z = probe / (g * g);
    const float3 unit = (float3(x, y, z) + 0.5) / float(g);
    const float3 origin = lerp(part.boxMin.xyz, part.boxMax.xyz, unit);
    const float2 f = (float2(dir % part.octave, dir / part.octave) + 0.5) / float(part.octave);
    const float3 direction = octDecode(f);

    ShadowState state = shadowBegin();
    RayDesc traced;
    traced.Origin = origin;
    traced.Direction = direction;
    traced.TMin = bakeParams.start;
    traced.TMax = 1.0e30;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, traced);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }
        const uint instance = query.CandidateInstanceIndex();
        const uint id = shadowInstanceIndices[instance * 2] + query.CandidatePrimitiveIndex() / kIcoTriangles;
        const float4 w0 = shadowInstanceData[instance * 3];
        const float4 w1 = shadowInstanceData[instance * 3 + 1];
        const float4 w2 = shadowInstanceData[instance * 3 + 2];
        const float3x3 toCloud = float3x3(w0.xyz, w1.xyz, w2.xyz);
        const float3 o = mul(toCloud, traced.Origin) + float3(w0.w, w1.w, w2.w);
        const float3 d = mul(toCloud, traced.Direction);
        const float opacity = shadowColours[id + shadowInstanceIndices[instance * 2 + 1]].w;
        float tPeak;
        const float alpha = shadowAlpha(shadowFrames[id * 4], shadowFrames[id * 4 + 1], shadowFrames[id * 4 + 2],
                                        opacity, o, d, traced.TMin, traced.TMax, 0.0, 1e30, tPeak);
        if (alpha > 0.0) {
            shadowTake(state, id, instance, alpha);
            if (shadowDone(state, bakeParams.cut)) {
                query.Abort();
                break;
            }
        }
    }
    storeHalf(part.offset + t, state.transmittance);
}

// ----------------------------------------------------------------------------
// 4. Gathering one part's gaussians into a cloud of their own, for the bake:
//    `keep[i]` is 1 where gaussian i is of the part, `dest[i]` its slot.

struct GatherParams {
    uint count;
    uint part;
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint>       gatherPartOf;
StructuredBuffer<uint>       dest;            // exclusive prefix sum of keep
RWStructuredBuffer<uint>     keep;
StructuredBuffer<float4>     restPositions;
StructuredBuffer<uint>       restShape;
RWStructuredBuffer<float4>   partPositions;
RWStructuredBuffer<uint>     partShape;
ConstantBuffer<GatherParams> gatherParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void visMark(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= gatherParams.count) {
        return;
    }
    keep[i] = gatherPartOf[i] == gatherParams.part ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void visGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= gatherParams.count || keep[i] == 0u) {
        return;
    }
    const uint at = dest[i];
    partPositions[at] = restPositions[i];
    [unroll] for (uint k = 0; k < 4; ++k) {
        partShape[at * 4 + k] = restShape[i * 4 + k];
    }
}

// ----------------------------------------------------------------------------
// 5. What a part's field holds, counted: texels under a half. Logged by the
//    bake so a field that stops nothing is seen as a number, not guessed at.

struct HistogramParams {
    uint offset;    // the part's first texel, or the first factor
    uint texels;
    float cut;
    uint floats;    // 1: count `fieldFloats` instead of the f16 words
};

StructuredBuffer<uint>          fieldWords;
StructuredBuffer<float>         fieldFloats;
RWStructuredBuffer<uint>        histogram;   // [under cut, total]
ConstantBuffer<HistogramParams> histogramParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void visHistogram(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= histogramParams.texels) {
        return;
    }
    const uint t = histogramParams.offset + k;
    float v;
    if (histogramParams.floats != 0u) {
        v = fieldFloats[t];
    } else {
        const uint word = fieldWords[t >> 1];
        v = f16tof32((t & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
    }
    if (v < histogramParams.cut) {
        InterlockedAdd(histogram[0], 1u);
    }
    InterlockedAdd(histogram[1], 1u);
}

// ----------------------------------------------------------------------------
// 6. The ambient term: one number a probe, the mean of its directions. What a
//    dome light is cut by -- a part's solid angle as seen from the probe --
//    since a dome comes from everywhere and has no direction to look up.

struct AmbientParams {
    uint part;
    uint probes;    // grid^3
    uint pad0;
    uint pad1;
};

RWStructuredBuffer<uint>       ambient;   // two f16 a word, grid^3 a part
ConstantBuffer<AmbientParams>  ambientParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void visAmbient(uint3 tid: SV_DispatchThreadID) {
    const uint probe = tid.x;
    if (probe >= ambientParams.probes) {
        return;
    }
    const VisibilityPart part = parts[ambientParams.part];
    const uint dirs = part.octave * part.octave;
    float sum = 0.0;
    for (uint d = 0; d < dirs; ++d) {
        const uint t = part.offset + probe * dirs + d;
        const uint word = texels[t >> 1];
        sum += f16tof32((t & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
    }
    const float mean = sum / float(max(dirs, 1u));
    const uint at = ambientParams.part * ambientParams.probes + probe;
    const uint bits = f32tof16(mean) & 0xFFFFu;
    InterlockedOr(ambient[at >> 1], (at & 1u) != 0u ? (bits << 16) : bits);
}
