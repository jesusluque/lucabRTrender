// Copyright (c) 2026 lucabRTrender contributors.
//
// The medium against what it must hold to. Through a box of constant
// density the transmittance is Beer-Lambert's, exp(-sigma L): ratio
// tracking estimates it and delta tracking's scatter-or-not estimates
// 1 - T; over many rays each mean must sit within the standard error the
// kernel computes from its own spread -- the threshold is derived, not
// chosen. No density may exceed the majorant over it, at any point. And
// an HDDA walk at the leaf's size visits exactly the leaves a
// voxel-by-voxel walk passes through.
import lrt.volume.medium;

struct MediumParams {
    uint   rays;
    uint   volume;
    float  expectedT;    // exp(-sigma L) for the ray below
    float  pad0;
    float3 origin;       // world
    float  tMin;
    float3 direction;    // unit
    float  tMax;
    float3 boxMin;       // world, for the majorant probe
    float  pad1;
    float3 boxMax;
    float  pad2;
};

ConstantBuffer<MediumParams> medium;
StructuredBuffer<uint>       words;
RWStructuredBuffer<uint>     counts;   // [scattered, majorant violations, leaf steps, leaves with a leaf, distinct leaves by voxel walk, probes]
RWStructuredBuffer<float>    sums;     // [T sum, T^2 sum, scatter sum, scatter^2 sum] by float atomics? no: one thread per ray writes, a second pass reduces

RWStructuredBuffer<float>    perRay;   // [rays] transmittance, [rays] scattered (0/1)

[shader("compute")]
[numthreads(64, 1, 1)]
void mediumRays(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= medium.rays) return;
    const VolumeRecord r = volumeRecord(words, medium.volume);
    uint rngA = 0x9E3779B9u ^ (i * 2654435761u);
    uint rngB = 0x7F4A7C15u ^ (i * 40503u + 1u);
    perRay[i] = mediumTransmittance(words, r, medium.origin, medium.direction, medium.tMin, medium.tMax, rngA);
    float t;
    const bool scattered = mediumScatter(words, r, medium.origin, medium.direction, medium.tMin, medium.tMax, rngB, t);
    perRay[medium.rays + i] = scattered ? 1.0 : 0.0;
    if (scattered) InterlockedAdd(counts[0], 1u);
}

/// One thread reduces the rays' estimates: sums and sums of squares.
[shader("compute")]
[numthreads(1, 1, 1)]
void mediumReduce(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    float t = 0.0, t2 = 0.0, s = 0.0, s2 = 0.0;
    for (uint i = 0; i < medium.rays; ++i) {
        const float a = perRay[i];
        const float b = perRay[medium.rays + i];
        t += a; t2 += a * a; s += b; s2 += b * b;
    }
    sums[0] = t; sums[1] = t2; sums[2] = s; sums[3] = s2;
}

uint probeHash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

/// Random world points in and about the box: the density's extinction
/// never exceeds the leaf majorant over it.
[shader("compute")]
[numthreads(64, 1, 1)]
void mediumMajorant(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= medium.rays) return;
    const VolumeRecord r = volumeRecord(words, medium.volume);
    const float3 u = float3(float(probeHash(i * 3u + 1u) >> 8), float(probeHash(i * 3u + 2u) >> 8),
                            float(probeHash(i * 3u + 3u) >> 8)) / 16777216.0;
    const float3 p = medium.boxMin + (medium.boxMax - medium.boxMin) * (u * 1.2 - 0.1);   // a tenth beyond each side
    const float sigma = mediumDensityAt(words, r, p) * r.sigmaScale;
    const float majorant = mediumMajorantAt(words, r, p);
    InterlockedAdd(counts[5], 1u);
    if (sigma > majorant + 1e-6) InterlockedAdd(counts[1], 1u);
}

/// The leaf walk's counts along the ray, and the distinct leaves a walk
/// voxel by voxel finds -- the same ray, a step of a tenth of a voxel.
[shader("compute")]
[numthreads(1, 1, 1)]
void mediumSteps(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const VolumeRecord r = volumeRecord(words, medium.volume);
    const uint2 steps = mediumLeafSteps(words, r, medium.origin, medium.direction, medium.tMin, medium.tMax);
    counts[2] = steps.x;
    counts[3] = steps.y;
    // Voxel by voxel: the leaf origin under each step, counted when it changes.
    const float3 o = mul(r.worldToIndex, float4(medium.origin, 1.0));
    const float3 d = mul(r.worldToIndex, float4(medium.direction, 0.0));
    const float stepT = 0.1 / max(max(abs(d.x), abs(d.y)), abs(d.z));
    int3 last = int3(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
    uint distinct = 0;
    for (float t = medium.tMin; t < medium.tMax; t += stepT) {
        const float3 p = o + d * t;
        if (any(p < r.boundsMin) || any(p >= r.boundsMax)) continue;
        const int3 leaf = int3(floor(p / 8.0));
        if (any(leaf != last)) {
            distinct += 1;
            last = leaf;
        }
    }
    counts[4] = distinct;
}

