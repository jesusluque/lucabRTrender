// Copyright (c) 2026 lucabRTrender contributors.
//
// Whether the path tracer's sampler gives each pixel its own sample sequence.
//
// PathTracer's random() mixes the pixel, the sample index, the bounce and the
// dimension with XOR before a murmur-style avalanche. XOR is a GF(2) tensor:
// the pre-avalanche value for a pixel is the sample's contribution XORed with a
// per-pixel constant, so two pixels whose constants differ in bits the sample
// also varies can walk the *same* set of pre-values in a different order --
// same sample set, same mean, correlated errors. Correlated pixel errors leave
// E[relMSE] alone and multiply the variance of its mean over the frame, which
// is what a draw-to-draw spread of 3.4x at one sample count looks like.
//
// This measures it instead of inferring it: a fingerprint of each pixel's first
// draws, sorted, and equal neighbours counted. A per-pixel repeat counter is
// the armed control -- it must be zero, since within a pixel the sample index
// is bijective through the avalanche.
module rng_probe;

struct ProbeParams {
    uint width;
    uint height;
    uint seed;      // what PathTracer would pass as path.seed
    uint draws;     // how many of a pixel's samples the fingerprint folds in
};

RWStructuredBuffer<uint>  keysLo;
RWStructuredBuffer<uint>  keysHi;
RWStructuredBuffer<uint>  values;
RWStructuredBuffer<uint>  counts;    // [0] duplicate fingerprints, [1] within-pixel repeats
ConstantBuffer<ProbeParams> probe;

/// PathTracer's random(), copied exactly so the probe measures the sampler the
/// renderer uses and not a paraphrase of it.
uint rawHash(uint2 pixel, uint sample, uint bounce, uint which, uint seed) {
    uint h = pixel.x * 73856093u ^ pixel.y * 19349663u ^ (sample + seed) * 83492791u ^
             bounce * 2654435761u ^ which * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return h;
}

/// A 64-bit fingerprint of the first `draws` samples a pixel takes, in the
/// dimension the light choice uses (which = 11, bounce = 0).
[shader("compute")]
[numthreads(64, 1, 1)]
void rngFingerprint(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    const uint pixels = probe.width * probe.height;
    if (at >= pixels) {
        return;
    }
    const uint2 pixel = uint2(at % probe.width, at / probe.width);
    uint lo = 0;
    uint hi = 0;
    for (uint k = 0; k < probe.draws; ++k) {
        const uint h = rawHash(pixel, k, 0u, 11u, probe.seed);
        // Fold so every draw reaches both halves; order matters, so a pixel
        // drawing the same set in another order gets a different fingerprint
        // only if the fold is order sensitive -- which it is.
        lo = lo * 2654435761u + h;
        hi = hi ^ (h + lo);
    }
    keysLo[at] = lo;
    keysHi[at] = hi;
    values[at] = at;
}

/// Equal neighbours in the sorted keys: pixels whose first draws coincide.
[shader("compute")]
[numthreads(1, 1, 1)]
void rngDuplicates(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint pixels = probe.width * probe.height;
    uint duplicates = 0;
    for (uint i = 1; i < pixels; ++i) {
        if (keysLo[i] == keysLo[i - 1] && keysHi[i] == keysHi[i - 1]) {
            ++duplicates;
        }
    }
    counts[0] = duplicates;
}

/// The control: within one pixel, `draws` samples must give `draws` distinct
/// values. Counted by brute force over a single pixel, which is cheap and
/// needs no sort.
[shader("compute")]
[numthreads(1, 1, 1)]
void rngWithinPixel(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint2 pixel = uint2(probe.width / 2u, probe.height / 2u);
    uint repeats = 0;
    for (uint a = 0; a < probe.draws; ++a) {
        const uint ha = rawHash(pixel, a, 0u, 11u, probe.seed);
        for (uint b = a + 1u; b < probe.draws; ++b) {
            if (rawHash(pixel, b, 0u, 11u, probe.seed) == ha) {
                ++repeats;
            }
        }
    }
    counts[1] = repeats;
}

/// The same fingerprint, made commutative: a sum and an XOR, both order
/// independent. Two pixels drawing the same SET of values in any order collide
/// here, which is the shape a GF(2) mix would produce and which the ordered
/// fingerprint above is blind to.
[shader("compute")]
[numthreads(64, 1, 1)]
void rngSetPrint(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    const uint pixels = probe.width * probe.height;
    if (at >= pixels) {
        return;
    }
    const uint2 pixel = uint2(at % probe.width, at / probe.width);
    uint sum = 0;
    uint x = 0;
    for (uint k = 0; k < probe.draws; ++k) {
        const uint h = rawHash(pixel, k, 0u, 11u, probe.seed);
        sum += h;
        x ^= h;
    }
    keysLo[at] = sum;
    keysHi[at] = x;
    values[at] = at;
}

