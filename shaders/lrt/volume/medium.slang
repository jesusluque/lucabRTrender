// Copyright (c) 2026 lucabRTrender contributors.
//
// A participating medium on a NanoVDB grid, as a path meets it: free-flight
// sampling by delta tracking, transmittance by ratio tracking, and the
// Henyey-Greenstein phase function. Both trackings walk the grid's leaves
// with PNanoVDB's HDDA at the leaf's size and take each leaf's own
// majorant -- the largest density in it, a kernel's (volume_prepare) --
// so an empty leaf costs one step and a dense one is sampled at its own
// rate. Everything is in the volume's index space, with the ray's
// parameter kept in world units: the index-space direction is the world
// one through the linear part of world-to-index, unnormalised, so `t`
// means the same distance on both sides.
module medium;

#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"

/// A volume as the frame's words hold it (VolumeSet lays it out):
/// 32 words a record, after the header.
public struct VolumeRecord {
    public uint     gridWord;      // the grid's word offset
    public uint     leafBase;      // word offset of its leaf maxima (floats)
    public uint     leafCount;
    public uint     leafWord0;     // word offset of leaf 0 in the tree, for a leaf's index from its address
    public float3x4 worldToIndex;  // rows
    public float3   boundsMin;     // index space, from the leaves
    public float3   boundsMax;
    public float    sigmaScale;    // density to extinction, per world unit
    public float3   albedo;        // single scattering
    public float    g;             // Henyey-Greenstein asymmetry
    public float    majorant;      // the grid's largest density (a kernel's)
};

public static const uint kVolumeHeaderWords = 8;
public static const uint kVolumeRecordWords = 32;

public uint volumeCount(StructuredBuffer<uint> words) { return words[0]; }

public VolumeRecord volumeRecord(StructuredBuffer<uint> words, uint index) {
    const uint b = kVolumeHeaderWords + index * kVolumeRecordWords;
    VolumeRecord r;
    r.gridWord = words[b + 0];
    r.leafBase = words[b + 1];
    r.leafCount = words[b + 2];
    r.leafWord0 = words[b + 3];
    r.worldToIndex = float3x4(asfloat(words[b + 4]), asfloat(words[b + 5]), asfloat(words[b + 6]), asfloat(words[b + 7]),
                              asfloat(words[b + 8]), asfloat(words[b + 9]), asfloat(words[b + 10]), asfloat(words[b + 11]),
                              asfloat(words[b + 12]), asfloat(words[b + 13]), asfloat(words[b + 14]), asfloat(words[b + 15]));
    r.boundsMin = float3(asfloat(words[b + 16]), asfloat(words[b + 17]), asfloat(words[b + 18]));
    r.boundsMax = float3(asfloat(words[b + 19]), asfloat(words[b + 20]), asfloat(words[b + 21]));
    r.sigmaScale = asfloat(words[b + 22]);
    r.albedo = float3(asfloat(words[b + 23]), asfloat(words[b + 24]), asfloat(words[b + 25]));
    r.g = asfloat(words[b + 26]);
    r.majorant = asfloat(words[b + 27]);
    return r;
}

// --- Random numbers: a counter hashed, advanced a draw at a time ------------

uint mediumHash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

/// A uniform in [0, 1) from the state, which advances.
public float mediumRandom(inout uint state) {
    state = state * 747796405u + 2891336453u;
    return float(mediumHash(state) >> 8) / 16777216.0;
}

// --- The phase function -------------------------------------------------------

public float hgPhase(float cosTheta, float g) {
    const float denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * 3.14159265358979 * denom * sqrt(max(denom, 1e-8)));
}

/// A direction drawn from the phase about `wo` (pointing away from the
/// point, towards where the light came from -- the same convention as a
/// lobe's), with the phase's own density, so weight = phase / pdf = 1.
public float3 hgSample(float3 wo, float g, float2 u) {
    float cosTheta;
    if (abs(g) < 1e-3) {
        cosTheta = 1.0 - 2.0 * u.x;
    } else {
        const float sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u.x);
        cosTheta = -(1.0 + g * g - sq * sq) / (2.0 * g);
    }
    const float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    const float phi = 2.0 * 3.14159265358979 * u.y;
    // A frame about -wo: the scattered direction leans forward for g > 0,
    // continuing the way the light was travelling.
    const float3 forward = -wo;
    const float3 a = abs(forward.x) > 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 t = normalize(cross(a, forward));
    const float3 b = cross(forward, t);
    return normalize(sinTheta * cos(phi) * t + sinTheta * sin(phi) * b + cosTheta * forward);
}

// --- The grid, as a ray sees it -----------------------------------------------

struct IndexRay {
    float3 origin;
    float3 direction;   // unnormalised: t stays the world parameter
};

IndexRay toIndex(VolumeRecord r, float3 origin, float3 direction) {
    IndexRay ray;
    ray.origin = mul(r.worldToIndex, float4(origin, 1.0));
    ray.direction = mul(r.worldToIndex, float4(direction, 0.0));
    return ray;
}

/// The ray's parameter range inside the volume's index bounds.
bool clipToBounds(VolumeRecord r, IndexRay ray, inout float t0, inout float t1) {
    for (int axis = 0; axis < 3; ++axis) {
        const float d = ray.direction[axis];
        const float o = ray.origin[axis];
        if (abs(d) < 1e-12) {
            if (o < r.boundsMin[axis] || o >= r.boundsMax[axis]) return false;
            continue;
        }
        float ta = (r.boundsMin[axis] - o) / d;
        float tb = (r.boundsMax[axis] - o) / d;
        if (ta > tb) { const float s = ta; ta = tb; tb = s; }
        t0 = max(t0, ta);
        t1 = min(t1, tb);
        if (t1 <= t0) return false;
    }
    return t1 > t0;
}

pnanovdb_grid_handle_t gridOf(VolumeRecord r) {
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = r.gridWord * 4u;
    return grid;
}

/// The leaf under `ijk` and its majorant (0 where no leaf is): the accessor
/// caches the leaf, and its address gives its index in the leaf array.
float leafMajorantAt(StructuredBuffer<uint> words, VolumeRecord r, inout pnanovdb_readaccessor_t acc,
                     pnanovdb_coord_t ijk) {
    const uint dim = pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, words, acc, ijk);
    if (dim != 1u) {
        return 0.0;   // a tile or the background: tiles were voxelised, so nothing is here
    }
    const uint leafSize = PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_FLOAT, leaf_size);
    const uint index = (acc.leaf.address.byte_offset - r.leafWord0 * 4u) / leafSize;
    if (index >= r.leafCount) {
        return 0.0;
    }
    return asfloat(words[r.leafBase + index]) * r.sigmaScale;
}

float densityAt(StructuredBuffer<uint> words, inout pnanovdb_readaccessor_t acc, float3 p) {
    pnanovdb_coord_t ijk;
    ijk.x = int(floor(p.x));
    ijk.y = int(floor(p.y));
    ijk.z = int(floor(p.z));
    const pnanovdb_address_t at = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, words, acc, ijk);
    return pnanovdb_read_float(words, at);
}

static const int kMaxLeafSteps = 2048;
static const int kMaxCollisions = 4096;

/// Delta tracking from `tMin` to `tMax` along the world ray: whether the
/// medium scatters the path before `tMax`, and where. Leaf by leaf, each at
/// its own majorant; a null collision costs one density lookup.
public bool mediumScatter(StructuredBuffer<uint> words, VolumeRecord r, float3 origin, float3 direction, float tMin,
                          float tMax, inout uint rng, out float tScatter) {
    tScatter = tMax;
    const IndexRay ray = toIndex(r, origin, direction);
    float t0 = tMin;
    float t1 = tMax;
    if (!clipToBounds(r, ray, t0, t1)) {
        return false;
    }
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, pnanovdb_tree_get_root(words, pnanovdb_grid_get_tree(words, gridOf(r))));
    pnanovdb_hdda_t hdda;
    pnanovdb_hdda_init(hdda, ray.origin, t0, ray.direction, t1, 8);
    for (int step = 0; step < kMaxLeafSteps; ++step) {
        const float tA = hdda.tmin;
        const float tB = min(min(hdda.next.x, hdda.next.y), min(hdda.next.z, hdda.tmax));
        if (tB > tA + 1e-6 * max(abs(tA), 1.0)) {   // not a sliver of rounding at a boundary
            const float majorant = leafMajorantAt(words, r, acc, hdda.voxel);
            if (majorant > 0.0) {
                float t = tA;
                for (int k = 0; k < kMaxCollisions; ++k) {
                    t -= log(max(1.0 - mediumRandom(rng), 1e-7)) / majorant;
                    if (t >= tB) break;
                    const float sigma = densityAt(words, acc, ray.origin + ray.direction * t) * r.sigmaScale;
                    if (mediumRandom(rng) * majorant < sigma) {
                        tScatter = t;
                        return true;
                    }
                }
            }
        }
        if (!pnanovdb_hdda_step(hdda)) break;
    }
    return false;
}

/// Ratio tracking: the transmittance from `tMin` to `tMax`, an unbiased
/// estimate with the same walk.
public float mediumTransmittance(StructuredBuffer<uint> words, VolumeRecord r, float3 origin, float3 direction,
                                 float tMin, float tMax, inout uint rng) {
    const IndexRay ray = toIndex(r, origin, direction);
    float t0 = tMin;
    float t1 = tMax;
    if (!clipToBounds(r, ray, t0, t1)) {
        return 1.0;
    }
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, pnanovdb_tree_get_root(words, pnanovdb_grid_get_tree(words, gridOf(r))));
    pnanovdb_hdda_t hdda;
    pnanovdb_hdda_init(hdda, ray.origin, t0, ray.direction, t1, 8);
    float transmittance = 1.0;
    for (int step = 0; step < kMaxLeafSteps; ++step) {
        const float tA = hdda.tmin;
        const float tB = min(min(hdda.next.x, hdda.next.y), min(hdda.next.z, hdda.tmax));
        if (tB > tA + 1e-6 * max(abs(tA), 1.0)) {   // not a sliver of rounding at a boundary
            const float majorant = leafMajorantAt(words, r, acc, hdda.voxel);
            if (majorant > 0.0) {
                float t = tA;
                for (int k = 0; k < kMaxCollisions; ++k) {
                    t -= log(max(1.0 - mediumRandom(rng), 1e-7)) / majorant;
                    if (t >= tB) break;
                    const float sigma = densityAt(words, acc, ray.origin + ray.direction * t) * r.sigmaScale;
                    transmittance *= max(1.0 - sigma / majorant, 0.0);
                    if (transmittance < 1e-4) return 0.0;
                }
            }
        }
        if (!pnanovdb_hdda_step(hdda)) break;
    }
    return transmittance;
}

/// The density at a world point, for a check.
public float mediumDensityAt(StructuredBuffer<uint> words, VolumeRecord r, float3 p) {
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, pnanovdb_tree_get_root(words, pnanovdb_grid_get_tree(words, gridOf(r))));
    return densityAt(words, acc, mul(r.worldToIndex, float4(p, 1.0)));
}

/// The leaf majorant over a world point (0 where no leaf is), for a check.
public float mediumMajorantAt(StructuredBuffer<uint> words, VolumeRecord r, float3 p) {
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, pnanovdb_tree_get_root(words, pnanovdb_grid_get_tree(words, gridOf(r))));
    const float3 pi = mul(r.worldToIndex, float4(p, 1.0));
    pnanovdb_coord_t ijk;
    ijk.x = int(floor(pi.x));
    ijk.y = int(floor(pi.y));
    ijk.z = int(floor(pi.z));
    return leafMajorantAt(words, r, acc, ijk);
}

/// How many leaves the ray's walk visits between `tMin` and `tMax`, and
/// how many of them hold a leaf: what an HDDA at the leaf's size must
/// agree with a voxel-by-voxel walk on.
public uint2 mediumLeafSteps(StructuredBuffer<uint> words, VolumeRecord r, float3 origin, float3 direction, float tMin,
                             float tMax) {
    const IndexRay ray = toIndex(r, origin, direction);
    float t0 = tMin;
    float t1 = tMax;
    if (!clipToBounds(r, ray, t0, t1)) {
        return uint2(0, 0);
    }
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, pnanovdb_tree_get_root(words, pnanovdb_grid_get_tree(words, gridOf(r))));
    pnanovdb_hdda_t hdda;
    pnanovdb_hdda_init(hdda, ray.origin, t0, ray.direction, t1, 8);
    uint2 counts = uint2(0, 0);
    for (int step = 0; step < kMaxLeafSteps; ++step) {
        const float tA = hdda.tmin;
        const float tB = min(min(hdda.next.x, hdda.next.y), min(hdda.next.z, hdda.tmax));
        if (tB > tA + 1e-6 * max(abs(tA), 1.0)) {   // not a sliver of rounding at a boundary
            counts.x += 1;
            if (leafMajorantAt(words, r, acc, hdda.voxel) > 0.0) counts.y += 1;
        }
        if (!pnanovdb_hdda_step(hdda)) break;
    }
    return counts;
}

