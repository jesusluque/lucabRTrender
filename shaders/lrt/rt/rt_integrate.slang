// Copyright (c) 2026 lucabRTrender contributors.
//
// What a ray does with the particles it enters, whoever finds them: the
// hardware route (rt_render.slang, a BVH of proxy triangles) and the compute
// route (rt_bvh_render.slang, an LBVH of particle boxes) both offer entries
// to an Integrator, which records, evaluates, orders, blends and segments.
//
// COST, MEASURED (Metal, M5 Pro, train_7k, 1080p, hardware route)
// An opaque closest-hit query over the 15M proxy triangles takes 6 ms a frame;
// enumerating every non-opaque proxy along every ray (~106 a ray) takes
// 255 ms. What a candidate costs is the trip back from traversal to the
// kernel, so the kernel does the least it can per candidate -- record the
// entry distance and which primitive, nothing sorted, nothing fetched -- and
// all the work on what was recorded after traversal ends. A k-nearest record
// kept sorted during traversal, restarted every 16 (3DGRT's k-buffer), cost
// 2.5 s: more trips (segments re-offer what they skip) and a struct shuffle
// in each. Colours are per particle per frame (rt_shade.slang), not per ray.
//
// SEGMENTS
// A traversal records up to kRecord entries. A ray entering more starts
// another traversal from the nearest entry it had to let go (a hair before
// it), recognising by id what the overlap offers again.
//
// ORDER
// By peak, not by entry. A particle peaks inside its proxy, so nothing a
// later segment finds (entered past where this one closed) can peak before
// that point. A segment blends what peaks up to there and carries the rest --
// large particles entered early and peaking late -- into the next. Blending
// in entry order instead draws every proxy's silhouette as a seam.
//
// What differs from the rasteriser, on purpose, and why a comparison has a
// tolerance (tests/render/test_ray_tracing.cpp):
//   - the particle is evaluated in 3D at its peak along the ray, not as an
//     EWA ellipse: exact for any lens, where EWA's affine approximation drifts
//     for large particles.
//   - no 0.3 px screen dilation: a particle smaller than a pixel is as small
//     as it is.
//   - order is by where each particle peaks along the ray (exactly, unless a
//     carry overflows kCarry); the rasteriser orders by centre depth.
//   - a camera inside a particle's bound does not see that particle: only
//     entries ahead of the ray's start count, which is what keeps a particle
//     from being taken twice when a segment restarts inside it.
module rt_integrate;
import lrt.common.color;
import rt_common;

/// Entries one traversal records. Well above what a ray meets in a trained
/// scene (train_7k: mean 106, p99 388), so most rays are one traversal.
/// Measured on train_7k at 1080p: 64 -> 845 ms, 128 -> 657, 192 -> 558,
/// 256 -> 530, 384 -> 548, 512 -> 553.
public static const uint kRecord = 256;
/// Particles evaluated but not yet blended that a segment may carry.
public static const uint kCarry = 64;
public static const uint kPending = kRecord + kCarry;
/// Overlap-window ids remembered from the previous segment.
public static const uint kWindow = 16;
/// Relative overlap between consecutive segments.
public static const float kOverlap = 1e-5;
public static const float kOpen = 3.4e38;
/// Buckets of the counting sort that orders a segment's particles by peak.
/// Measured on the compute route: 32 -> 362 ms, 64 -> 351, 128 -> 344,
/// 256 -> 342.
public static const uint kBuckets = 128;

public StructuredBuffer<float4>   frames;          // per particle, four: rt_frames.slang
public StructuredBuffer<float4>   colours;         // per instance particle: rt_shade.slang
public StructuredBuffer<float4>   instanceData;    // per instance: world->cloud rows 0..2
public StructuredBuffer<uint>     instanceIndices; // per instance: id base, colour offset
public RWStructuredBuffer<float4> colour;
public RWStructuredBuffer<float>  depth;
public ConstantBuffer<RtParams>   params;

/// A ray, as both routes carry it. Not `RayDesc`: that type exists on a
/// target only where ray tracing does, and the compute BVH route is meant to
/// run where there is none (CUDA without OptiX -- measured, nvrtc: "identifier
/// RayDesc is undefined"). The hardware route fills a RayDesc from this.
public struct LrtRay {
    public float3 Origin;
    public float  TMin;
    public float3 Direction;
    public float  TMax;
};

/// The ray through pixel (px, py) in world space. Its direction has view
/// z = 1, so t is view depth for every ray.
public LrtRay primaryRay(uint px, uint py) {
    const float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);
    float3 viewOrigin = float3(0.0);
    float3 viewDirection;
    if (params.orthographic != 0) {
        viewOrigin = float3((pixel.x - params.centreX) / params.focalX,
                            (pixel.y - params.centreY) / params.focalY, 0.0);
        viewDirection = float3(0.0, 0.0, 1.0);
    } else {
        viewDirection = float3((pixel.x - params.centreX) / params.focalX,
                               (pixel.y - params.centreY) / params.focalY, 1.0);
    }
    const float3x3 toWorld = float3x3(params.v00, params.v01, params.v02, params.v10, params.v11,
                                      params.v12, params.v20, params.v21, params.v22);
    LrtRay ray;
    ray.Origin = mul(toWorld, viewOrigin) + float3(params.v03, params.v13, params.v23);
    ray.Direction = mul(toWorld, viewDirection);
    ray.TMin = params.nearZ;
    ray.TMax = params.farZ;
    return ray;
}

/// The ray in an instance's cloud space. Not normalised: t means the same.
public void toCloud(LrtRay ray, uint instance, out float3 origin, out float3 direction) {
    const float4 w0 = instanceData[instance * 3];
    const float4 w1 = instanceData[instance * 3 + 1];
    const float4 w2 = instanceData[instance * 3 + 2];
    const float3x3 m = float3x3(w0.xyz, w1.xyz, w2.xyz);
    origin = mul(m, ray.Origin) + float3(w0.w, w1.w, w2.w);
    direction = mul(m, ray.Direction);
}

/// How a route's recorded primitive index names a particle of its instance.
/// A compile-time choice, so the division is by a constant.
public interface IPrimitives {
    static uint particleOf(uint raw);
}

public enum Offer {
    Skipped,    // at or past where the record closed, or seen last segment
    Recorded,
    Dropped,    // the record was full and this was the farthest: nothing past it is wanted
}

/// The scalars traversal touches for every entry, kept apart from the
/// integrator's arrays: a separate small local stays in registers, where the
/// same fields inside the integrator went through memory and cost 11% of the
/// frame (train_7k, 1080p, Metal).
public struct Cursor {
    public uint count;
    public float farthest;
    public uint farthestAt;
    /// Where the record is complete: every entry before it was offered and
    /// kept. Dropping an entry closes the record at it.
    public float closed;
    public float windowEnd;
};

public struct Integrator<P : IPrimitives> {
    // Not zeroed: every slot is written before it is read, and zeroing the
    // ~1000 slots cost 8% of the frame.
    // One traversal's record, in the order traversal offered it.
    float recordT[kRecord];
    uint recordRaw[kRecord];       // primitive (or leaf) index; id = base + P.particleOf(raw)
    uint recordInstance[kRecord];
    // Evaluated, not yet blended, in no order; `order` sorts them by peak.
    // Sorting indices moves one word where the entries are four.
    uint pending;
    float peakT[kPending];
    float peakAlpha[kPending];
    uint peakId[kPending];
    uint peakInstance[kPending];
    uint order[kPending];
    // What the previous segment recorded in the overlap window.
    uint windowCount;
    uint windowId[kWindow];
    uint windowInstance[kWindow];
    // The pixel.
    public float3 rgb;
    public float transmittance;
    float depthSum;
    float depthAt;
    uint lastId;
    uint lastInstance;
    float lastT;

    [mutating]
    [ForceInline]
    public void begin(out Cursor cursor) {
        cursor.count = 0;
        cursor.windowEnd = -1.0;
        cursor.closed = kOpen;
        cursor.farthest = -1.0;
        cursor.farthestAt = 0;
        pending = 0;
        windowCount = 0;
        rgb = float3(0.0);
        transmittance = 1.0;
        depthSum = 0.0;
        depthAt = -1.0;
        lastId = 0xFFFFFFFFu;
        lastInstance = 0xFFFFFFFFu;
        lastT = -1.0;
    }

    [mutating]
    [ForceInline]
    public void beginSegment(inout Cursor cursor) {
        cursor.closed = kOpen;
        cursor.farthest = -1.0;
        cursor.farthestAt = 0;
        cursor.count = 0;
    }

    /// An entry at `t` into primitive `raw` of `instance`. `id` is only asked
    /// for when the entry falls in the overlap with the previous segment.
    [mutating]
    [ForceInline]
    public Offer offer(inout Cursor cursor, float t, uint raw, uint instance) {
        if (t >= cursor.closed) {
            return Offer.Skipped;
        }
        if (t <= cursor.windowEnd) {
            const uint id = instanceIndices[instance * 2] + P.particleOf(raw);
            bool again = false;
            for (uint k = 0; k < windowCount; ++k) {
                again = again || (windowId[k] == id && windowInstance[k] == instance);
            }
            if (again) {
                return Offer.Skipped;
            }
        }
        if (cursor.count < kRecord) {
            recordT[cursor.count] = t;
            recordRaw[cursor.count] = raw;
            recordInstance[cursor.count] = instance;
            if (t > cursor.farthest) {
                cursor.farthest = t;
                cursor.farthestAt = cursor.count;
            }
            cursor.count += 1;
            return Offer.Recorded;
        }
        if (t >= cursor.farthest) {
            cursor.closed = t;
            return Offer.Dropped;
        }
        // The farthest goes; the record closes there.
        cursor.closed = cursor.farthest;
        recordT[cursor.farthestAt] = t;
        recordRaw[cursor.farthestAt] = raw;
        recordInstance[cursor.farthestAt] = instance;
        cursor.farthest = -1.0;
        for (uint k = 0; k < kRecord; ++k) {
            if (recordT[k] > cursor.farthest) {
                cursor.farthest = recordT[k];
                cursor.farthestAt = k;
            }
        }
        return Offer.Recorded;
    }

    /// `order` = the pending entries by peak, ties in the order they were
    /// evaluated. A counting sort into kBuckets by peak, then an insertion
    /// sort that only has the buckets' insides left to do: insertion sort
    /// alone moved ~n^2/4 entries (~2500 a ray) and cost 157 ms of a 427 ms
    /// frame (train_7k, 1080p, compute route).
    [mutating]
    [ForceInline]
    void sortPending() {
        float lo = kOpen;
        float hi = -kOpen;
        for (uint k = 0; k < pending; ++k) {
            lo = min(lo, peakT[k]);
            hi = max(hi, peakT[k]);
        }
        const float scale = hi > lo ? float(kBuckets) / (hi - lo) : 0.0;
        uint starts[kBuckets];
        for (uint b = 0; b < kBuckets; ++b) {
            starts[b] = 0;
        }
        uint bucket[kPending];
        for (uint k = 0; k < pending; ++k) {
            bucket[k] = min(uint((peakT[k] - lo) * scale), kBuckets - 1);
            starts[bucket[k]] += 1;
        }
        uint running = 0;
        for (uint b = 0; b < kBuckets; ++b) {
            const uint n = starts[b];
            starts[b] = running;
            running += n;
        }
        for (uint k = 0; k < pending; ++k) {
            order[starts[bucket[k]]] = k;
            starts[bucket[k]] += 1;
        }
        for (uint k = 1; k < pending; ++k) {
            const uint slot = order[k];
            const float t = peakT[slot];
            uint at = k;
            while (at > 0 && peakT[order[at - 1]] > t) {
                order[at] = order[at - 1];
                at -= 1;
            }
            order[at] = slot;
        }
    }

    /// Evaluates the record, blends what is settled, carries the rest.
    /// Evaluation happens here even for a route that already had the peak in
    /// hand when it offered the entry: carrying peak and alpha in the record
    /// cost more than evaluating again (344 -> 389 ms, compute route). False
    /// when the ray is done; otherwise `ray.TMin` is where the next segment
    /// starts. `last`: no segment follows, whatever is pending goes on.
    [mutating]
    [ForceInline]
    public bool endSegment(inout Cursor cursor, inout LrtRay ray, bool last) {
        const float closed = cursor.closed;
        const uint recordCount = cursor.count;
        const bool exhausted = closed >= kOpen;
        const float settled = exhausted || last ? kOpen : closed;

        uint nextWindow = 0;
        const float windowStart = closed * (1.0 - 2.0 * kOverlap);
        for (uint r = 0; r < recordCount; ++r) {
            const float entry = recordT[r];
            if (entry >= closed) {
                continue;
            }
            const uint instance = recordInstance[r];
            const uint id = instanceIndices[instance * 2] + P.particleOf(recordRaw[r]);
            if (!exhausted && entry >= windowStart && nextWindow < kWindow) {
                windowId[nextWindow] = id;
                windowInstance[nextWindow] = instance;
                nextWindow += 1;
            }
            float3 origin;
            float3 direction;
            toCloud(ray, instance, origin, direction);
            const float4 f0 = frames[id * 4];
            const float4 f1 = frames[id * 4 + 1];
            const float4 f2 = frames[id * 4 + 2];
            // The opacity this instance draws the particle with, after its
            // SplatEdit (rt_shade.slang); zero is removed. The cut follows it,
            // as the rasteriser's does.
            const float opacity = colours[id + instanceIndices[instance * 2 + 1]].w;
            if (!(opacity >= 1.0 / 255.0)) {
                continue;
            }
            // Into the particle's frame, where it is the unit sphere.
            const float3x3 m = float3x3(f0.xyz, f1.xyz, f2.xyz);
            const float3 o = mul(m, origin - float3(f0.w, f1.w, f2.w));
            const float3 d = mul(m, direction);
            const float tPeak = -dot(o, d) / max(dot(d, d), 1e-30);
            const float3 closest = o + tPeak * d;
            const float power = dot(closest, closest);
            if (power > rtCutoff(opacity) || tPeak < params.nearZ || tPeak > params.farZ) {
                continue;
            }
            const float alpha = min(0.99, opacity * exp(-0.5 * power));
            if (alpha < 1.0 / 255.0) {
                continue;
            }
            peakT[pending] = tPeak;
            peakAlpha[pending] = alpha;
            peakId[pending] = id;
            peakInstance[pending] = instance;
            pending += 1;
        }
        sortPending();
        windowCount = nextWindow;
        cursor.windowEnd = closed * (1.0 + kOverlap) + kOverlap;

        // Blend what nothing later can precede; past a carry of kCarry, blend
        // the nearest anyway (the only place order can be approximate).
        uint e = 0;
        while (e < pending && transmittance >= 1e-4 && (peakT[order[e]] <= settled || pending - e > kCarry)) {
            const uint slot = order[e];
            const uint id = peakId[slot];
            const uint instance = peakInstance[slot];
            const float tPeak = peakT[slot];
            // Two faces of one proxy offered in one traversal give the same
            // particle at the same peak twice.
            if (!(id == lastId && instance == lastInstance && tPeak == lastT)) {
                lastId = id;
                lastInstance = instance;
                lastT = tPeak;
                const float3 c = colours[id + instanceIndices[instance * 2 + 1]].xyz;
                const float weight = transmittance * peakAlpha[slot];
                rgb += weight * c;
                depthSum += weight * tPeak;
                transmittance *= 1.0 - peakAlpha[slot];
                if (depthAt < 0.0 && 1.0 - transmittance >= params.depthThreshold) {
                    depthAt = tPeak;
                }
            }
            e += 1;
        }
        if (exhausted || last || transmittance < 1e-4) {
            return false;
        }
        // Carry the rest, in order, to the front: at most kCarry of them.
        uint carried = 0;
        float carryT[kCarry];
        float carryAlpha[kCarry];
        uint carryId[kCarry];
        uint carryInstance[kCarry];
        for (uint k = e; k < pending; ++k) {
            const uint slot = order[k];
            carryT[carried] = peakT[slot];
            carryAlpha[carried] = peakAlpha[slot];
            carryId[carried] = peakId[slot];
            carryInstance[carried] = peakInstance[slot];
            carried += 1;
        }
        for (uint k = 0; k < carried; ++k) {
            peakT[k] = carryT[k];
            peakAlpha[k] = carryAlpha[k];
            peakId[k] = carryId[k];
            peakInstance[k] = carryInstance[k];
            order[k] = k;
        }
        pending = carried;
        ray.TMin = closed * (1.0 - kOverlap);
        return true;
    }

    /// The finished pixel into the targets. Mutating only so that the call
    /// does not copy the integrator (a non-mutating method takes a copy).
    [mutating]
    [ForceInline]
    public void write(uint px, uint py) {
        const float coverage = 1.0 - transmittance;
        float3 linearRgb = rgb;
        if (params.linearise != 0 && coverage > 1e-6) {
            linearRgb = srgbToLinear(rgb / coverage) * coverage;
        }
        const uint at = py * params.width + px;
        colour[at] = float4(linearRgb + transmittance * float3(params.bgR, params.bgG, params.bgB),
                            coverage + transmittance * params.bgA);
        depth[at] = params.depthMode == 0 ? (coverage > 1e-6 ? depthSum / coverage : 0.0) : max(depthAt, 0.0);
    }
};
