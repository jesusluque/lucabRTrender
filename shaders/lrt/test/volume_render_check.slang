// Copyright (c) 2026 lucabRTrender contributors.
//
// The path tracer through an absorbing box, against Beer-Lambert. The plane
// behind is lit evenly and every sample of a pixel carries the same
// radiance, so the frame with the box over the frame without is, pixel by
// pixel, the fraction of that pixel's paths the medium let through: the
// camera's chord through the box and, where the sun casts shadows, the
// chord from the plane back towards the sun. In a medium of constant
// density each tracking is a coin of probability exp(-sigma chord), so
// the ratio is a binomial mean of exp(-sigma (camera + light chord)) whose
// standard deviation sqrt(T (1 - T) / N) the kernel derives from the number
// of paths. Nothing is chosen but how many deviations count as a failure.
import lrt.technique.camera;

struct RenderCheck {
    float3 boxMin;    // world
    float  sigma;     // extinction per world unit inside the box
    float3 boxMax;
    float  paths;     // a pixel
    float3 towardLight;   // unit, world: where the sun is
    float  planeZ;        // the lit plane, facing +z
    float  deviations;
    uint   shadows;       // 1: the sun's light crosses the box too
    uint   pad0; uint pad1;
};

struct Rows {
    float4 row0;
    float4 row1;
    float4 row2;
};

ConstantBuffer<CameraParams> camera;
ConstantBuffer<Rows>         toWorld;   // view to world
ConstantBuffer<RenderCheck>  check;
StructuredBuffer<float4>     with;
StructuredBuffer<float4>     without;
RWStructuredBuffer<uint>     counts;    // [through the box, beyond the deviations, outside it, outside and changed, uncovered]
RWStructuredBuffer<uint>     zBits;     // the sum of z^2 over pixels through the box, as float bits (one thread writes it)

float chordThroughBox(float3 o, float3 d) {
    float t0 = 0.0;
    float t1 = 1.0e30;
    for (int a = 0; a < 3; ++a) {
        if (abs(d[a]) < 1e-12) {
            if (o[a] < check.boxMin[a] || o[a] > check.boxMax[a]) return 0.0;
            continue;
        }
        float ta = (check.boxMin[a] - o[a]) / d[a];
        float tb = (check.boxMax[a] - o[a]) / d[a];
        if (ta > tb) { const float s = ta; ta = tb; tb = s; }
        t0 = max(t0, ta);
        t1 = min(t1, tb);
    }
    return max(t1 - t0, 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void volumeRenderCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    float zSquares = 0.0;
    float ratios = 0.0;
    float expected = 0.0;
    float chords = 0.0;
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            const uint at = y * camera.width + x;
            const float reference = without[at].r;
            if (reference <= 0.0) {
                counts[4] += 1;
                continue;
            }
            float3 origin;
            float3 direction;
            viewRay(camera, float2(x, y) + 0.5, origin, direction);
            const float3 o = float3(dot(toWorld.row0.xyz, origin) + toWorld.row0.w,
                                    dot(toWorld.row1.xyz, origin) + toWorld.row1.w,
                                    dot(toWorld.row2.xyz, origin) + toWorld.row2.w);
            const float3 d = normalize(float3(dot(toWorld.row0.xyz, direction), dot(toWorld.row1.xyz, direction),
                                              dot(toWorld.row2.xyz, direction)));
            float chord = chordThroughBox(o, d);
            if (check.shadows != 0 && abs(d.z) > 1e-6) {
                const float3 onPlane = o + d * ((check.planeZ - o.z) / d.z);
                chord += chordThroughBox(onPlane, check.towardLight);
            }
            const float ratio = with[at].r / reference;
            if (chord <= 0.0) {
                counts[2] += 1;
                if (abs(ratio - 1.0) > 1e-5) counts[3] += 1;
                continue;
            }
            const float T = exp(-check.sigma * chord);
            const float sd = sqrt(max(T * (1.0 - T), 0.0) / check.paths);
            const float z = (ratio - T) / max(sd, 1e-6);
            counts[0] += 1;
            zSquares += z * z;
            ratios += ratio;
            expected += T;
            chords += chord;
            if (abs(ratio - T) > check.deviations * sd + 1e-4) counts[1] += 1;
        }
    }
    zBits[0] = asuint(zSquares);
    zBits[1] = asuint(ratios);
    zBits[2] = asuint(expected);
    zBits[3] = asuint(chords);
}

struct FurnaceCheck {
    uint  pixels;
    float minAlpha;   // a pixel's paths that met the medium, at least
    uint  pad0; uint pad1;
};

ConstantBuffer<FurnaceCheck> furnace;

/// Each covered pixel's radiance over its opacity -- the mean over the
/// paths that met the medium -- summed with its square, for the mean and
/// the spread across pixels.
[shader("compute")]
[numthreads(1, 1, 1)]
void volumeFurnaceCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    float sum = 0.0;
    float squares = 0.0;
    uint judged = 0;
    uint touched = 0;
    for (uint i = 0; i < furnace.pixels; ++i) {
        const float4 c = with[i];
        if (c.a > 0.0) {
            touched += 1;
        }
        if (c.a < furnace.minAlpha) {
            continue;
        }
        const float r = (c.r + c.g + c.b) / (3.0 * c.a);
        judged += 1;
        sum += r;
        squares += r * r;
    }
    counts[0] = judged;
    counts[1] = touched;
    zBits[0] = asuint(sum);
    zBits[1] = asuint(squares);
}
