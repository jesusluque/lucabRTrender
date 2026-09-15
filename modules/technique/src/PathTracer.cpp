// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/PathTracer.h"

#include <algorithm>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

namespace {

/// The same lights, lobes and surface the raster shading uses, plus a bounce.
/// Written as a generated kernel for the same reason MaterialShading is: it
/// dispatches to whatever materials the frame compiled.
const char* kPrelude = R"(
import lrt.light.lights_image;
import lrt.light.light_bvh;

struct PathParams {
    uint samples;      // paths a pixel this call
    uint bounces;      // indirect bounces after the first hit
    uint seed;         // which samples these are
    uint accumulated;  // paths a pixel already in `sum`
    uint chooseLights;
    uint headlight;    // 1: with no lights, the first vertex is lit from the eye, as the raster lights it
    uint writeAux;     // 1: write the first hit's albedo and normal
    uint adaptive;     // 1: a converged pixel takes no more paths
    float errorTarget; // relative standard error of the mean a pixel stops at
    uint minSamples;   // and not before this many
    uint ownRays;      // 1: the tracer casts its own primary rays (lens, distortion, motion)
    float lensRadius;  // scene units; 0 for a pinhole
    float focusDistance;
    float distortionK1;
    float distortionK2;
    uint buckets;      // motion blur: shutter slices; a sample's rays answer to one slice's mask bit
    uint mis;          // 1: the lights' and the material's strategies are weighed (power heuristic)
    uint cameraMoves;  // 1: the camera moves under the shutter: its view to world between two samples
    float4 cameraStart0; float4 cameraStart1; float4 cameraStart2;   // view to world at the earlier sample
    float4 cameraEnd0; float4 cameraEnd1; float4 cameraEnd2;         // and at the later
    float cameraTime0; // when the two samples are, and the shutter, in the same units
    float cameraTime1;
    float shutterOpen;
    float shutterClose;
};

Texture2D<uint4>               visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>     sum;          // paths added so far, a pixel
RWStructuredBuffer<float4>     colour;       // their mean
RWStructuredBuffer<float>      depth;
RWStructuredBuffer<float4>     aux;          // written when path.writeAux: the albedo's plane, then the normal's
RWStructuredBuffer<uint>       moments;      // the luminance's second moment as float bits, a pixel; then 1 once adaptive sampling stopped the pixel, a pixel

static const float3 kPathLuminance = float3(0.2126, 0.7152, 0.0722);
ConstantBuffer<CameraParams>   camera;
StructuredBuffer<LightRecord>  lights;
uniform uint                   lightCount;
uniform uint                   lightMotionBase;   // in iesValues: moving lights' samples, 26 floats a record; 0: none moves
uniform uint                   lightNodeBase;
uniform uint                   lightTreeNodes;
uniform uint                   lightUnboundedCount;
ConstantBuffer<PathParams>     path;

/// One round of PCG's output permutation over an LCG step: a hash of a
/// counter whose every input bit reaches every output bit.
uint pcgHash(uint input) {
    const uint state = input * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Everything that makes a sample -- the pixel, the frame's stream, which path
/// it is, which bounce, which dimension -- folded in one after another, each
/// through the full hash, so no two inputs meet by XOR and no arithmetic
/// progression in one survives into the output.
///
/// The path's index is absolute: `path.accumulated + sample`, so a frame
/// gathered in one pass and in many draws the same samples (checked to 4e-15).
///
/// What was here before mixed the inputs by XOR before one avalanche. Every
/// pixel had its own sequence and its own set (measured: 0 of 3072 shared
/// either), and neighbours' errors were independent (block variance fell
/// 4.42x and 15.94x for 2x2 and 4x4 against 4 and 16) -- and still the
/// variance of a pixel's mean fell as N^-0.86 rather than N^-1, which is
/// what samples correlated *within* a pixel look like. Fitted exponent on
/// sqrt: -0.428 before this change; the number after it is in the test.
float random(uint2 pixel, uint sample, uint bounce, uint which) {
    uint key = pcgHash(pixel.y * camera.width + pixel.x);
    key = pcgHash(key + path.seed);
    key = pcgHash(key + path.accumulated + sample);
    key = pcgHash(key + bounce);
    key = pcgHash(key + which);
    return float(key >> 8) * (1.0 / 16777216.0);
}

float2 random2(uint2 pixel, uint sample, uint bounce, uint which) {
    return float2(random(pixel, sample, bounce, which), random(pixel, sample, bounce, which + 1u));
}

// There was a power heuristic here. It is gone rather than left unused,
// because a function of that name invites being plugged back in: weighing
// next event estimation against a strategy that covers no analytic light is
// what lost half the dome. When mesh lights arrive the two strategies will
// genuinely overlap and a weight will belong again -- with a "does this
// direction reach light k, and with what radiance" beside lightPdf, which
// does not exist yet.
)";

/// Where the device traces, an indirect ray finds the next surface and a
/// shadow ray decides whether a light is reached. Without ray queries there is
/// no bounce to trace, so the tracer is direct light alone and says so.
const char* kRays = R"(
uniform RaytracingAccelerationStructure scene;

struct PathHit {
    uint4  seen;          // (instance + 1, triangle, 0, 0); 0 for nothing
    float  t;
    float2 barycentrics;  // of the committed triangle: where the ray met it
};

PathHit traceNearestFrom(float3 origin, float3 direction, float tMin, uint mask) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = 3.0e38;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
    query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, mask, ray);
    query.Proceed();
    PathHit hit;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        hit.seen = uint4(query.CommittedInstanceID() + 1, query.CommittedPrimitiveIndex(), 0, 0);
        hit.t = query.CommittedRayT();
        hit.barycentrics = query.CommittedTriangleBarycentrics();
    } else {
        hit.seen = uint4(0);
        hit.t = 0.0;
        hit.barycentrics = float2(0.0);
    }
    return hit;
}

bool pathOccluded(float3 p, float3 n, float3 wi, float distance, uint shadowCategory, uint mask) {
    const float scale = max(1.0, length(p));
    const float3 away = dot(n, wi) < 0.0 ? -n : n;
    RayDesc ray;
    ray.Origin = p + (away + wi) * (1.0e-3 * scale);
    ray.Direction = wi;
    ray.TMin = 1.0e-3 * scale;
    ray.TMax = max(distance - ray.TMin, 0.0);
    if (ray.TMax <= ray.TMin) {
        return false;
    }
    if (shadowCategory == kLightUnlinked) {
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, mask, ray);
        query.Proceed();
        return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    }
    for (uint step = 0; step < 16; ++step) {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, mask, ray);
        query.Proceed();
        if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            return false;
        }
        if (lightLinked(shadowCategory, instances[query.CommittedInstanceID()].categoriesLo,
                        instances[query.CommittedInstanceID()].categoriesHi)) {
            return true;
        }
        const float t = query.CommittedRayT();
        ray.TMin = t + max(1.0e-4, t * 1.0e-5);
        if (ray.TMin >= ray.TMax) {
            return false;
        }
    }
    return false;
}

static const bool kTraces = true;
)";

const char* kNoRays = R"(
struct PathHit {
    uint4  seen;
    float  t;
    float2 barycentrics;
};

PathHit traceNearestFrom(float3 origin, float3 direction, float tMin, uint mask) {
    PathHit hit;
    hit.seen = uint4(0);
    hit.t = 0.0;
    hit.barycentrics = float2(0.0);
    return hit;
}

bool pathOccluded(float3 p, float3 n, float3 wi, float distance, uint shadowCategory, uint mask) {
    return false;
}

static const bool kTraces = false;
)";

/// The surface a hit lands on, its material evaluated, and the light it
/// gathers: shared by the camera's hit and the bounce's, so the two are shaded
/// by the same rules.
/// The light groups: declared only in the kernel of a frame that has them.
/// Their planes live in `sum` after the colour's -- the sums at planes
/// 1..groupCount, the means after those -- since a Metal kernel binds at
/// most 31 buffers and this one is at the limit: two more put it at
/// buffer(32), which the compiler in the process never returned from.
const char* kGroups = R"(
static const bool kLightGroups = true;
uniform uint                   groupCount;
// Eight registers and a select each, never an array indexed by the group
// nor a buffer written inside the bounce loop: either made the Metal
// compiler take longer than ten minutes over this kernel.
struct Groups {
    float3 g0; float3 g1; float3 g2; float3 g3; float3 g4; float3 g5; float3 g6; float3 g7;
};
Groups groupsZero() {
    Groups g;
    g.g0 = float3(0.0); g.g1 = float3(0.0); g.g2 = float3(0.0); g.g3 = float3(0.0);
    g.g4 = float3(0.0); g.g5 = float3(0.0); g.g6 = float3(0.0); g.g7 = float3(0.0);
    return g;
}
void addGroup(inout Groups g, uint group, float3 c) {
    g.g0 += group == 1u ? c : float3(0.0);
    g.g1 += group == 2u ? c : float3(0.0);
    g.g2 += group == 3u ? c : float3(0.0);
    g.g3 += group == 4u ? c : float3(0.0);
    g.g4 += group == 5u ? c : float3(0.0);
    g.g5 += group == 6u ? c : float3(0.0);
    g.g6 += group == 7u ? c : float3(0.0);
    g.g7 += group == 8u ? c : float3(0.0);
}
float3 groupOf(Groups g, uint k) {
    return k == 0u ? g.g0 : k == 1u ? g.g1 : k == 2u ? g.g2 : k == 3u ? g.g3
         : k == 4u ? g.g4 : k == 5u ? g.g5 : k == 6u ? g.g6 : g.g7;
}
void endGroups(uint at, uint pixels, Groups g, float alpha, float totalSamples) {
    for (uint k = 0; k < groupCount && k < 8; ++k) {
        const uint slot = (1u + k) * pixels + at;
        const float4 kept = path.accumulated != 0 ? sum[slot] : float4(0.0);
        const float4 now = kept + float4(groupOf(g, k), alpha);
        sum[slot] = now;
        sum[(1u + groupCount + k) * pixels + at] = totalSamples > 0.0 ? now / totalSamples : float4(0.0);
    }
}
)";

const char* kNoGroups = R"(
static const bool kLightGroups = false;
struct Groups { float3 g0; };
Groups groupsZero() { Groups g; g.g0 = float3(0.0); return g; }
void addGroup(inout Groups g, uint group, float3 c) {}
void endGroups(uint at, uint pixels, Groups g, float alpha, float totalSamples) {}
)";

/// The volumes: the medium module and the frame's words, in the kernel of
/// a frame that has volumes; stubs that scatter nothing otherwise, so a
/// frame without compiles exactly the kernel it always did.
const char* kVolumes = R"(
static const bool kVolumes = true;
StructuredBuffer<uint> volumeWords;   // world::VolumeSet's words: header, records, grids, leaf maxima
/// The first real collision along the ray among the frame's volumes before
/// tMax, if any: each volume's own free flight, the nearest of them.
bool mediumScatterAny(float3 origin, float3 direction, float tMin, float tMax, inout uint rng, out float t,
                      out float3 albedo, out float g) {
    t = tMax;
    albedo = float3(1.0);
    g = 0.0;
    bool any = false;
    const uint count = volumeCount(volumeWords);
    for (uint v = 0; v < count; ++v) {
        const VolumeRecord r = volumeRecord(volumeWords, v);
        float tv;
        if (mediumScatter(volumeWords, r, origin, direction, tMin, t, rng, tv) && tv < t) {
            t = tv;
            albedo = r.albedo;
            g = r.g;
            any = true;
        }
    }
    return any;
}
float mediumTransmittanceAny(float3 origin, float3 direction, float tMin, float tMax, inout uint rng) {
    float transmittance = 1.0;
    const uint count = volumeCount(volumeWords);
    for (uint v = 0; v < count; ++v) {
        transmittance *= mediumTransmittance(volumeWords, volumeRecord(volumeWords, v), origin, direction, tMin, tMax, rng);
    }
    return transmittance;
}
)";

const char* kNoVolumes = R"(
static const bool kVolumes = false;
bool mediumScatterAny(float3 origin, float3 direction, float tMin, float tMax, inout uint rng, out float t,
                      out float3 albedo, out float g) {
    t = tMax;
    albedo = float3(1.0);
    g = 0.0;
    return false;
}
float mediumTransmittanceAny(float3 origin, float3 direction, float tMin, float tMax, inout uint rng) { return 1.0; }
float mediumRandom(inout uint state) { return 0.0; }
float hgPhase(float cosTheta, float g) { return 0.0; }
float3 hgSample(float3 wo, float g, float2 u) { return -wo; }
)";

const char* kBody = R"(
struct Shaded {
    LobeStack      stack;
    MaterialInputs inputs;
    float3         toEye;
    float          depth;
    uint           categoriesLo;   // the instance's, for a light's link
    uint           categoriesHi;
    bool           valid;
    float3         rayOrigin;      // where the ray that found it started, and how far it went: a medium's segment
    float          rayT;
};

/// Evaluates whatever material a surface wears. `pixel` is only the uv
/// footprint's: materialInputsAt takes it from the neighbouring pixels' rays,
/// which is right for the camera's hit and an approximation for a bounce's.
Shaded shadeSurface(uint2 pixel, Surface s) {
    Shaded out;
    out.categoriesLo = s.instance.categoriesLo;
    out.categoriesHi = s.instance.categoriesHi;
    out.inputs = materialInputsAt(camera, toWorld, pixel.x, pixel.y, s, lookup.time);
    const MaterialRecord m = materials[materialRowOf(s)];
    evaluateMaterial(m.function, out.inputs, m.blob);
    out.stack = gLrtResult;
    out.toEye = normalize(out.inputs.viewPosition - out.inputs.positionWorld);
    out.depth = s.depth;
    out.valid = true;
    out.rayOrigin = out.inputs.viewPosition;
    out.rayT = length(out.inputs.positionWorld - out.inputs.viewPosition);
    return out;
}

/// A surface a ray found, before its material is evaluated. Finding and
/// shading are apart so the kernel evaluates materials at one place only:
/// every call site of the material dispatch is a copy of every material once
/// the Metal compiler inlines it, and with the camera's hit, a lens sample's,
/// a bounce's and a medium bounce's each shading, a stage of fifteen
/// materials (the chess set) ran the compiler service out after four minutes
/// where one site builds the pipeline in six seconds.
struct Found {
    Surface s;
    bool    valid;
    float3  positionWorld;
    float3  eye;           // where the ray came from: the camera's position, or the bounce's
    float3  rayOrigin;     // where the ray that found it started, and how far it went: a medium's segment
    float   rayT;
    float   depth;
};

Found foundNothing() {
    Found f;
    f.valid = false;
    f.depth = 0.0;
    f.rayT = 0.0;
    return f;
}

/// The camera's hit, rebuilt from the visibility pair as the raster shading
/// does, so the two shade the same point.
Found foundAt(uint2 pixel, uint4 seen) {
    if (seen.x == 0) {
        return foundNothing();
    }
    Found f;
    f.s = surfaceAt(camera, pixel.x, pixel.y, seen);
    f.valid = true;
    f.positionWorld = applyRows(toWorld, f.s.viewPosition, 1.0);
    f.eye = applyRows(toWorld, float3(0.0), 1.0);
    f.rayOrigin = f.eye;
    f.rayT = length(f.positionWorld - f.eye);
    f.depth = f.s.depth;
    return f;
}

/// A surface away from the camera: the same reconstruction, but the eye is
/// wherever the ray came from.
/// A bounce's hit, rebuilt from where the ray met the triangle -- not from
/// the pixel. It used to re-intersect the *camera's* ray with the triangle
/// the bounce found: a point that is not on the bounce ray at all, with
/// barycentrics and a normal to match. The closed box read 5 pi times its
/// geometric series while that stood.
Found foundHit(PathHit hit, float3 from, float3 direction) {
    if (hit.seen.x == 0) {
        return foundNothing();
    }
    // The bounce's direction in view space, for which side of the surface it
    // arrives at: toWorld's rows are view-to-world, their transpose takes a
    // direction back -- the same rigid reading of those rows materialInputsAt
    // makes when it turns a normal to world with them.
    const float3 d = direction;
    const float3 viewDirection = float3(toWorld.row0.x * d.x + toWorld.row1.x * d.y + toWorld.row2.x * d.z,
                                        toWorld.row0.y * d.x + toWorld.row1.y * d.y + toWorld.row2.y * d.z,
                                        toWorld.row0.z * d.x + toWorld.row1.z * d.y + toWorld.row2.z * d.z);
    const float3 weights = float3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x,
                                  hit.barycentrics.y);
    Found f;
    f.s = surfaceFromWeights(hit.seen, weights, viewDirection);
    f.valid = true;
    f.positionWorld = applyRows(toWorld, f.s.viewPosition, 1.0);
    f.eye = from;
    f.rayOrigin = from;
    f.rayT = hit.t;
    f.depth = f.s.depth;
    return f;
}

/// What was found, its material evaluated: the kernel's one call of it.
Shaded shadeFound(uint2 pixel, Found f) {
    Shaded out;
    out.valid = false;
    out.depth = 0.0;
    if (!f.valid) {
        return out;
    }
    out = shadeSurface(pixel, f.s);
    out.toEye = normalize(f.eye - out.inputs.positionWorld);
    out.rayOrigin = f.rayOrigin;
    out.rayT = f.rayT;
    return out;
}

/// The camera's hit found by a ray of the tracer's own, for a lens with a
/// diaphragm or a distorting one: neither passes through the pixel's centre,
/// so the visibility buffer's hit is not this sample's. The pixel's ray is
/// distorted radially in ndc, then a thin lens bends it: every ray through the
/// pixel meets the pixel's ray at the depth in focus, and leaves the lens from
/// a point drawn uniformly on its disk. Rigid rows take it to world.
/// The shutter slice a sample's rays answer to: every ray of one path is
/// traced at one time.
uint sampleMask(uint2 pixel, uint sample) {
    if (path.buckets <= 1) {
        return 0xFF;
    }
    const float t = random(pixel, sample, 0u, 17u);
    return 1u << min(uint(t * float(path.buckets)), path.buckets - 1);
}

/// View to world for a sample's rays: the frame's, or -- when the camera
/// moves under the shutter -- the camera's between its two samples, at the
/// centre of the shutter slice the sample's rays answer to (sampleMask draws
/// the same number), which is when the moving geometry it meets is drawn.
ViewToWorld cameraFor(uint2 pixel, uint sample) {
    if (path.cameraMoves == 0) {
        ViewToWorld frame;
        frame.row0 = toWorld.row0;
        frame.row1 = toWorld.row1;
        frame.row2 = toWorld.row2;
        return frame;
    }
    const float u = random(pixel, sample, 0u, 17u);
    const uint buckets = max(path.buckets, 1u);
    const uint b = min(uint(u * float(buckets)), buckets - 1);
    const float at = path.shutterOpen + (float(b) + 0.5) / float(buckets) * (path.shutterClose - path.shutterOpen);
    const float span = path.cameraTime1 - path.cameraTime0;
    const float t = abs(span) > 1.0e-12 ? (at - path.cameraTime0) / span : 0.0;
    ViewToWorld m;
    m.row0 = lerp(path.cameraStart0, path.cameraEnd0, t);
    m.row1 = lerp(path.cameraStart1, path.cameraEnd1, t);
    m.row2 = lerp(path.cameraStart2, path.cameraEnd2, t);
    return m;
}

Found foundLensSample(uint2 pixel, uint sample, uint mask) {
    float3 origin;
    float3 direction;
    viewRay(camera, float2(pixel) + 0.5, origin, direction);
    if (camera.orthographic == 0) {
        const float2 ndc = float2(direction.x * camera.focalX / (0.5 * float(camera.width)),
                                  direction.y * camera.focalY / (0.5 * float(camera.height)));
        const float r2 = dot(ndc, ndc);
        direction.xy *= 1.0 + path.distortionK1 * r2 + path.distortionK2 * r2 * r2;
        if (path.lensRadius > 0.0) {
            const float3 focus = direction * max(path.focusDistance, camera.nearZ);
            // Uniform on the lens disk, by area.
            const float2 u = random2(pixel, sample, 0u, 13u);
            const float r = sqrt(u.x) * path.lensRadius;
            const float phi = 2.0 * 3.14159265358979 * u.y;
            origin = float3(r * cos(phi), r * sin(phi), 0.0);
            direction = focus - origin;
        }
    }
    direction = normalize(direction);
    const ViewToWorld eye = cameraFor(pixel, sample);
    const float3 originWorld = float3(dot(eye.row0.xyz, origin) + eye.row0.w,
                                      dot(eye.row1.xyz, origin) + eye.row1.w,
                                      dot(eye.row2.xyz, origin) + eye.row2.w);
    const float3 directionWorld = normalize(float3(dot(eye.row0.xyz, direction),
                                                   dot(eye.row1.xyz, direction),
                                                   dot(eye.row2.xyz, direction)));
    const PathHit hit = traceNearestFrom(originWorld, directionWorld, camera.nearZ, mask);
    return foundHit(hit, originWorld, directionWorld);
}

/// One light, sampled and weighed: next event estimation, with the density of
/// the material's own sampling folded in by the power heuristic so the two
/// strategies do not double count.
/// A medium's random state for one purpose of one bounce of one sample.
uint mediumSeed(uint2 pixel, uint sample, uint bounce, uint dim) {
    return asuint(random(pixel, sample, bounce, dim)) ^ 0x5bd1e995u;
}

/// The camera's ray through a pixel, in world space, lens and distortion
/// included when the tracer casts its own: for a sample that found no
/// surface, along which a volume may still lie.
void primaryRay(uint2 pixel, uint sample, out float3 originWorld, out float3 directionWorld) {
    float3 origin;
    float3 direction;
    viewRay(camera, float2(pixel) + 0.5, origin, direction);
    if (camera.orthographic == 0 && path.ownRays != 0) {
        const float2 ndc = float2(direction.x * camera.focalX / (0.5 * float(camera.width)),
                                  direction.y * camera.focalY / (0.5 * float(camera.height)));
        const float r2 = dot(ndc, ndc);
        direction.xy *= 1.0 + path.distortionK1 * r2 + path.distortionK2 * r2 * r2;
        if (path.lensRadius > 0.0) {
            const float3 focus = direction * max(path.focusDistance, camera.nearZ);
            const float2 u = random2(pixel, sample, 0u, 13u);
            const float r = sqrt(u.x) * path.lensRadius;
            const float phi = 2.0 * 3.14159265358979 * u.y;
            origin = float3(r * cos(phi), r * sin(phi), 0.0);
            direction = focus - origin;
        }
    }
    direction = normalize(direction);
    const ViewToWorld eye = cameraFor(pixel, sample);   // the frame's unless the camera moves
    originWorld = float3(dot(eye.row0.xyz, origin) + eye.row0.w, dot(eye.row1.xyz, origin) + eye.row1.w,
                         dot(eye.row2.xyz, origin) + eye.row2.w);
    directionWorld = normalize(float3(dot(eye.row0.xyz, direction), dot(eye.row1.xyz, direction),
                                      dot(eye.row2.xyz, direction)));
}

/// Light k's record for a sample: as the table holds it, or -- when the
/// light moves under the shutter -- placed between its two samples at the
/// centre of the shutter slice the sample's rays answer to, as the camera
/// and the moving geometry are.
LightRecord lightFor(uint k, uint2 pixel, uint sample) {
    LightRecord l = lights[k];
    if (lightMotionBase == 0) {
        return l;
    }
    const uint at = lightMotionBase + k * 26;
    const float t0 = iesValues[at + 24];
    const float t1 = iesValues[at + 25];
    if (!(t1 > t0)) {
        return l;
    }
    const float u = random(pixel, sample, 0u, 17u);
    const uint buckets = max(path.buckets, 1u);
    const uint b = min(uint(u * float(buckets)), buckets - 1);
    const float centre = path.shutterOpen + (float(b) + 0.5) / float(buckets) * (path.shutterClose - path.shutterOpen);
    const float f = (centre - t0) / (t1 - t0);
    const float4 s0 = float4(iesValues[at], iesValues[at + 1], iesValues[at + 2], iesValues[at + 3]);
    const float4 s1 = float4(iesValues[at + 4], iesValues[at + 5], iesValues[at + 6], iesValues[at + 7]);
    const float4 s2 = float4(iesValues[at + 8], iesValues[at + 9], iesValues[at + 10], iesValues[at + 11]);
    const float4 e0 = float4(iesValues[at + 12], iesValues[at + 13], iesValues[at + 14], iesValues[at + 15]);
    const float4 e1 = float4(iesValues[at + 16], iesValues[at + 17], iesValues[at + 18], iesValues[at + 19]);
    const float4 e2 = float4(iesValues[at + 20], iesValues[at + 21], iesValues[at + 22], iesValues[at + 23]);
    l.row0 = lerp(s0, e0, f);
    l.row1 = lerp(s1, e1, f);
    l.row2 = lerp(s2, e2, f);
    return l;
}

/// Light k's probability of being chosen at p, n, as gatherLight chooses.
float lightChoiceProbability(uint k, float3 p, float3 n) {
    if (path.chooseLights == 2) {
        return lightPdfChoiceAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount, lights, k, p, n);
    }
    const float power = lights[lightCount - 1].cumulative;
    if (!(power > 0.0)) {
        return 0.0;
    }
    const float before = k == 0 ? 0.0 : lights[k - 1].cumulative;
    return max(lights[k].cumulative - before, 1.0e-9) / power;
}

/// Lights past this many are not met by the material's rays: each bounce
/// tests its direction against every light, and a frame of thousands would
/// pay that per vertex. Their next event estimates keep a weight of one.
static const uint kMisLightLimit = 64;

/// Whether light l is reached by both strategies and weighed between them.
/// Not for a light whose shadow ignores what a ray would stop at: one that
/// casts none, or one whose shadow links leave some occluders out -- the
/// material's ray is stopped by any surface, so the two strategies would
/// see different visibilities. Not through media: a volume between would
/// have to dim the material's ray as it dims the shadow ray.
bool misWeighs(LightRecord l) {
    return path.mis != 0 && !kVolumes && lightCount <= kMisLightLimit && (l.flags & kLightShadow) != 0 &&
           l.shadowCategory == kLightUnlinked;
}

float3 gatherLight(Shaded sh, uint2 pixel, uint sample, uint bounce, uint mask, out uint group) {
    group = 0;
    if (lightCount == 0) {
        // A stage without lights is lit as the raster lights it, when the
        // engine asks: the headlight, unit radiance from the eye, at the
        // first vertex alone. Paths go on and find nothing more, so both
        // techniques draw one image of an unlit stage rather than rt drawing
        // it black. Not otherwise: a scene lit by its emission alone is lit
        // by that.
        return bounce == 0 && path.headlight != 0 ? kPi * stackEval(sh.stack, sh.toEye, sh.toEye) : float3(0.0);
    }
    const float pick = random(pixel, sample, bounce, 11u);
    const LightChoice choice = path.chooseLights == 2
                                   ? chooseLightAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount,
                                                    sh.inputs.positionWorld, sh.inputs.normalWorld, pick)
                                   : chooseLight(lights, lightCount, pick);
    if (!choice.valid) {
        return float3(0.0);
    }
    const LightRecord light = lightFor(choice.index, pixel, sample);
    group = light.group;
    if (!lightLinked(light.lightCategory, sh.categoriesLo, sh.categoriesHi)) {
        return float3(0.0);
    }
    const LightSample ls =
        sampleLightImaged(light, sh.inputs.positionWorld, sh.inputs.normalWorld, random2(pixel, sample, bounce, 3u));
    if (!ls.valid) {
        return float3(0.0);
    }
    const float3 f = stackEval(sh.stack, sh.toEye, ls.wi);
    if (!any(f > float3(0.0))) {
        return float3(0.0);
    }
    if ((light.flags & kLightShadow) != 0 &&
        pathOccluded(sh.inputs.positionWorld, sh.inputs.normalWorld, ls.wi, ls.distance, light.shadowCategory,
                     mask)) {
        return float3(0.0);
    }
    // Through whatever medium lies between: a transmittance of one without,
    // and for a light that casts no shadows -- a volume is an occluder like
    // any other, and `shadow:enable` off means none dims it.
    float transmittance = 1.0;
    if (kVolumes && (light.flags & kLightShadow) != 0) {
        uint rng = mediumSeed(pixel, sample, bounce, 21u);
        transmittance = mediumTransmittanceAny(sh.inputs.positionWorld, ls.wi, 1.0e-3 * max(1.0, length(sh.inputs.positionWorld)),
                                               ls.distance, rng);
    }
    const float density = ls.pdf * choice.probability;
    // Weighed against the material's strategy by the power heuristic where
    // that strategy reaches this light too: at a vertex the path leaves by
    // sampling its material, whose ray gathers what it meets of the lights
    // (lightHit) with the complementary weight. A delta light, a light the
    // material's ray does not weigh (misWeighs), and the last vertex, which
    // samples no material, keep a weight of one. The weight was once taken
    // with no complement at all, and a white furnace under an imageless dome
    // -- the dome's density and the Lambert lobe's the same function -- read
    // 0.5453 of what it should; the complement is what makes it whole.
    float weight = 1.0;
    if (!ls.delta && kTraces && bounce < path.bounces && misWeighs(light)) {
        const float other = stackPdf(sh.stack, sh.toEye, ls.wi);
        weight = density * density / (density * density + other * other);
    }
    return f * ls.radiance * transmittance * weight / density;
}

/// The same at a point inside a medium: the phase function for the lobes,
/// no surface to offset from, the lights chosen by power.
float3 gatherLightMedium(float3 p, float3 wo, float g, uint2 pixel, uint sample, uint bounce, uint mask,
                         out uint group) {
    group = 0;
    if (lightCount == 0) {
        return float3(0.0);
    }
    const LightChoice choice = chooseLight(lights, lightCount, random(pixel, sample, bounce, 11u));
    if (!choice.valid) {
        return float3(0.0);
    }
    const LightRecord light = lightFor(choice.index, pixel, sample);
    group = light.group;
    LightSample ls;
    if (light.kind == kLightDome && !domeHasImage(light)) {
        // A point in a medium faces no hemisphere: the dome's cosine
        // sampling about a normal would never draw the half behind it, and
        // flipping between the two halves weighs a sample near their
        // horizon by one over its cosine. Uniform over the sphere instead,
        // which for an isotropic phase weighs every sample the same.
        const float2 u = random2(pixel, sample, bounce, 3u);
        const float z = 1.0 - 2.0 * u.x;
        const float r = sqrt(max(0.0, 1.0 - z * z));
        const float phi = 2.0 * 3.14159265358979 * u.y;
        ls.wi = float3(r * cos(phi), r * sin(phi), z);
        ls.distance = 1.0e30;
        ls.pdf = 1.0 / (4.0 * 3.14159265358979);
        ls.radiance = lightEmission(light) * domeImage(light, ls.wi);
        ls.delta = false;
        ls.valid = true;
    } else {
        ls = sampleLightImaged(light, p, wo, random2(pixel, sample, bounce, 3u));
    }
    if (!ls.valid) {
        return float3(0.0);
    }
    const float f = hgPhase(dot(wo, ls.wi), g);
    if (f <= 0.0) {
        return float3(0.0);
    }
    if ((light.flags & kLightShadow) != 0 && pathOccluded(p, ls.wi, ls.wi, ls.distance, light.shadowCategory, mask)) {
        return float3(0.0);
    }
    float transmittance = 1.0;
    if ((light.flags & kLightShadow) != 0) {
        uint rng = mediumSeed(pixel, sample, bounce, 25u);
        transmittance = mediumTransmittanceAny(p, ls.wi, 1.0e-4 * max(1.0, length(p)), ls.distance, rng);
    }
    return f * ls.radiance * transmittance / (ls.pdf * choice.probability);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void tracePaths(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    const uint2 tid = lrtQuadPixel(group.xy, index);
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    const uint pixels = camera.width * camera.height;
    if (path.accumulated == 0) {
        moments[pixels + at] = 0;   // a frame of its own: no flag from the last one
    } else if (path.adaptive != 0 && moments[pixels + at] != 0) {
        return;   // converged: its mean stands
    }
    const uint samples = max(path.samples, 1u);
    // The camera's hit does not depend on the sample: rebuilt once, and its
    // material evaluated the first time a path reaches it, then kept, not
    // evaluated once a path. What the denoiser wants of it is written then,
    // since it is the same for every sample. With a lens it does depend on
    // the sample, and each path casts its own; the aux then carry the first
    // hit a path shaded.
    const bool ownRays = kTraces && path.ownRays != 0;
    const Found firstFound = ownRays ? foundLensSample(tid, 0u, sampleMask(tid, 0u)) : foundAt(tid, seen);
    Shaded first;
    first.valid = false;
    first.depth = 0.0;
    bool firstShaded = false;
    bool auxWritten = false;
    if (path.writeAux != 0) {
        aux[at] = float4(0.0);
        aux[pixels + at] = float4(0.0);
    }
    float3 total = float3(0.0);
    float  alpha = 0.0;
    float  hitDepth = 0.0;
    float  squares = 0.0;   // sum of each sample's luminance squared
    uint group = 0;
    Groups groups = groupsZero();
    for (uint sample = 0; sample < samples && (firstFound.valid || ownRays || kVolumes); ++sample) {
        const uint mask = sampleMask(tid, sample);
        Found found = ownRays && sample > 0 ? foundLensSample(tid, sample, mask) : firstFound;
        // The ray to the first vertex: from the surface it found, or the
        // pixel's ray to nothing -- along which a volume may still lie.
        float3 o;
        float3 d;
        float tHit;
        if (found.valid) {
            o = found.rayOrigin;
            d = normalize(found.positionWorld - found.rayOrigin);
            tHit = found.rayT;
        } else if (kVolumes) {
            primaryRay(tid, sample, o, d);
            tHit = 1.0e30;
        } else {
            continue;   // a lens ray that found nothing: transparent, and counted
        }
        uint rng = mediumSeed(tid, sample, 0u, 23u);
        float3 carried = float3(0.0);
        float3 throughput = float3(1.0);
        // The first vertex's opacity is the pixel's, and its depth; a
        // medium's collision is opaque.
        float  opacity = 0.0;
        float  depthHere = 0.0;
        bool   vertexSeen = false;
        // The vertices, each either a surface the ray met or a collision in
        // a medium before it: each gathers its light through what the path
        // has kept, and sends the path on -- through a lobe or the phase.
        for (uint bounce = 0; bounce <= path.bounces; ++bounce) {
            float tS;
            float3 albedo;
            float g;
            const float tMin = bounce == 0 && found.valid && !ownRays ? camera.nearZ : 0.0;
            if (kVolumes && mediumScatterAny(o, d, tMin, tHit, rng, tS, albedo, g)) {
                const float3 p = o + d * tS;
                if (!vertexSeen) {
                    vertexSeen = true;
                    opacity = 1.0;
                    depthHere = found.valid ? found.depth : tS;
                }
                throughput *= albedo;
                const float3 direct = gatherLightMedium(p, -d, g, tid, sample, bounce, mask, group);
                carried += throughput * direct;
                if (kLightGroups) {
                    addGroup(groups, group, throughput * direct * opacity);
                }
                if (bounce == path.bounces || !kTraces) {
                    break;
                }
                d = hgSample(-d, g, float2(mediumRandom(rng), mediumRandom(rng)));
                o = p;
                const PathHit hit = traceNearestFrom(o, d, 1.0e-4 * max(1.0, length(o)), mask);
                found = foundHit(hit, o, d);
                tHit = hit.seen.x == 0 ? 1.0e30 : hit.t;
                continue;
            }
            if (!found.valid) {
                break;   // the ray escaped
            }
            // The one place materials are evaluated (Found says why); the
            // camera's hit once a pixel.
            Shaded cur;
            if (bounce == 0 && !ownRays && firstShaded) {
                cur = first;
            } else {
                cur = shadeFound(tid, found);
                if (bounce == 0 && !ownRays) {
                    first = cur;
                    firstShaded = true;
                }
            }
            if (bounce == 0 && path.writeAux != 0 && !auxWritten) {
                aux[at] = float4(stackAlbedo(cur.stack, cur.toEye), 1.0);
                aux[pixels + at] = float4(cur.inputs.normalWorld, 1.0);
                auxWritten = true;
            }
            if (!vertexSeen) {
                vertexSeen = true;
                opacity = cur.stack.opacity;
                depthHere = cur.depth;
            }
            const float3 direct = gatherLight(cur, tid, sample, bounce, mask, group);
            carried += throughput * (cur.stack.emission + direct);
            if (kLightGroups) {
                addGroup(groups, group, throughput * direct * opacity);
            }
            if (bounce == path.bounces || !kTraces) {
                break;
            }
            const LobeSample ms = stackSample(cur.stack, cur.toEye, float3(random2(tid, sample, bounce, 5u),
                                                                          random(tid, sample, bounce, 7u)));
            if (!ms.valid || ms.pdf <= 0.0) {
                break;
            }
            // LobeSample.weight is already f |cos| / pdf (lobes.slang); dividing
            // by the pdf again turned rho into rho pi / cos, and the closed box
            // read 5 pi times its geometric series.
            throughput *= ms.weight;
            if (!any(throughput > float3(0.0))) {
                break;
            }
            const float3 p = cur.inputs.positionWorld;
            const float3 n = cur.inputs.normalWorld;
            const float scale = max(1.0, length(p));
            const float3 away = dot(n, ms.wi) < 0.0 ? -n : n;
            o = p + (away + ms.wi) * (1.0e-3 * scale);
            d = ms.wi;
            const PathHit hit = traceNearestFrom(o, d, 1.0e-3 * scale, mask);
            // The material's strategy for the lights: what of each light this
            // direction meets before the surface it found (or, escaping, a
            // dome's or a distant light's), weighed against the density next
            // event estimation would have drawn it with.
            if (path.mis != 0 && !kVolumes && lightCount <= kMisLightLimit) {
                const bool escaped = hit.seen.x == 0;
                const float reached = hit.t + dot(o - p, d);
                for (uint k = 0; k < lightCount; ++k) {
                    const LightRecord light = lightFor(k, tid, sample);
                    if (!misWeighs(light) || !lightLinked(light.lightCategory, cur.categoriesLo, cur.categoriesHi)) {
                        continue;
                    }
                    const LightHit lh = lightHitImaged(light, p, d);
                    // A dome or a distant light (t infinite) only where the ray
                    // escapes; a light at a distance, if nothing is before it.
                    // (Comparing an escape's infinity with the light's own once
                    // dropped every dome here: 1e30 is not less than 1e30.)
                    const bool infinite = lh.t >= 1.0e29;
                    if (!lh.valid || (infinite && !escaped) || (!infinite && !escaped && lh.t >= reached)) {
                        continue;
                    }
                    float weight = 1.0;
                    if (!ms.delta) {
                        const float other = lightChoiceProbability(k, p, n) * lightPdfImaged(light, p, n, d);
                        weight = ms.pdf * ms.pdf / (ms.pdf * ms.pdf + other * other);
                    }
                    const float3 reachedLight = throughput * lh.radiance * weight;
                    carried += reachedLight;
                    if (kLightGroups) {
                        addGroup(groups, light.group, reachedLight * opacity);
                    }
                }
            }
            // Rebuilt from where the ray met the triangle, its eye where the
            // bounce left from; shaded when the next vertex comes to it.
            found = foundHit(hit, p, d);
            tHit = hit.seen.x == 0 ? 1.0e30 : hit.t;
        }
        if (!vertexSeen) {
            continue;   // nothing along the ray at all: transparent, and counted
        }
        hitDepth = depthHere;
        alpha += opacity;
        total += carried * opacity;
        const float lum = dot(carried * opacity, kPathLuminance);
        squares += lum * lum;
    }
    // A mean of products, not a product of means: `total` already carries each
    // sample's colour times that sample's opacity, so the sum is the sum. The
    // two are the same while every sample is opaque -- which is every test
    // there is today -- and differ as soon as opacity varies per sample, and
    // the product form also broke the invariant that 256 paths in one pass
    // must equal 64 in each of four.
    const float4 added = float4(total, alpha) / float(samples);
    const float before = float(path.accumulated);
    const float4 kept = path.accumulated != 0 ? sum[at] : float4(0.0);
    const float4 now = kept + added * float(samples);
    sum[at] = now;
    const float total_samples = before + float(samples);
    colour[at] = total_samples > 0.0 ? now / total_samples : float4(0.0);
    depth[at] = hitDepth;
    endGroups(at, pixels, groups, alpha, total_samples);
    // The luminance's second moment. Whether the pixel stops is decided by
    // pathDecide, after the pass, where its neighbours' spread can be read
    // without a race.
    const float keptSquares = path.accumulated != 0 ? asfloat(moments[at]) : 0.0;
    moments[at] = asuint(keptSquares + squares);
}

// The adaptive stop rule, one thread a pixel, after a pass. A pixel stops when
// the relative standard error of its mean has fallen below the target after
// minSamples paths -- with its variance taken as the larger of its own and
// the mean of its 3x3 neighbours'. A pixel's own sample variance cannot see
// an event that has not happened to it yet: under a rare bright bounce the
// first N paths may all miss it, and the spread they show is the direct
// light's alone, a hundred times too small. A neighbour that did see it
// stands in. Measured without this, at min 16: 526 of 4212 stopped pixels
// beyond three of their own sigma, worst 133 sigma, all of them below the
// reference.
RWStructuredBuffer<uint> progress;   // [covered, converged], by atomics; cleared by the host

float pixelVariance(uint at, out float mean, out float n) {
    n = max(sum[at].w, 1.0);
    mean = dot(sum[at].rgb, kPathLuminance) / n;
    const float second = asfloat(moments[at]) / n;
    return max(second - mean * mean, 0.0) / n;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pathDecide(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= camera.width * camera.height) {
        return;
    }
    const uint x = at % camera.width;
    const uint y = at / camera.width;
    const uint4 seen = visibility.Load(int3(int(x), int(camera.height - 1 - y), 0));
    const uint pixels = camera.width * camera.height;
    if (seen.x == 0) {
        moments[pixels + at] = 1;   // nothing drawn converges at once, and is not counted
        return;
    }
    InterlockedAdd(progress[0], 1u);
    if (moments[pixels + at] != 0) {
        InterlockedAdd(progress[1], 1u);
        return;
    }
    float mean;
    float n;
    const float own = pixelVariance(at, mean, n);
    if (n < float(path.minSamples)) {
        return;
    }
    float around = 0.0;
    uint count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = int(x) + dx;
            const int ny = int(y) + dy;
            if (nx < 0 || ny < 0 || nx >= int(camera.width) || ny >= int(camera.height)) {
                continue;
            }
            const uint4 nseen = visibility.Load(int3(nx, int(camera.height) - 1 - ny, 0));
            if (nseen.x == 0) {
                continue;
            }
            float nmean;
            float nn;
            // The neighbour's per-sample spread, scaled to this pixel's count.
            around += pixelVariance(uint(ny) * camera.width + uint(nx), nmean, nn) * nn / n;
            ++count;
        }
    }
    const float variance = max(own, count > 0 ? around / float(count) : 0.0);
    const float relative = sqrt(variance) / max(mean, 1.0e-3);
    if (relative < path.errorTarget) {
        moments[pixels + at] = 1;
        InterlockedAdd(progress[1], 1u);
    }
}

)";

}   // namespace

Result<PathTracer> PathTracer::create(gpu::ShaderLibrary& library) {
    PathTracer tracer;
    tracer.library_ = &library;
    tracer.device_ = &library.device();
    return tracer;
}

Result<void> PathTracer::setPrograms(const MaterialPrograms& programs) {
    return setPrograms(programs, groups_, volumes_);
}

Result<void> PathTracer::setPrograms(const MaterialPrograms& programs, bool groups, bool volumes) {
    if (programs.module() == module_ && groups == groups_ && volumes == volumes_ && kernel_.has_value()) {
        return ok();
    }
    const bool traces = device_->caps().rayQuery && device_->caps().accelerationStructure;
    const std::string name = programs.module() + (traces ? "_path_traced" : "_path_direct") +
                             (groups ? "_groups" : "") + (volumes ? "_volumes" : "");
    const std::string source = "import " + programs.module() + ";\n" +
                               (volumes ? std::string("import lrt.volume.medium;\n") : std::string()) + kPrelude +
                               (groups ? kGroups : kNoGroups) + (volumes ? kVolumes : kNoVolumes) +
                               (traces ? kRays : kNoRays) + kBody;
    auto program = library_->loadSource(name, source, {"tracePaths", "pathDecide"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "tracePaths");
    if (!kernel) return std::move(kernel).error();
    kernel_.emplace(std::move(*kernel));
    auto progressKernel = gpu::ComputeKernel::create(*library_, name, "pathDecide");
    if (!progressKernel) return std::move(progressKernel).error();
    progressKernel_.emplace(std::move(*progressKernel));
    module_ = programs.module();
    groups_ = groups;
    volumes_ = volumes;
    accumulated_ = 0;
    return ok();
}

Result<void> PathTracer::trace(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                               const render::Projection& projection, const MaterialFrame& frame,
                               const PathSettings& settings, render::RenderTargets& out, PathAux* aux) {
    if (!kernel_.has_value()) {
        return Error(ErrorCode::InvalidArgument, "path tracer: no materials set");
    }
    const bool groups = frame.groups.count > 0;
    const bool volumes = frame.volumes != nullptr && frame.volumeCount > 0 && frame.volumes->valid();
    if ((groups != groups_ || volumes != volumes_) && frame.programs != nullptr) {
        LRT_TRY(setPrograms(*frame.programs, groups, volumes));
    }
    const uint32_t groupCount = std::min(frame.groups.count, kMaxLightGroups);
    const uint64_t pixels = uint64_t{targets.width} * targets.height;
    const bool resized = out.width != targets.width || out.height != targets.height || !out.colour.valid();
    if (resized) {
        gpu::BufferDesc colour;
        colour.bytes = pixels * 16;
        colour.elementBytes = 16;
        colour.label = "path.colour";
        auto madeColour = gpu::Buffer::create(*device_, colour);
        if (!madeColour) return std::move(madeColour).error();
        gpu::BufferDesc depth;
        depth.bytes = pixels * 4;
        depth.elementBytes = 4;
        depth.label = "path.depth";
        auto madeDepth = gpu::Buffer::create(*device_, depth);
        if (!madeDepth) return std::move(madeDepth).error();
        out.colour = std::move(*madeColour);
        out.depth = std::move(*madeDepth);
        out.width = targets.width;
        out.height = targets.height;
    }
    // The sum holds the colour's plane and, with light groups, their sums
    // and their means after it.
    const uint32_t planes = 1 + 2 * groupCount;
    if (resized || width_ != targets.width || height_ != targets.height || !sum_.valid() || planes != sumPlanes_) {
        gpu::BufferDesc sum;
                sum.bytes = pixels * 16 * planes;
        sum.elementBytes = 16;
        sum.label = "path.sum";
        auto made = gpu::Buffer::create(*device_, sum);
        if (!made) return std::move(made).error();
        sum_ = std::move(*made);
        sumPlanes_ = planes;
        gpu::BufferDesc moments;
        moments.bytes = pixels * 4 * 2;
        moments.elementBytes = 4;
        moments.label = "path.moments";
        auto madeMoments = gpu::Buffer::create(*device_, moments);
        if (!madeMoments) return std::move(madeMoments).error();
        moments_ = std::move(*madeMoments);
        if (!progress_.valid()) {
            gpu::BufferDesc progress;
            progress.bytes = 2 * 4;
            progress.elementBytes = 4;
            progress.label = "path.progress";
            auto madeProgress = gpu::Buffer::create(*device_, progress);
            if (!madeProgress) return std::move(madeProgress).error();
            progress_ = std::move(*madeProgress);
        }
        width_ = targets.width;
        height_ = targets.height;
        accumulated_ = 0;
    }
    if (!settings.accumulate) {
        accumulated_ = 0;
    }
    if (aux != nullptr && (aux->width != targets.width || aux->height != targets.height || !aux->valid())) {
        gpu::BufferDesc desc;
        desc.bytes = pixels * 16 * 2;
        desc.elementBytes = 16;
        desc.label = "path.aux";
        auto planes = gpu::Buffer::create(*device_, desc);
        if (!planes) return std::move(planes).error();
        aux->planes = std::move(*planes);
        aux->width = targets.width;
        aux->height = targets.height;
    }
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    const uint32_t samples = std::max(settings.samples, 1u);
    const uint32_t already = accumulated_;
    kernel_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
        bindMaterialFrame(cursor, frame, projection);
        if (frame.lights != nullptr) {
            frame.lights->bind(cursor);
        }
        if (frame.shadows != nullptr) {
            cursor["scene"].setBinding(frame.shadows);
        }
        cursor["visibility"].setBinding((*ids).get());
        cursor["sum"].setBinding(sum_.rhi());
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        // A buffer has to be bound either way; without aux the colour stands in
        // and the kernel never writes it.
        cursor["aux"].setBinding(aux != nullptr ? aux->planes.rhi() : out.colour.rhi());
        cursor["path"]["writeAux"].setData(uint32_t{aux != nullptr ? 1u : 0u});
        cursor["moments"].setBinding(moments_.rhi());
        if (groups) {
            cursor["groupCount"].setData(groupCount);
        }
        if (volumes) {
            cursor["volumeWords"].setBinding(frame.volumes->rhi());
        }
        cursor["path"]["adaptive"].setData(uint32_t{settings.adaptive ? 1u : 0u});
        cursor["path"]["headlight"].setData(uint32_t{settings.headlight ? 1u : 0u});
        cursor["path"]["mis"].setData(uint32_t{settings.mis ? 1u : 0u});
        // A moving camera: its view to world at the two samples, for the rays.
        cursor["path"]["cameraMoves"].setData(uint32_t{projection.cameraMoves ? 1u : 0u});
        {
            const std::array<float, 12> start = projection.viewToWorldStart.rows3x4();
            const std::array<float, 12> end = projection.viewToWorldEnd.rows3x4();
            static constexpr const char* kStart[3] = {"cameraStart0", "cameraStart1", "cameraStart2"};
            static constexpr const char* kEnd[3] = {"cameraEnd0", "cameraEnd1", "cameraEnd2"};
            for (size_t r = 0; r < 3; ++r) {
                cursor["path"][kStart[r]].setData(start.data() + r * 4, sizeof(float) * 4);
                cursor["path"][kEnd[r]].setData(end.data() + r * 4, sizeof(float) * 4);
            }
            cursor["path"]["cameraTime0"].setData(static_cast<float>(projection.cameraTimeStart));
            cursor["path"]["cameraTime1"].setData(static_cast<float>(projection.cameraTimeEnd));
            cursor["path"]["shutterOpen"].setData(static_cast<float>(frame.scene != nullptr ? frame.scene->shutterOpen() : 0.0));
            cursor["path"]["shutterClose"].setData(static_cast<float>(frame.scene != nullptr ? frame.scene->shutterClose() : 1.0));
        }
        cursor["path"]["errorTarget"].setData(settings.errorTarget);
        cursor["path"]["minSamples"].setData(settings.minSamples);
        const uint32_t buckets = frame.scene != nullptr ? frame.scene->buckets() : 1u;
        const bool ownRays = (!projection.orthographic &&
                              (projection.lensRadius > 0.0 || projection.distortionK1 != 0.0 ||
                               projection.distortionK2 != 0.0)) ||
                             buckets > 1 || projection.cameraMoves;
        cursor["path"]["ownRays"].setData(uint32_t{ownRays ? 1u : 0u});
        cursor["path"]["buckets"].setData(buckets);
        cursor["path"]["lensRadius"].setData(static_cast<float>(projection.lensRadius));
        cursor["path"]["focusDistance"].setData(static_cast<float>(projection.focusDistance));
        cursor["path"]["distortionK1"].setData(static_cast<float>(projection.distortionK1));
        cursor["path"]["distortionK2"].setData(static_cast<float>(projection.distortionK2));
        setCamera(cursor["camera"], projection, targets.width, targets.height);
        cursor["path"]["samples"].setData(samples);
        cursor["path"]["bounces"].setData(settings.bounces);
        cursor["path"]["seed"].setData(settings.seed);
        cursor["path"]["accumulated"].setData(already);
        const bool tree = frame.chooseLights && frame.lightBvh && frame.lights != nullptr && frame.lights->hasBvh();
        cursor["path"]["chooseLights"].setData(uint32_t{tree ? 2u : frame.chooseLights ? 1u : 0u});
    });
    accumulated_ = already + samples;
    lastErrorTarget_ = settings.errorTarget;
    lastMinSamples_ = settings.minSamples;
    return ok();
}

Result<PathProgress> PathTracer::progress(const VisibilityTargets& targets) {
    if (!progressKernel_.has_value() || !moments_.valid()) {
        return Error(ErrorCode::InvalidArgument, "path tracer: nothing traced yet");
    }
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    const uint32_t zero[2] = {0, 0};
    LRT_TRY(progress_.write(*device_, 0, sizeof(zero), zero));
    gpu::CommandBatch batch(*device_);
    progressKernel_->dispatch(batch, {targets.width * targets.height, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["visibility"].setBinding((*ids).get());
        cursor["sum"].setBinding(sum_.rhi());
        cursor["moments"].setBinding(moments_.rhi());
        cursor["progress"].setBinding(progress_.rhi());
        setCamera(cursor["camera"], render::Projection{}, targets.width, targets.height);
        cursor["path"]["errorTarget"].setData(lastErrorTarget_);
        cursor["path"]["minSamples"].setData(lastMinSamples_);
    });
    LRT_TRY(batch.submit(true));
    uint32_t counts[2] = {0, 0};
    LRT_TRY(progress_.read(*device_, 0, sizeof(counts), counts));
    PathProgress out;
    out.covered = counts[0];
    out.converged = counts[1];
    return out;
}

}   // namespace lrt::technique
