// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/PathTracer.h"

#include "lrt/core/Log.h"
#include "lrt/technique/SplatShadows.h"

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
// The harmonics a bake projects onto, and the space a cloud is blended in.
import lrt.common.sh;
import lrt.common.color;

struct PathParams {
    uint samples;      // paths a pixel this call
    uint bounces;      // indirect bounces after the first hit
    uint seed;         // which samples these are
    uint accumulated;  // paths a pixel already in `sum`
    uint chooseLights;
    uint headlight;    // 1: with no lights, the first vertex is lit from the eye, as the raster lights it
    uint writeAux;     // 1: write the first hit's albedo and normal
    uint bakeCount;    // a bake: how many harmonics it fits, 1 (a colour) to 16
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
    // Where the packed splat tables start, in float4 entries, and when a
    // shadow ray through the cloud gives up. Here rather than in a constant
    // buffer of their own: this kernel binds 31 buffers on Metal, which is
    // all Metal gives, so a block of its own would not fit (kSplats).
    uint splatFrames;
    uint splatColours;
    uint splatInstances;
    uint splatIndices;
    uint splatInstanceCount;
    float splatCut;
};

Texture2D<uint4>               visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>     sum;          // paths added so far, a pixel
RWStructuredBuffer<float4>     colour;       // their mean
RWStructuredBuffer<float>      depth;
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

/// The same two questions where the device traces only in a pipeline: CUDA
/// has OptiX and Slang offers no inline `RayQuery` for that target, so the
/// nearest hit comes back in a payload from a closest hit program and the
/// shadow ray is an any hit that ends the ray at the first occluder its
/// light is linked to. The linking loop the inline route walks -- trace,
/// test, move TMin past the hit, trace again -- is one traversal here, which
/// is what an any hit is for.
const char* kRaysPipeline = R"(
uniform RaytracingAccelerationStructure scene;

struct PathHit {
    uint4  seen;          // (instance + 1, triangle, 0, 0); 0 for nothing
    float  t;
    float2 barycentrics;  // of the committed triangle: where the ray met it
};

struct NearestPayload {
    uint4  seen;
    float  t;
    float2 barycentrics;
};

struct ShadowPayload {
    uint occluded;
    uint shadowCategory;
};

[shader("miss")]
void pathNearestMiss(inout NearestPayload payload) {
    payload.seen = uint4(0);
    payload.t = 0.0;
    payload.barycentrics = float2(0.0);
}

[shader("closesthit")]
void pathNearestHit(inout NearestPayload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    payload.seen = uint4(InstanceID() + 1, PrimitiveIndex(), 0, 0);
    payload.t = RayTCurrent();
    payload.barycentrics = attributes.barycentrics;
}

[shader("miss")]
void pathShadowMiss(inout ShadowPayload payload) {
    payload.occluded = 0;
}

[shader("anyhit")]
void pathShadowAnyHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    // A light reaches only what its collection says: an occluder outside it
    // casts nothing, so the ray carries on past it.
    if (payload.shadowCategory != kLightUnlinked &&
        !lightLinked(payload.shadowCategory, instances[InstanceID()].categoriesLo,
                     instances[InstanceID()].categoriesHi)) {
        IgnoreHit();
        return;
    }
    payload.occluded = 1;
    AcceptHitAndEndSearch();
}

PathHit traceNearestFrom(float3 origin, float3 direction, float tMin, uint mask) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = 3.0e38;
    NearestPayload payload;
    payload.seen = uint4(0);
    payload.t = 0.0;
    payload.barycentrics = float2(0.0);
    TraceRay(scene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, mask, 0, 2, 0, ray, payload);
    PathHit hit;
    hit.seen = payload.seen;
    hit.t = payload.t;
    hit.barycentrics = payload.barycentrics;
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
    ShadowPayload payload;
    payload.occluded = 0;
    payload.shadowCategory = shadowCategory;
    // Not FORCE_OPAQUE: the any hit is where linking is decided.
    TraceRay(scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, mask, 1, 2, 1, ray, payload);
    return payload.occluded != 0;
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

/// Splats between a surface and a light.
///
/// The path tracer traces triangles; a splat cloud is not one, and until this
/// a relit cloud lit a floor it never darkened. The query is the shadow
/// query the splat tracer already has (rt_shadow.slang), over the cloud's
/// tables packed into one buffer -- because this kernel is at Metal's limit
/// of 31 buffers and the four the query wants would not fit
/// (rt_shadow_packed.slang, technique::SplatShadows).
///
/// Only where the device has inline rays: a ray tracing pipeline would need
/// the whole traversal as an any hit program of its own, and this one already
/// carries two.
const char* kSplats = R"(
static const bool kSplatShadows = true;

StructuredBuffer<float4>          splatPacked;
uniform RaytracingAccelerationStructure splatScene;

/// The offsets the packing kernel wrote the tables at, out of `path`.
PackedShadow splatTables() {
    PackedShadow where;
    where.frames = path.splatFrames;
    where.colours = path.splatColours;
    where.instances = path.splatInstances;
    where.indices = path.splatIndices;
    where.instanceCount = path.splatInstanceCount;
    return where;
}

/// What the cloud lets through between a surface and a light: one, where it
/// stands in no light's way. Offset off the surface exactly as pathOccluded
/// offsets, so the two answers are about the same segment.
float splatVisibility(float3 p, float3 n, float3 wi, float distance) {
    const float scale = max(1.0, length(p));
    const float3 away = dot(n, wi) < 0.0 ? -n : n;
    LrtRay ray;
    ray.Origin = p + (away + wi) * (1.0e-3 * scale);
    ray.Direction = wi;
    ray.TMin = 1.0e-3 * scale;
    ray.TMax = max(distance - ray.TMin, 0.0);
    if (ray.TMax <= ray.TMin) {
        return 1.0;
    }
    // Every particle between the two ends counts, however near or far.
    return packedShadowTransmittance(splatPacked, splatTables(), splatScene, ray, path.splatCut, 0.0, 3.0e38);
}
)";

const char* kNoSplats = R"(
static const bool kSplatShadows = false;
float splatVisibility(float3 p, float3 n, float3 wi, float distance) { return 1.0; }
)";

/// The denoiser's guides -- the first hit's albedo and shading normal -- are
/// a buffer of their own, and a buffer is what this kernel has none of to
/// spare on Metal: 31 is the limit and the frame that shadows meshes with a
/// cloud is at it. So they are declared only in the kernel of a frame that
/// asked for them, as the light groups and the volumes are.
const char* kAux = R"(
static const bool kAux = true;
RWStructuredBuffer<float4> aux;   // the albedo's plane, then the normal's
void writeAuxAt(uint at, uint pixels, float4 albedo, float4 normal) {
    aux[at] = albedo;
    aux[pixels + at] = normal;
}
)";

/// Baking: the first hit of every path comes from a ray a caller wrote down,
/// not from a camera. That is the whole of the difference between a frame and
/// a bake -- what a path does after its first vertex is the same integrator,
/// the same lights, the same bounces -- so the bake is a variant of this
/// kernel rather than a second one that would drift from it.
///
/// A variant and not a uniform because a buffer is what this kernel has none
/// of to spare on Metal: 31 is the limit (kSplats says the rest).
const char* kBake = R"(
static const bool kBake = true;
StructuredBuffer<float4> bakeRays;   // two entries a point: where it is and how far off to start, then its normal

/// The material a bake shades its first vertex with: its body, not its polish.
///
/// A gaussian carries one colour, and a reflection is exactly the part of a
/// surface that one colour cannot hold: baked in, it is the same from every
/// direction, and the chess set's marble came back as smooth grey plastic
/// with its veining gone -- the texture was there, at three percent of a
/// specular that covered it.
///
/// So the bake keeps what a colour *can* hold and what the renderer cannot
/// work out for itself: the diffuse lobes, which carry the texture and the
/// light that reached it, what a material transmits, and a conductor's
/// reflection -- a metal has no body but that, and dropping it would leave
/// gold black. What it drops is the dielectric polish and the sheen, which
/// `splat_relight` puts back at render time, from the metallic and roughness
/// the gaussian carries, and puts back *with a direction in it*.
LobeStack bakeBody(LobeStack stack) {
    LobeStack body = stack;
    body.count = 0;
    for (uint k = 0; k < stack.count; ++k) {
        const Lobe lobe = stack.lobes[k];
        const bool diffuse = lobe.kind == kLobeOrenNayar || lobe.kind == kLobeBurley ||
                             lobe.kind == kLobeTranslucent || lobe.kind == kLobeHair;
        const bool metal = lobe.kind == kLobeConductor;
        const bool through = lobe.scatter == kScatterTransmit;
        if (diffuse || metal || through) {
            body.lobes[body.count] = lobe;
            body.count += 1;
        }
    }
    return body;
}

/// The point seen from one direction of the hemisphere it faces.
///
/// A colour that is one number cannot be view-dependent, so what a bake stores
/// is the average of the radiance leaving the surface over that hemisphere,
/// weighted by the cosine -- which is exactly the radiance a diffuse surface
/// of the same radiosity has, and which spreads a highlight over the
/// gaussians that would show it instead of putting all of it on the ones whose
/// normal points at the light. Baking along the normal alone was tried first:
/// under a dome it makes every gaussian show the same reflection, and the
/// marble it was converted from came out as polished plastic.
///
/// One direction a sample, so the mean over the paths is the mean over the
/// hemisphere. The ray still starts where the caller put it and still ends on
/// the surface it named.
/// A point of the half of the sphere `n` faces, from a point of the unit
/// square: the elevation straight from `u.x`, so that the square's area maps
/// to the hemisphere's solid angle, and the azimuth from `u.y`.
float3 bakeAim(float3 n, float2 u) {
    const float z = u.x;
    const float r = sqrt(max(1.0 - z * z, 0.0));
    const float phi = 2.0 * 3.14159265358979 * u.y;
    float3 tangent = abs(n.z) < 0.999 ? normalize(cross(float3(0.0, 0.0, 1.0), n))
                                      : float3(1.0, 0.0, 0.0);
    const float3 bitangent = cross(n, tangent);
    return normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * z);
}

/// Which way this sample looks at the point, in the world.
float3 bakeDirection(uint at, uint sample) {
    const float4 d = bakeRays[at * 2 + 1];
    const float3 n = normalize(d.xyz);
    const uint2 pixel = uint2(at % max(camera.width, 1u), at / max(camera.width, 1u));
    // Stratified, not drawn: the radiance leaving a glossy surface swings by
    // orders of magnitude across the hemisphere, so directions taken at
    // random leave one gaussian in the mirror of the sun and its neighbour
    // nowhere near it -- salt and pepper that more paths barely touch. A
    // grid with a jitter in each cell covers it evenly, and the same count
    // then answers a different question.
    const uint side = max(uint(sqrt(float(max(path.samples, 1u))) + 0.5), 1u);
    const uint2 cell = uint2(sample % side, (sample / side) % side);
    const float2 u = saturate((float2(cell) + random2(pixel, sample, 0u, 41u)) / float(side));
    // UNIFORM, and over the half of the sphere the surface faces.
    //
    // These directions are what the harmonics are projected over, and the
    // harmonics are orthonormal over the sphere and over nothing else: fitted
    // a coefficient at a time over a hemisphere they are not orthogonal, each
    // one explains the same light again, and their sum overshoots -- measured,
    // the pawn came back ten times too bright. So the integral is over the
    // sphere with the far half taken as nothing, which is what a one-sided
    // surface sends there. Drawing only from the near half and measuring it
    // with 2 pi is the same integral with none of the samples thrown away.
    //
    // Not cosine weighted: the cosine belongs to a reflection integral, and
    // this one is a projection.
    return bakeAim(n, u);
}

/// Whether a sample looks at the surface from the side it faces. Every one of
/// them does now that they are drawn from that half, and the test stands for
/// a normal that is not quite the one the ray was built on.
bool bakeInFront(uint at, uint sample) {
    return dot(bakeDirection(at, sample), normalize(bakeRays[at * 2 + 1].xyz)) > 0.0;
}

Found foundBaked(uint at, uint sample, uint mask) {
    const float4 o = bakeRays[at * 2];   // the point, and how far off the surface to start
    if (!bakeInFront(at, sample)) {
        return foundNothing();   // behind the surface: this sample is a zero, and costs no ray
    }
    // WHICH SURFACE, and FROM WHERE, are two questions.
    //
    // The surface is found by a ray straight down the normal, which always
    // meets the point it was built from. Aiming the ray along the direction
    // the sample looks from was tried instead, and at grazing angles it
    // travels beside the surface rather than onto it: on the pawn 55000
    // gaussians of 729073 found nothing, came back black, and speckled the
    // model in a way no number of paths touched.
    //
    // The direction is the sample's, and it is set afterwards: the geometry
    // is the same point either way, and what the material is asked is what it
    // sends towards the eye -- which for a bake is wherever this sample looks
    // from.
    const float3 n = normalize(bakeRays[at * 2 + 1].xyz);
    const float3 from = o.xyz + n * o.w;
    const PathHit hit = traceNearestFrom(from, -n, o.w * 0.01, mask);
    Found f = foundHit(hit, from, -n);
    if (f.valid) {
        f.eye = f.positionWorld + bakeDirection(at, sample) * o.w;
        f.rayOrigin = f.eye;
        f.rayT = o.w;
    }
    return f;
}

/// One basis function where a sample looks from.
///
/// The cloud stores its colours as harmonics of the direction from the eye to
/// the splat, so the basis is read at `-towards`: the sample looks from the
/// surface out, and the renderer looks the other way along the same line.
float bakeBasisAt(uint at, uint sample, uint basis) {
    return shBasisValue(basis, -bakeDirection(at, sample));
}

/// The measure a uniform sample of the sphere stands for: the estimator of
/// `integral(L * Y)` over N of them is `(4 pi / N) * sum(L * Y)`.
static const float kBakeMeasure = 2.0 * 3.14159265358979;

/// Which way the surface under a gaussian faces.
float3 bakeNormalAt(uint at) {
    return normalize(bakeRays[at * 2 + 1].xyz);
}

/// Which band a coefficient belongs to: 0, 1, 1, 1, 2, 2, 2, 2, 2, 3...
uint bakeBand(uint k) {
    return uint(sqrt(float(k)));
}

/// THE FIT IS OVER THE HALF OF THE SPHERE THE SURFACE FACES, and the basis is
/// not orthogonal there.
///
/// Projecting instead -- over the whole sphere, with the far half taken as
/// nothing -- is what the bake did, and it is a different question with a
/// visibly different answer. The function being projected jumps from the
/// radiance to zero at the equator, so the series takes the middle of that
/// jump where the two meet: **half**. And the equator is exactly where a
/// silhouette is looked at. Measured across the pawn's glass ball, the mesh
/// reads a flat 0.309 and the baked cloud read 0.152 at the edge, climbing
/// over a hundred pixels to 0.325 at the centre -- a dark rim around every
/// silhouette in the model. Degree 3 rang about it (0.224, 0.271, 0.193) and
/// came out further from the mesh than degree 2, which is Gibbs and not noise.
///
/// The fit is the normal equations, `G c = b`, with `b_k = integral(L Y_k)`
/// over that half -- what the samples already sum to -- and
/// `G_kj = integral(Y_k Y_j)` over it. What makes them small enough to solve
/// a gaussian at a time:
///
/// **Two bands of the same parity are orthogonal over any half of the
/// sphere.** `Y(-w) = (-1)^l Y(w)`, so the far half's integral is the near
/// half's times `(-1)^(l_k + l_j)`; when that is +1 the two halves are equal
/// and each is half the sphere's, which is `delta_kj / 2` -- whatever the
/// normal is. So the even-even and odd-odd blocks are exactly `I/2` and are
/// not accumulated at all. Only the even-odd block `E` depends on the
/// direction the surface faces, and only it is summed: at most six by ten.
///
/// That leaves `[[I/2, E], [E^T, I/2]]`, whose Schur complement on the even
/// side is `I/2 - 2 E E^T` -- **six by six at most**, symmetric, and positive
/// definite because `G` is. Cholesky solves it, and the odd half follows.
///
/// Degree 0 falls out of the same arithmetic: no odd terms, so `E` is empty,
/// the system is `x / 2 = b`, and the constant a gaussian stores is twice the
/// projection. Which is why the projection had it at a fifth of the mesh.
/// THE MATRIX COSTS NO RAYS. It is `integral(Y_k Y_j)` over the half of the
/// sphere the surface faces: it depends on the normal and on nothing else, not
/// on the light, not on the material, not on a single path. So it is not
/// estimated from the paths -- taken that way, at 256 of them, the whole fit
/// came apart: its worst mode is a hundred times smaller than its best, the
/// sampling error on the matrix is of the same size as the entries it is
/// estimating, and the pawn came back with pixels in the thousands.
///
/// It is integrated instead, deterministically, over a grid that costs no
/// rays: 32 elevations by 16 azimuths, which for a product of two basis
/// functions -- a trigonometric polynomial of degree six in the azimuth and,
/// once that sum has killed the terms with unequal order, a polynomial in the
/// elevation -- is worth about four figures. 512 directions of arithmetic
/// against 256 of ray tracing.
void bakeGram(float3 n, out float e[6][10], uint evens[6], uint ne, uint odds[10], uint no) {
    for (uint i = 0; i < ne; ++i) {
        for (uint j = 0; j < no; ++j) {
            e[i][j] = 0.0;
        }
    }
    const uint kElevations = 32;
    const uint kAzimuths = 16;
    for (uint zi = 0; zi < kElevations; ++zi) {
        for (uint pi = 0; pi < kAzimuths; ++pi) {
            const float2 u = float2((float(zi) + 0.5) / float(kElevations),
                                    (float(pi) + 0.5) / float(kAzimuths));
            // The same direction the samples are read at, so the matrix and
            // the right hand side are over the same half of the sphere.
            const float3 towards = -bakeAim(n, u);
            float basis[16];
            for (uint c = 0; c < 16; ++c) {
                basis[c] = shBasisValue(c, towards);
            }
            for (uint i = 0; i < ne; ++i) {
                for (uint j = 0; j < no; ++j) {
                    e[i][j] += basis[evens[i]] * basis[odds[j]];
                }
            }
        }
    }
    const float measure = 2.0 * 3.14159265358979 / float(kElevations * kAzimuths);
    for (uint i = 0; i < ne; ++i) {
        for (uint j = 0; j < no; ++j) {
            e[i][j] *= measure;
        }
    }
}

/// WHERE THE FIT IS STOPPED FROM AMPLIFYING THE PATHS` OWN NOISE.
///
/// Half of the sphere does not determine sixteen harmonics equally: the
/// combinations that are nearly nothing on the half the surface faces, and
/// large on the half it does not, are what the data cannot see. Solved
/// exactly, the fit puts the noise of a few hundred paths into exactly those.
/// On the pawn a few gaussians came back with coefficients of **65344** --
/// fp16's ceiling, which is what a cloud stores them in -- and burned out as
/// white blobs.
///
/// A ridge was tried first, added to every diagonal. It works, and it charges
/// every gaussian for the few that need it: at 0.02 the fit shrinks by
/// `0.5 / 0.52`, and a Lambertian plane whose light is 0.4614 came back at
/// 0.4436 -- 3.9% low, exactly that ratio -- with degree 2 at 9%.
///
/// A floor under the pivot charges nobody. Cholesky reaches a small pivot
/// exactly where the matrix is near singular, which is the direction the data
/// could not see, and flooring it there bounds what that direction can
/// contribute while leaving every well determined gaussian solved exactly.
/// The diagonal being a half, a floor of 0.005 caps the amplification at
/// about fourteen.
static const float kBakePivot = 0.005;

void bakeFit(inout float3 c[16], float e[6][10], uint evens[6], uint ne, uint odds[10], uint no,
             float measure) {
    float3 be[6];
    float3 bo[10];
    for (uint i = 0; i < ne; ++i) {
        be[i] = c[evens[i]] * measure;
    }
    for (uint j = 0; j < no; ++j) {
        bo[j] = c[odds[j]] * measure;
    }
    // The diagonal: a half, from the two bands being of the same parity, which
    // holds whatever the normal is.
    const float d = 0.5;
    // The Schur complement, and the even side's right hand side with the odd
    // half eliminated.
    float  m[6][6];
    float3 r[6];
    for (uint i = 0; i < ne; ++i) {
        float3 sum = float3(0.0);
        for (uint j = 0; j < no; ++j) {
            sum += e[i][j] * bo[j];
        }
        r[i] = be[i] - sum / d;
        for (uint k = 0; k <= i; ++k) {
            float dot = 0.0;
            for (uint j = 0; j < no; ++j) {
                dot += e[i][j] * e[k][j];
            }
            m[i][k] = (i == k ? d : 0.0) - dot / d;
            m[k][i] = m[i][k];
        }
    }
    // Cholesky, in place: m = L L^T, then two triangular solves.
    for (uint i = 0; i < ne; ++i) {
        for (uint k = 0; k <= i; ++k) {
            float sum = m[i][k];
            for (uint j = 0; j < k; ++j) {
                sum -= m[i][j] * m[k][j];
            }
            m[i][k] = i == k ? sqrt(max(sum, kBakePivot)) : sum / m[k][k];
        }
    }
    float3 x[6];
    for (uint i = 0; i < ne; ++i) {
        float3 sum = r[i];
        for (uint j = 0; j < i; ++j) {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
    for (int i = int(ne) - 1; i >= 0; --i) {
        float3 sum = x[i];
        for (uint j = uint(i) + 1; j < ne; ++j) {
            sum -= m[j][i] * x[j];
        }
        x[i] = sum / m[i][i];
    }
    for (uint i = 0; i < ne; ++i) {
        c[evens[i]] = x[i];
    }
    for (uint j = 0; j < no; ++j) {
        float3 sum = bo[j];
        for (uint i = 0; i < ne; ++i) {
            sum -= e[i][j] * x[i];
        }
        c[odds[j]] = sum / d;
    }
}
)";

const char* kNoBake = R"(
static const bool kBake = false;
Found foundBaked(uint at, uint sample, uint mask) { return foundNothing(); }
LobeStack bakeBody(LobeStack stack) { return stack; }
float bakeBasisAt(uint at, uint sample, uint basis) { return 0.0; }
static const float kBakeMeasure = 0.0;
uint bakeBand(uint k) { return 0; }
float3 bakeNormalAt(uint at) { return float3(0.0, 0.0, 1.0); }
void bakeGram(float3 n, out float e[6][10], uint evens[6], uint ne, uint odds[10], uint no) {
    for (uint i = 0; i < 6; ++i) {
        for (uint j = 0; j < 10; ++j) {
            e[i][j] = 0.0;
        }
    }
}
void bakeFit(inout float3 c[16], float e[6][10], uint evens[6], uint ne, uint odds[10], uint no,
             float measure) {}
)";

const char* kNoAux = R"(
static const bool kAux = false;
void writeAuxAt(uint at, uint pixels, float4 albedo, float4 normal) {}
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
static const bool kEmissive = false;
static const uint emissiveOn = 0;
float emissiveWord(uint i) { return 0.0; }
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
// Emitting triangles as a light (EmissiveTable.h): a frame without media only.
static const bool kEmissive = true;
StructuredBuffer<float>        emissiveTable;
uniform uint                   emissiveOn;
float emissiveWord(uint i) { return emissiveTable[i]; }
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
    // A bake takes no footprint from the camera it does not have.
    out.inputs = materialInputsAt(camera, toWorld, pixel.x, pixel.y, s, lookup.time, !kBake);
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

// ---- emitting triangles as a light -------------------------------------------

uint emissiveCount(uint word) {
    return asuint(emissiveWord(word));
}

/// The emitting triangles' share of next event estimation's choices: their
/// power against the lights'. 0 where nothing emits or the frame has media.
float emissiveShare() {
    // Without MIS the material's rays gather emission at full weight, so
    // next event estimation leaves the emitting triangles to them.
    if (!kEmissive || emissiveOn == 0 || path.mis == 0) {
        return 0.0;
    }
    const float emitted = emissiveWord(2);
    if (!(emitted > 0.0)) {
        return 0.0;
    }
    const float lit = lightCount > 0 ? max(lights[lightCount - 1].cumulative, 0.0) : 0.0;
    return emitted / (emitted + lit);
}

/// A surface's triangle in world space, as the record that drew it holds it.
void worldTriangle(InstanceRecord r, MeshRecord m, uint triangle, out float3 a, out float3 b, out float3 c) {
    const uint t = m.firstTriangle + triangle;
    const uint base = m.firstPoint + r.pointsOffset;
    a = rowsApply(r.world0, r.world1, r.world2, positions[base + indices[t * 3]].xyz, 1.0);
    b = rowsApply(r.world0, r.world1, r.world2, positions[base + indices[t * 3 + 1]].xyz, 1.0);
    c = rowsApply(r.world0, r.world1, r.world2, positions[base + indices[t * 3 + 2]].xyz, 1.0);
}

/// The density, per solid angle at `from`, that next event estimation gives
/// the direction that met surface s at `at` -- by the table's own weights
/// (area times the row's luminance), so the two strategies' weights sum to one.
float emissivePdfAt(Surface s, float3 at, float3 from) {
    const float share = emissiveShare();
    if (!(share > 0.0)) {
        return 0.0;
    }
    const uint rows = emissiveCount(3);
    const uint row = materialRowOf(s);
    const float luminance = row < rows ? emissiveWord(4 + row) : 0.0;
    if (!(luminance > 0.0)) {
        return 0.0;
    }
    float3 a, b, c;
    worldTriangle(s.instance, s.mesh, s.triangle, a, b, c);
    const float3 crossed = cross(b - a, c - a);
    const float twiceArea = length(crossed);
    const float3 offset = at - from;
    const float distance2 = dot(offset, offset);
    if (!(twiceArea > 0.0) || !(distance2 > 0.0)) {
        return 0.0;
    }
    const float cosine = abs(dot(crossed / twiceArea, offset / sqrt(distance2)));
    if (cosine <= 1.0e-6) {
        return 0.0;
    }
    return share * luminance * distance2 / (cosine * emissiveWord(2));
}

/// A point on an emitting triangle, for next event estimation from `from`:
/// the triangle chosen by its power, the point uniform over it. The surface
/// comes back unshaded -- its emission is the material's, evaluated where
/// every material is -- with the true density it was drawn with (`pdf`) and
/// the tabled one both strategies weigh by (`weighPdf`).
struct LightPoint {
    Found  found;
    float3 wi;
    float  distance;
    float  pdf;
    float  weighPdf;
    bool   valid;
};

LightPoint sampleEmissive(float3 from, float u, float2 v) {
    LightPoint lp;
    lp.valid = false;
    lp.pdf = 0.0;
    lp.weighPdf = 0.0;
    lp.distance = 0.0;
    lp.wi = float3(0.0, 0.0, 1.0);
    const float share = emissiveShare();
    const uint triangles = emissiveCount(0);
    const uint records = emissiveCount(1);
    const uint rows = emissiveCount(3);
    const float emitted = emissiveWord(2);
    if (!(share > 0.0) || triangles == 0) {
        return lp;
    }
    const uint cumulative = 4 + rows + records + 1;
    const float target = clamp(u, 0.0, 0.999999) * emitted;
    uint lo = 0;
    uint hi = triangles - 1;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (emissiveWord(cumulative + 1 + mid) <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const uint g = lo;
    const float probability = (emissiveWord(cumulative + 1 + g) - emissiveWord(cumulative + g)) / emitted;
    if (!(probability > 0.0)) {
        return lp;
    }
    uint rlo = 0;
    uint rhi = records;
    while (rhi - rlo > 1) {
        const uint mid = (rlo + rhi) / 2;
        if (emissiveCount(4 + rows + mid) <= g) {
            rlo = mid;
        } else {
            rhi = mid;
        }
    }
    const uint triangle = g - emissiveCount(4 + rows + rlo);
    const InstanceRecord r = instances[rlo];
    const MeshRecord m = meshes[r.mesh];
    float3 a, b, c;
    worldTriangle(r, m, triangle, a, b, c);
    const float root = sqrt(v.x);
    const float3 weights = float3(1.0 - root, v.y * root, root * (1.0 - v.y));
    const float3 at = a * weights.x + b * weights.y + c * weights.z;
    const float3 crossed = cross(b - a, c - a);
    const float twiceArea = length(crossed);
    const float3 offset = at - from;
    const float distance2 = dot(offset, offset);
    if (!(twiceArea > 0.0) || !(distance2 > 0.0)) {
        return lp;
    }
    lp.distance = sqrt(distance2);
    lp.wi = offset / lp.distance;
    const float cosine = abs(dot(crossed / twiceArea, lp.wi));
    if (cosine <= 1.0e-6) {
        return lp;
    }
    lp.pdf = share * probability * distance2 / (cosine * 0.5 * twiceArea);
    // The surface there, rebuilt as a bounce's hit is: the direction it is
    // reached along, turned to view space for which side it shows.
    const float3 w = lp.wi;
    const float3 viewDirection = float3(toWorld.row0.x * w.x + toWorld.row1.x * w.y + toWorld.row2.x * w.z,
                                        toWorld.row0.y * w.x + toWorld.row1.y * w.y + toWorld.row2.y * w.z,
                                        toWorld.row0.z * w.x + toWorld.row1.z * w.y + toWorld.row2.z * w.z);
    Found f;
    f.s = surfaceFromWeights(uint4(rlo + 1, triangle, 0, 0), weights, viewDirection);
    f.valid = true;
    f.positionWorld = at;
    f.eye = from;
    f.rayOrigin = from;
    f.rayT = lp.distance;
    f.depth = f.s.depth;
    lp.found = f;
    lp.weighPdf = emissivePdfAt(f.s, at, from);
    lp.valid = lp.weighPdf > 0.0;
    return lp;
}

/// Light k's probability of being chosen at p, n, as gatherLight chooses:
/// among the lights, times the lights' share against the emitting triangles.
float lightChoiceProbability(uint k, float3 p, float3 n) {
    const float lights_ = 1.0 - emissiveShare();
    if (path.chooseLights == 2) {
        return lights_ * lightPdfChoiceAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount, lights, k, p, n);
    }
    const float power = lights[lightCount - 1].cumulative;
    if (!(power > 0.0)) {
        return 0.0;
    }
    const float before = k == 0 ? 0.0 : lights[k - 1].cumulative;
    return lights_ * max(lights[k].cumulative - before, 1.0e-9) / power;
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
    // The choice's first part is the emitting triangles' (the caller takes
    // those); what is left of the number chooses among the lights.
    const float share = emissiveShare();
    const float pick = (random(pixel, sample, bounce, 11u) - share) / max(1.0 - share, 1.0e-9);
    const LightChoice choice = path.chooseLights == 2
                                   ? chooseLightAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount,
                                                    sh.inputs.positionWorld, sh.inputs.normalWorld, saturate(pick))
                                   : chooseLight(lights, lightCount, saturate(pick));
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
    if (kSplatShadows && (light.flags & kLightShadow) != 0) {
        transmittance *= splatVisibility(sh.inputs.positionWorld, sh.inputs.normalWorld, ls.wi, ls.distance);
        if (!(transmittance > 0.0)) {
            return float3(0.0);
        }
    }
    const float density = ls.pdf * choice.probability * (1.0 - share);
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
    float splatThrough = 1.0;
    if (kSplatShadows && (light.flags & kLightShadow) != 0) {
        splatThrough = splatVisibility(p, ls.wi, ls.wi, ls.distance);
        if (!(splatThrough > 0.0)) {
            return float3(0.0);
        }
    }
    if ((light.flags & kLightShadow) != 0 && pathOccluded(p, ls.wi, ls.wi, ls.distance, light.shadowCategory, mask)) {
        return float3(0.0);
    }
    float transmittance = 1.0;
    if ((light.flags & kLightShadow) != 0) {
        uint rng = mediumSeed(pixel, sample, bounce, 25u);
        transmittance = mediumTransmittanceAny(p, ls.wi, 1.0e-4 * max(1.0, length(p)), ls.distance, rng);
    }
    return f * ls.radiance * transmittance * splatThrough / (ls.pdf * choice.probability);
}

void tracePathsAt(uint2 group, uint index) {
    const uint2 tid = lrtQuadPixel(group, index);
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    // A bake has no visibility buffer: its first hits are rays of its own,
    // and the texture bound in its place is not this size.
    const uint4 seen = kBake ? uint4(0, 0, 0, 0)
                             : visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
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
    const Found firstFound = kBake      ? foundBaked(at, 0u, sampleMask(tid, 0u))
                             : ownRays  ? foundLensSample(tid, 0u, sampleMask(tid, 0u))
                                        : foundAt(tid, seen);
    Shaded first;
    first.valid = false;
    first.depth = 0.0;
    bool firstShaded = false;
    bool auxWritten = false;
    if (kAux && path.writeAux != 0) {
        writeAuxAt(at, pixels, float4(0.0), float4(0.0));
    }
    float3 total = float3(0.0);
    // A bake fits every harmonic in the one pass: the paths are the same for
    // all of them and only the weight differs, so tracing them once and
    // weighing them sixteen ways is sixteen times less work than tracing them
    // sixteen times. Measured before this, the pawn took 5m43 at degree 2.
    float3 coefficients[16];
    // Which coefficient is on which side of the fit's matrix: bakeGram says
    // why only the block between the two is worked out at all.
    uint   evens[6];
    uint   odds[10];
    uint   ne = 0;
    uint   no = 0;
    if (kBake) {
        for (uint c = 0; c < 16; ++c) {
            coefficients[c] = float3(0.0);
        }
        for (uint c = 0; c < min(path.bakeCount, 16u); ++c) {
            if ((bakeBand(c) & 1u) == 0u) {
                evens[ne++] = c;
            } else {
                odds[no++] = c;
            }
        }
    }
    float  alpha = 0.0;
    float  hitDepth = 0.0;
    float  squares = 0.0;   // sum of each sample's luminance squared
    uint group = 0;
    Groups groups = groupsZero();
    for (uint sample = 0; sample < samples && (firstFound.valid || ownRays || kVolumes); ++sample) {
        const uint mask = sampleMask(tid, sample);
        // A bake looks from a new direction every sample (foundBaked says
        // why); a frame's first hit is the same one for all of them unless a
        // lens moves it.
        Found found = kBake ? (sample == 0 ? firstFound : foundBaked(at, sample, mask))
                      : ownRays && sample > 0 ? foundLensSample(tid, sample, mask)
                                              : firstFound;
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
        // Steps, not bounces: a step shades either a vertex of the path or a
        // point next event estimation chose on an emitting triangle, whose
        // emission is its material's -- evaluated at the same one call as
        // every surface's (Found says why there is one). A vertex that chose
        // such a point waits while it is shaded, then goes on. The trip count
        // is a uniform's, so the compiler cannot unroll the call into copies.
        uint bounce = 0;
        Shaded cur;
        cur.valid = false;
        bool   lightStep = false;     // the next step shades the light point, not the path
        Found  lightPoint;
        float3 lightScale = float3(0.0);
        float  previousPdf = 0.0;     // the material's density for the direction that reached `found`
        bool   previousWeighs = false;   // and whether emission met there is weighed against it
        const uint steps = 2 * path.bounces + 3;
        for (uint step = 0; step < steps; ++step) {
            if (!lightStep) {
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
                    previousWeighs = false;
                    ++bounce;
                    continue;
                }
                if (!found.valid) {
                    break;   // the ray escaped
                }
            }
            // The one place materials are evaluated; the camera's hit once a pixel.
            const Found target = lightStep ? lightPoint : found;
            // A bake looks at its point from a new direction every sample, so
            // its first vertex is not the one hit a pixel shades once: cached,
            // every sample would answer with sample zero's eye and the
            // harmonics past the constant would come back as noise about zero.
            const bool cameraHit = !lightStep && bounce == 0 && !ownRays && !kBake;
            Shaded shaded;
            if (cameraHit && firstShaded) {
                shaded = first;
            } else {
                shaded = shadeFound(tid, target);
                if (cameraHit) {
                    first = shaded;
                    firstShaded = true;
                }
            }
            if (lightStep) {
                // The light point's emission, back to the vertex that chose it.
                carried += lightScale * shaded.stack.emission;
                lightStep = false;
            } else {
                cur = shaded;
                if (kBake && bounce == 0) {
                    // The body of the material, never its polish (bakeBody
                    // says why). Baking the polish into harmonics was tried:
                    // a reflection off a surface of roughness 0.1 is far too
                    // sharp for sixteen coefficients, and the pawn came back
                    // silver -- 0.276 where the mesh reads 0.085. What the
                    // harmonics hold is how the body's own light changes with
                    // the direction; the reflection stays a lobe, which knows
                    // where the eye is.
                    cur.stack = bakeBody(cur.stack);
                }
                if (kAux && bounce == 0 && path.writeAux != 0 && !auxWritten) {
                    writeAuxAt(at, pixels, float4(stackAlbedo(cur.stack, cur.toEye), 1.0),
                               float4(cur.inputs.normalWorld, 1.0));
                    auxWritten = true;
                }
                if (!vertexSeen) {
                    vertexSeen = true;
                    opacity = cur.stack.opacity;
                    depthHere = cur.depth;
                }
                // Emission met by the material's ray, weighed against next event
                // estimation where that could have chosen this point.
                float emissionWeight = 1.0;
                if (previousWeighs && any(cur.stack.emission > float3(0.0))) {
                    const float other = emissivePdfAt(found.s, found.positionWorld, found.rayOrigin);
                    emissionWeight = previousPdf * previousPdf / (previousPdf * previousPdf + other * other);
                }
                carried += throughput * cur.stack.emission * emissionWeight;
                const float share = emissiveShare();
                const bool continues = kTraces && bounce < path.bounces;
                if (share > 0.0 && random(tid, sample, bounce, 11u) < share) {
                    // An emitting triangle: sampled here, shaded next step.
                    const LightPoint lp = sampleEmissive(cur.inputs.positionWorld,
                                                         random(tid, sample, bounce, 11u) / share,
                                                         random2(tid, sample, bounce, 3u));
                    if (lp.valid) {
                        const float3 f = stackEval(cur.stack, cur.toEye, lp.wi);
                        // The shadow ray stops short of the emitting triangle by more
                        // than its origin is moved off the surface (pathOccluded moves
                        // it up to 2e-3 of the scale along the ray): at a relative
                        // 1e-4 short it reached the triangle itself, and every sample
                        // was its own shadow.
                        const float shortOf = 3.0e-3 * max(1.0, length(cur.inputs.positionWorld));
                        if (any(f > float3(0.0)) && lp.distance > 2.0 * shortOf &&
                            !pathOccluded(cur.inputs.positionWorld, cur.inputs.normalWorld, lp.wi,
                                          lp.distance - shortOf, kLightUnlinked, mask)) {
                            float weight = 1.0;
                            if (continues && path.mis != 0) {
                                const float other = stackPdf(cur.stack, cur.toEye, lp.wi);
                                weight = lp.weighPdf * lp.weighPdf / (lp.weighPdf * lp.weighPdf + other * other);
                            }
                            const float throughSplats =
                                kSplatShadows ? splatVisibility(cur.inputs.positionWorld,
                                                                cur.inputs.normalWorld, lp.wi, lp.distance)
                                              : 1.0;
                            if (!(throughSplats > 0.0)) {
                                continue;
                            }
                            lightScale = throughput * f * weight * throughSplats / lp.pdf;
                            lightPoint = lp.found;
                            lightStep = true;
                            continue;
                        }
                    }
                } else {
                    const float3 direct = gatherLight(cur, tid, sample, bounce, mask, group);
                    carried += throughput * direct;
                    if (kLightGroups) {
                        addGroup(groups, group, throughput * direct * opacity);
                    }
                }
            }
            // The path goes on from `cur`.
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
            // bounce left from; shaded when the next step comes to it.
            found = foundHit(hit, p, d);
            tHit = hit.seen.x == 0 ? 1.0e30 : hit.t;
            previousPdf = ms.pdf;
            previousWeighs = path.mis != 0 && !ms.delta;
            ++bounce;
        }
        if (!vertexSeen) {
            continue;   // nothing along the ray at all: transparent, and counted
        }
        hitDepth = depthHere;
        alpha += opacity;
        // A bake fits harmonics: each sample is weighed by the basis where it
        // looked from, and it is fitted in the space a cloud is blended in
        // (frame.slang), so that what the renderer decodes is what the tracer
        // answered.
        const float3 sampleColour = kBake ? linearToSrgb(carried) : carried;
        if (kBake) {
            for (uint c = 0; c < min(path.bakeCount, 16u); ++c) {
                coefficients[c] += sampleColour * opacity * bakeBasisAt(at, sample, c);
            }
        }
        total += sampleColour * opacity;
        const float lum = dot(sampleColour * opacity, kPathLuminance);
        squares += lum * lum;
    }
    // A mean of products, not a product of means: `total` already carries each
    // sample's colour times that sample's opacity, so the sum is the sum. The
    // two are the same while every sample is opaque -- which is every test
    // there is today -- and differ as soon as opacity varies per sample, and
    // the product form also broke the invariant that 256 paths in one pass
    // must equal 64 in each of four.
    // A frame wants the mean of its samples; a bake wants the coefficient,
    // which is the same sum over the basis against itself rather than over
    // the count. The constant term is shifted to where a cloud keeps its DC.
    float4 added = float4(total, alpha) / float(samples);
    if (kBake) {
        // Each coefficient into a plane of its own, and the first one shifted
        // to where a cloud keeps its DC: it stores the constant term the way
        // 3DGS trains it, `colour = 0.5 + kSH0 * dc`, while a projection gives
        // `colour = c0 * kSH0`.
        // The samples are a fit, not a projection: bakeFit says what that
        // changes and why the matrix it solves is small, and bakeGram why the
        // matrix costs no rays.
        float e[6][10];
        bakeGram(bakeNormalAt(at), e, evens, ne, odds, no);
        bakeFit(coefficients, e, evens, ne, odds, no, kBakeMeasure / float(samples));
        for (uint c = 0; c < min(path.bakeCount, 16u); ++c) {
            float3 value = coefficients[c];
            if (c == 0) {
                value -= float3(kShDcOffset);
            }
            colour[c * pixels + at] = float4(value, alpha / float(samples));
        }
        return;
    }
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

/// The pixel's own entry, where the device runs a compute kernel.
const char* kEntryCompute = R"(
[shader("compute")]
[numthreads(16, 16, 1)]
void tracePaths(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    tracePathsAt(group.xy, index);
}
)";

/// And where it runs a ray generation program instead. A dispatch of rays has
/// no thread groups, so the launch index is cut into the group and the index
/// within it -- the same 16 x 16 block, walked in the same quad order, since
/// the material's derivatives are read across the quad's lanes and those are
/// consecutive launch indices here.
const char* kEntryRays = R"(
[shader("raygeneration")]
void tracePathsGen() {
    const uint flat = DispatchRaysIndex().x;
    const uint groupsAcross = (camera.width + 15u) / 16u;
    const uint group = flat / 256u;
    tracePathsAt(uint2(group % groupsAcross, group / groupsAcross), flat % 256u);
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
    return setPrograms(programs, groups_, volumes_, splats_, aux_);
}

Result<void> PathTracer::setPrograms(const MaterialPrograms& programs, bool groups, bool volumes, bool splats,
                                    bool aux) {
    // `bake_` is set by the caller before this, and it is a variant like the
    // rest: left out of this test, a frame that turned it on ran the kernel
    // compiled without it -- which has no rays to read and is the frame's own
    // size, so exactly one point of the bake came back (the camera's pixel).
    if (programs.module() == module_ && groups == groups_ && volumes == volumes_ && splats == splats_ &&
        aux == aux_ && bake_ == bakeBuilt_ && (kernel_.has_value() || rayKernel_.has_value())) {
        return ok();
    }
    bakeBuilt_ = bake_;
    splats_ = splats;
    aux_ = aux;
    const bool traces = device_->caps().rayQuery && device_->caps().accelerationStructure;
    const std::string name = programs.module() + (traces ? "_path_traced" : "_path_direct") +
                             (groups ? "_groups" : "") + (volumes ? "_volumes" : "") +
                             (traces && splats ? "_splatshadows" : "") + (aux_ ? "_aux" : "") +
                             (bake_ ? "_bake" : "");
    // Three ways to trace, and the module carries exactly one: inline rays in
    // a compute kernel, a ray tracing pipeline where there are no inline rays
    // (CUDA, through OptiX), or no rays at all -- direct light alone, which
    // the tracer says of itself.
    const bool pipeline = !traces && device_->caps().rayTracing && device_->caps().accelerationStructure;
    // Splats shadow meshes only where the rays are inline: the pipeline route
    // would need the traversal again as an any hit program, and it carries two.
    const bool withSplats = traces && splats;
    const bool withAux = aux_;
    const std::string source = "import " + programs.module() + ";\n" +
                               (volumes ? std::string("import lrt.volume.medium;\n") : std::string()) +
                               (withSplats ? std::string("import lrt.rt.rt_shadow_packed;\n") : std::string()) + kPrelude +
                               (groups ? kGroups : kNoGroups) + (volumes ? kVolumes : kNoVolumes) +
                               (traces ? kRays : pipeline ? kRaysPipeline : kNoRays) +
                               (withSplats ? kSplats : kNoSplats) + (withAux ? kAux : kNoAux) +
                               (bake_ ? kBake : kNoBake) + kBody +
                               (pipeline ? kEntryRays : kEntryCompute);
    std::vector<std::string> entries{pipeline ? "tracePathsGen" : "tracePaths", "pathDecide"};
    if (pipeline) {
        entries.insert(entries.end(), {"pathNearestMiss", "pathNearestHit", "pathShadowMiss", "pathShadowAnyHit"});
    }
    auto program = library_->loadSource(name, source, entries);
    if (!program) return std::move(program).error();
    if (pipeline) {
        gpu::RayTracingDesc desc;
        desc.module = name;
        desc.rayGen = "tracePathsGen";
        desc.misses = {"pathNearestMiss", "pathShadowMiss"};
        desc.hitGroups = {{"nearest", "pathNearestHit", "", ""}, {"shadow", "", "pathShadowAnyHit", ""}};
        desc.maxRecursion = 1;
        desc.payloadBytes = 64;
        auto rays = gpu::RayTracingKernel::create(*library_, desc);
        if (!rays) return std::move(rays).error();
        rayKernel_.emplace(std::move(*rays));
        kernel_.reset();
    } else {
        auto kernel = gpu::ComputeKernel::create(*library_, name, "tracePaths");
        if (!kernel) return std::move(kernel).error();
        kernel_.emplace(std::move(*kernel));
        rayKernel_.reset();
    }
    auto progressKernel = gpu::ComputeKernel::create(*library_, name, "pathDecide");
    if (!progressKernel) return std::move(progressKernel).error();
    progressKernel_.emplace(std::move(*progressKernel));
    module_ = programs.module();
    groups_ = groups;
    volumes_ = volumes;
    accumulated_ = 0;
    return ok();
}

Result<void> PathTracer::bake(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                              const render::Projection& projection, const MaterialFrame& frame,
                              const PathSettings& settings, const BakePoints& points,
                              render::RenderTargets& out) {
    if (points.rays == nullptr || !points.rays->valid() || points.count == 0 || points.width == 0) {
        return Error(ErrorCode::InvalidArgument, "path tracer: nothing to bake");
    }
    return trace(batch, targets, projection, frame, settings, out, nullptr, &points);
}

Result<void> PathTracer::trace(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                               const render::Projection& projection, const MaterialFrame& frame,
                               const PathSettings& settings, render::RenderTargets& out, PathAux* aux,
                               const BakePoints* bake) {
    if (!kernel_.has_value() && !rayKernel_.has_value()) {
        return Error(ErrorCode::InvalidArgument, "path tracer: no materials set");
    }
    // A bake is dispatched over its points, not over the frame's pixels, and
    // every buffer below is that size.
    const uint32_t width = bake != nullptr ? bake->width : targets.width;
    const uint32_t height = bake != nullptr ? bake->height : targets.height;
    const bool groups = frame.groups.count > 0;
    const bool volumes = frame.volumes != nullptr && frame.volumeCount > 0 && frame.volumes->valid();
    // A cloud that shadows meshes is another kernel: compiled when one
    // arrives, and the kernel without it again when the frame has none.
    const bool wantAux = aux != nullptr;
    // The cloud's tables and the denoiser's guides are a buffer each, and on
    // Metal this kernel has one to spare: with both asked for, the guides win
    // -- the denoiser cannot work without them, and a cloud that shadows
    // nothing is a frame that is merely too bright in one place. Said once.
    bool splats = frame.splatShadows != nullptr && frame.splatShadows->valid();
    if (splats && wantAux && device_->backend() == gpu::Backend::Metal) {
        if (!saidNoRoomForSplats_) {
            saidNoRoomForSplats_ = true;
            lrt::log::warn("path tracer: the denoiser's guides and a cloud's shadows want the same buffer, "
                      "and Metal binds 31: the cloud casts no shadow on meshes this frame");
        }
        splats = false;
    }
    const bool wantBake = bake != nullptr;
    if ((groups != groups_ || volumes != volumes_ || splats != splats_ || wantAux != aux_ ||
         wantBake != bake_) &&
        frame.programs != nullptr) {
        bake_ = wantBake;
        LRT_TRY(setPrograms(*frame.programs, groups, volumes, splats, wantAux));
    }
    const uint32_t groupCount = std::min(frame.groups.count, kMaxLightGroups);
    const uint64_t pixels = uint64_t{width} * height;
    // A bake writes one plane a coefficient into the same buffer.
    const uint64_t planesOut = bake != nullptr ? std::clamp(bake->coefficients, 1u, 16u) : 1u;
    const bool resized = out.width != width || out.height != height || !out.colour.valid() ||
                         out.colour.bytes() < pixels * 16 * planesOut;
    if (resized) {
        gpu::BufferDesc colour;
        colour.bytes = pixels * 16 * planesOut;
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
        out.width = width;
        out.height = height;
    }
    // The sum holds the colour's plane and, with light groups, their sums
    // and their means after it.
    const uint32_t planes = 1 + 2 * groupCount;
    if (resized || width_ != width || height_ != height || !sum_.valid() || planes != sumPlanes_) {
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
        width_ = width;
        height_ = height;
        accumulated_ = 0;
    }
    if (!settings.accumulate) {
        accumulated_ = 0;
    }
    if (aux != nullptr && (aux->width != width || aux->height != height || !aux->valid())) {
        gpu::BufferDesc desc;
        desc.bytes = pixels * 16 * 2;
        desc.elementBytes = 16;
        desc.label = "path.aux";
        auto planes = gpu::Buffer::create(*device_, desc);
        if (!planes) return std::move(planes).error();
        aux->planes = std::move(*planes);
        aux->width = width;
        aux->height = height;
    }
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    const uint32_t samples = std::max(settings.samples, 1u);
    const uint32_t already = accumulated_;
    const auto bindPath = [&](rhi::ShaderCursor cursor) {
        bindMaterialFrame(cursor, frame, projection);
        if (frame.lights != nullptr) {
            frame.lights->bind(cursor);
        }
        if (frame.shadows != nullptr) {
            cursor["scene"].setBinding(frame.shadows);
        }
        if (splats_ && frame.splatShadows != nullptr && frame.splatShadows->valid()) {
            const PackedShadowLayout& where = frame.splatShadows->layout();
            cursor["splatPacked"].setBinding(frame.splatShadows->packed().rhi());
            cursor["splatScene"].setBinding(frame.splatShadows->topLevel());
            cursor["path"]["splatFrames"].setData(where.frames);
            cursor["path"]["splatColours"].setData(where.colours);
            cursor["path"]["splatInstances"].setData(where.instances);
            cursor["path"]["splatIndices"].setData(where.indices);
            cursor["path"]["splatInstanceCount"].setData(where.instanceCount);
            cursor["path"]["splatCut"].setData(frame.splatShadowCut);
        }
        cursor["visibility"].setBinding((*ids).get());
        // Only the bake's own kernel declares the rays.
        if (const rhi::ShaderCursor rays = cursor["bakeRays"]; rays.isValid() && bake != nullptr) {
            rays.setBinding(bake->rays->rhi());
        }
        cursor["sum"].setBinding(sum_.rhi());
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        // A buffer has to be bound either way; without aux the colour stands in
        // and the kernel never writes it.
        // Only the kernel of a frame that asked for the guides declares them.
        if (const rhi::ShaderCursor guides = cursor["aux"]; guides.isValid()) {
            guides.setBinding(aux != nullptr ? aux->planes.rhi() : out.colour.rhi());
        }
        cursor["path"]["writeAux"].setData(uint32_t{aux != nullptr ? 1u : 0u});
        cursor["moments"].setBinding(moments_.rhi());
        if (groups) {
            cursor["groupCount"].setData(groupCount);
        }
        if (volumes) {
            cursor["volumeWords"].setBinding(frame.volumes->rhi());
        }
        cursor["path"]["adaptive"].setData(uint32_t{settings.adaptive ? 1u : 0u});
        cursor["path"]["bakeCount"].setData(bake != nullptr ? std::clamp(bake->coefficients, 1u, 16u) : 1u);
        cursor["path"]["headlight"].setData(uint32_t{settings.headlight ? 1u : 0u});
        cursor["path"]["mis"].setData(uint32_t{settings.mis ? 1u : 0u});
        // The emitting triangles, where the kernel samples them: their table,
        // or a word to bind in its place.
        if (const rhi::ShaderCursor table = cursor["emissiveTable"]; table.isValid()) {
            const bool on = frame.emissive != nullptr && frame.emissive->valid() && frame.emissivePower > 0.0F;
            table.setBinding(on ? frame.emissive->rhi() : moments_.rhi());
            cursor["emissiveOn"].setData(uint32_t{on ? 1u : 0u});
        }
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
        setCamera(cursor["camera"], projection, width, height);
        cursor["path"]["samples"].setData(samples);
        cursor["path"]["bounces"].setData(settings.bounces);
        cursor["path"]["seed"].setData(settings.seed);
        cursor["path"]["accumulated"].setData(already);
        const bool tree = frame.chooseLights && frame.lightBvh && frame.lights != nullptr && frame.lights->hasBvh();
        cursor["path"]["chooseLights"].setData(uint32_t{tree ? 2u : frame.chooseLights ? 1u : 0u});
    };
    if (rayKernel_.has_value()) {
        // A dispatch of rays has no groups: the launch is one ray a pixel of
        // the 16 x 16 blocks the body still walks in quad order.
        const uint32_t groupsAcross = (width + 15) / 16;
        const uint32_t groupsDown = (height + 15) / 16;
        rayKernel_->dispatch(batch, groupsAcross * groupsDown * 256, 1, 1, bindPath);
    } else {
        kernel_->dispatch(batch, {width, height, 1}, bindPath);
    }
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
