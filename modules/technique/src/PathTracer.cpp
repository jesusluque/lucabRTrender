// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/PathTracer.h"

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

struct PathParams {
    uint samples;      // paths a pixel this call
    uint bounces;      // indirect bounces after the first hit
    uint seed;         // which samples these are
    uint accumulated;  // paths a pixel already in `sum`
    uint chooseLights;
    float power;
    uint writeAux;     // 1: write the first hit's albedo and normal
    uint adaptive;     // 1: a converged pixel takes no more paths
    float errorTarget; // relative standard error of the mean a pixel stops at
    uint minSamples;   // and not before this many
    uint pad2; uint pad3;
};

Texture2D<uint4>               visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>     sum;          // paths added so far, a pixel
RWStructuredBuffer<float4>     colour;       // their mean
RWStructuredBuffer<float>      depth;
RWStructuredBuffer<float4>     auxAlbedo;    // written when path.writeAux
RWStructuredBuffer<float4>     auxNormal;
RWStructuredBuffer<float>      sumSquares;   // the luminance's second moment, a pixel
RWStructuredBuffer<uint>       done;         // 1 once adaptive sampling stopped the pixel

static const float3 kPathLuminance = float3(0.2126, 0.7152, 0.0722);
ConstantBuffer<CameraParams>   camera;
StructuredBuffer<LightRecord>  lights;
uniform uint                   lightCount;
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

PathHit traceNearestFrom(float3 origin, float3 direction, float tMin) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = 3.0e38;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
    query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, ray);
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

bool pathOccluded(float3 p, float3 n, float3 wi, float distance, uint shadowCategory) {
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
        query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
        query.Proceed();
        return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    }
    for (uint step = 0; step < 16; ++step) {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
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

PathHit traceNearestFrom(float3 origin, float3 direction, float tMin) {
    PathHit hit;
    hit.seen = uint4(0);
    hit.t = 0.0;
    hit.barycentrics = float2(0.0);
    return hit;
}

bool pathOccluded(float3 p, float3 n, float3 wi, float distance, uint shadowCategory) {
    return false;
}

static const bool kTraces = false;
)";

/// The surface a hit lands on, its material evaluated, and the light it
/// gathers: shared by the camera's hit and the bounce's, so the two are shaded
/// by the same rules.
const char* kBody = R"(
struct Shaded {
    LobeStack      stack;
    MaterialInputs inputs;
    float3         toEye;
    float          depth;
    uint           categoriesLo;   // the instance's, for a light's link
    uint           categoriesHi;
    bool           valid;
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
    return out;
}

/// The camera's hit, rebuilt from the visibility pair as the raster shading
/// does, so the two shade the same point.
Shaded shadeAt(uint2 pixel, uint4 seen) {
    Shaded out;
    out.valid = false;
    out.depth = 0.0;
    if (seen.x == 0) {
        return out;
    }
    return shadeSurface(pixel, surfaceAt(camera, pixel.x, pixel.y, seen));
}

/// A surface away from the camera: the same reconstruction, but the eye is
/// wherever the ray came from.
/// A bounce's hit, rebuilt from where the ray met the triangle -- not from
/// the pixel. It used to call shadeAt, which re-intersects the *camera's* ray
/// with the triangle the bounce found: a point that is not on the bounce ray
/// at all, with barycentrics and a normal to match. The closed box read 5 pi
/// times its geometric series while that stood.
Shaded shadeHit(uint2 pixel, PathHit hit, float3 from, float3 direction) {
    Shaded out;
    out.valid = false;
    out.depth = 0.0;
    if (hit.seen.x == 0) {
        return out;
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
    out = shadeSurface(pixel, surfaceFromWeights(hit.seen, weights, viewDirection));
    out.toEye = normalize(from - out.inputs.positionWorld);
    return out;
}

/// One light, sampled and weighed: next event estimation, with the density of
/// the material's own sampling folded in by the power heuristic so the two
/// strategies do not double count.
float3 gatherLight(Shaded sh, uint2 pixel, uint sample, uint bounce) {
    if (lightCount == 0) {
        return float3(0.0);
    }
    const float pick = random(pixel, sample, bounce, 11u);
    const LightChoice choice = chooseLight(lights, lightCount, path.power, pick);
    if (!choice.valid) {
        return float3(0.0);
    }
    const LightRecord light = lights[choice.index];
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
        pathOccluded(sh.inputs.positionWorld, sh.inputs.normalWorld, ls.wi, ls.distance, light.shadowCategory)) {
        return float3(0.0);
    }
    const float density = ls.pdf * choice.probability;
    // No weight. The two strategies cover disjoint sets of emitters, so there
    // is nothing to share: next event estimation covers the analytic lights of
    // the light table, and sampling the material covers emissive geometry. A
    // light in the table has no geometry for a sampled direction to hit, and a
    // bounce ray that escapes breaks without gathering the dome it passed
    // through -- so a weight here would scale this estimator down and nothing
    // would pay the remainder back. Measured before it was removed: a white
    // furnace under an imageless dome, where the dome's density and the
    // Lambert lobe's are the same function, read 0.5453 of what unweighted
    // NEE reads.
    //
    // Real MIS belongs with mesh lights, where the two strategies genuinely
    // overlap; it needs a "does this direction reach light k, and with what
    // radiance" beside lightPdf, which does not exist yet.
    return f * ls.radiance / density;
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
    if (path.accumulated == 0) {
        done[at] = 0;   // a frame of its own: no flag from the last one
    } else if (path.adaptive != 0 && done[at] != 0) {
        return;   // converged: its mean stands
    }
    const uint samples = max(path.samples, 1u);
    // The camera's hit does not depend on the sample: rebuilt and its material
    // evaluated once, not once a path. What the denoiser wants of it is written
    // here too, since it is the same for every sample.
    const Shaded first = shadeAt(tid, seen);
    if (path.writeAux != 0) {
        auxAlbedo[at] = first.valid ? float4(stackAlbedo(first.stack, first.toEye), 1.0) : float4(0.0);
        auxNormal[at] = first.valid ? float4(first.inputs.normalWorld, 1.0) : float4(0.0);
    }
    float3 total = float3(0.0);
    float  alpha = 0.0;
    float  hitDepth = 0.0;
    float  squares = 0.0;   // sum of each sample's luminance squared
    for (uint sample = 0; sample < samples && first.valid; ++sample) {
        Shaded sh = first;
        hitDepth = sh.depth;
        // The first hit's, kept before the bounces: `sh` walks on to whatever
        // the path finds, and it is this surface's opacity the pixel carries.
        const float opacity = sh.stack.opacity;
        alpha += opacity;
        float3 carried = sh.stack.emission + gatherLight(sh, tid, sample, 0u);
        float3 throughput = float3(1.0);
        // The bounces. Each one samples the material, traces where it points,
        // and gathers that surface's light through what the path has kept.
        for (uint bounce = 0; kTraces && bounce < path.bounces; ++bounce) {
            const LobeSample ms = stackSample(sh.stack, sh.toEye, float3(random2(tid, sample, bounce, 5u),
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
            const float3 p = sh.inputs.positionWorld;
            const float3 n = sh.inputs.normalWorld;
            const float scale = max(1.0, length(p));
            const float3 away = dot(n, ms.wi) < 0.0 ? -n : n;
            const PathHit hit = traceNearestFrom(p + (away + ms.wi) * (1.0e-3 * scale), ms.wi, 1.0e-3 * scale);
            if (hit.seen.x == 0) {
                break;
            }
            Shaded next = shadeHit(tid, hit, p, ms.wi);
            if (!next.valid) {
                break;
            }
            // What the bounce found: its own emission weighed against the
            // light sampling that could have found it, and its direct light.
            carried += throughput * next.stack.emission;
            carried += throughput * gatherLight(next, tid, sample, bounce + 1u);
            sh = next;
        }
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
    // The luminance's second moment. Whether the pixel stops is decided by
    // pathDecide, after the pass, where its neighbours' spread can be read
    // without a race.
    const float keptSquares = path.accumulated != 0 ? sumSquares[at] : 0.0;
    sumSquares[at] = keptSquares + squares;
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
    const float second = sumSquares[at] / n;
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
    if (seen.x == 0) {
        done[at] = 1;   // nothing drawn converges at once, and is not counted
        return;
    }
    InterlockedAdd(progress[0], 1u);
    if (done[at] != 0) {
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
        done[at] = 1;
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
    if (programs.module() == module_ && kernel_.has_value()) {
        return ok();
    }
    const bool traces = device_->caps().rayQuery && device_->caps().accelerationStructure;
    const std::string name = programs.module() + (traces ? "_path_traced" : "_path_direct");
    const std::string source =
        "import " + programs.module() + ";\n" + kPrelude + (traces ? kRays : kNoRays) + kBody;
    auto program = library_->loadSource(name, source, {"tracePaths", "pathDecide"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "tracePaths");
    if (!kernel) return std::move(kernel).error();
    kernel_.emplace(std::move(*kernel));
    auto progressKernel = gpu::ComputeKernel::create(*library_, name, "pathDecide");
    if (!progressKernel) return std::move(progressKernel).error();
    progressKernel_.emplace(std::move(*progressKernel));
    module_ = programs.module();
    accumulated_ = 0;
    return ok();
}

Result<void> PathTracer::trace(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                               const render::Projection& projection, const MaterialFrame& frame,
                               const PathSettings& settings, render::RenderTargets& out, PathAux* aux) {
    if (!kernel_.has_value()) {
        return Error(ErrorCode::InvalidArgument, "path tracer: no materials set");
    }
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
    if (resized || width_ != targets.width || height_ != targets.height || !sum_.valid()) {
        gpu::BufferDesc sum;
        sum.bytes = pixels * 16;
        sum.elementBytes = 16;
        sum.label = "path.sum";
        auto made = gpu::Buffer::create(*device_, sum);
        if (!made) return std::move(made).error();
        sum_ = std::move(*made);
        gpu::BufferDesc squares;
        squares.bytes = pixels * 4;
        squares.elementBytes = 4;
        squares.label = "path.squares";
        auto madeSquares = gpu::Buffer::create(*device_, squares);
        if (!madeSquares) return std::move(madeSquares).error();
        sumSquares_ = std::move(*madeSquares);
        gpu::BufferDesc done;
        done.bytes = pixels * 4;
        done.elementBytes = 4;
        done.label = "path.done";
        auto madeDone = gpu::Buffer::create(*device_, done);
        if (!madeDone) return std::move(madeDone).error();
        done_ = std::move(*madeDone);
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
        desc.bytes = pixels * 16;
        desc.elementBytes = 16;
        desc.label = "path.albedo";
        auto albedo = gpu::Buffer::create(*device_, desc);
        if (!albedo) return std::move(albedo).error();
        desc.label = "path.normal";
        auto normal = gpu::Buffer::create(*device_, desc);
        if (!normal) return std::move(normal).error();
        aux->albedo = std::move(*albedo);
        aux->normal = std::move(*normal);
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
        cursor["auxAlbedo"].setBinding(aux != nullptr ? aux->albedo.rhi() : out.colour.rhi());
        cursor["auxNormal"].setBinding(aux != nullptr ? aux->normal.rhi() : out.colour.rhi());
        cursor["path"]["writeAux"].setData(uint32_t{aux != nullptr ? 1u : 0u});
        cursor["sumSquares"].setBinding(sumSquares_.rhi());
        cursor["done"].setBinding(done_.rhi());
        cursor["path"]["adaptive"].setData(uint32_t{settings.adaptive ? 1u : 0u});
        cursor["path"]["errorTarget"].setData(settings.errorTarget);
        cursor["path"]["minSamples"].setData(settings.minSamples);
        setCamera(cursor["camera"], projection, targets.width, targets.height);
        cursor["path"]["samples"].setData(samples);
        cursor["path"]["bounces"].setData(settings.bounces);
        cursor["path"]["seed"].setData(settings.seed);
        cursor["path"]["accumulated"].setData(already);
        cursor["path"]["chooseLights"].setData(uint32_t{frame.chooseLights ? 1u : 0u});
        cursor["path"]["power"].setData(frame.lights != nullptr ? frame.lights->power() : 0.0F);
    });
    accumulated_ = already + samples;
    lastErrorTarget_ = settings.errorTarget;
    lastMinSamples_ = settings.minSamples;
    return ok();
}

Result<PathProgress> PathTracer::progress(const VisibilityTargets& targets) {
    if (!progressKernel_.has_value() || !done_.valid()) {
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
        cursor["sumSquares"].setBinding(sumSquares_.rhi());
        cursor["done"].setBinding(done_.rhi());
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
