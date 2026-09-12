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
    uint pad0; uint pad1;
};

Texture2D<uint4>               visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>     sum;          // paths added so far, a pixel
RWStructuredBuffer<float4>     colour;       // their mean
RWStructuredBuffer<float>      depth;
ConstantBuffer<CameraParams>   camera;
StructuredBuffer<LightRecord>  lights;
uniform uint                   lightCount;
ConstantBuffer<PathParams>     path;

/// Hash of everything that makes a sample: the pixel, which path it is, and
/// which bounce. Cheap, and independent enough for a mean to converge.
float random(uint2 pixel, uint sample, uint bounce, uint which) {
    uint h = pixel.x * 73856093u ^ pixel.y * 19349663u ^ (sample + path.seed) * 83492791u ^
             bounce * 2654435761u ^ which * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return float(h >> 8) * (1.0 / 16777216.0);
}

float2 random2(uint2 pixel, uint sample, uint bounce, uint which) {
    return float2(random(pixel, sample, bounce, which), random(pixel, sample, bounce, which + 1u));
}

/// The power heuristic, with the exponent everyone uses.
float misWeight(float a, float b) {
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}
)";

/// Where the device traces, an indirect ray finds the next surface and a
/// shadow ray decides whether a light is reached. Without ray queries there is
/// no bounce to trace, so the tracer is direct light alone and says so.
const char* kRays = R"(
uniform RaytracingAccelerationStructure scene;

struct PathHit {
    uint4 seen;   // (instance + 1, triangle, 0, 0); 0 for nothing
    float t;
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
    } else {
        hit.seen = uint4(0);
        hit.t = 0.0;
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
    uint4 seen;
    float t;
};

PathHit traceNearestFrom(float3 origin, float3 direction, float tMin) {
    PathHit hit;
    hit.seen = uint4(0);
    hit.t = 0.0;
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

/// Rebuilds a surface from the visibility pair as the raster shading does,
/// then evaluates whatever material it wears.
Shaded shadeAt(uint2 pixel, uint4 seen) {
    Shaded out;
    out.valid = false;
    out.depth = 0.0;
    if (seen.x == 0) {
        return out;
    }
    const Surface s = surfaceAt(camera, pixel.x, pixel.y, seen);
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

/// A surface away from the camera: the same reconstruction, but the eye is
/// wherever the ray came from.
Shaded shadeHit(uint2 pixel, uint4 seen, float3 from) {
    Shaded out = shadeAt(pixel, seen);
    if (out.valid) {
        out.toEye = normalize(from - out.inputs.positionWorld);
    }
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
    // A delta light cannot be hit by a sampled direction, so it takes the
    // whole weight; anything else shares it with the material's sampling.
    const float weight = ls.delta ? 1.0 : misWeight(density, stackPdf(sh.stack, sh.toEye, ls.wi));
    return f * ls.radiance * (weight / density);
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
    const uint samples = max(path.samples, 1u);
    float3 total = float3(0.0);
    float  alpha = 0.0;
    float  hitDepth = 0.0;
    for (uint sample = 0; sample < samples; ++sample) {
        Shaded sh = shadeAt(tid, seen);
        if (!sh.valid) {
            continue;
        }
        hitDepth = sh.depth;
        alpha += sh.stack.opacity;
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
            throughput *= ms.weight / ms.pdf;
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
            Shaded next = shadeHit(tid, hit.seen, p);
            if (!next.valid) {
                break;
            }
            // What the bounce found: its own emission weighed against the
            // light sampling that could have found it, and its direct light.
            carried += throughput * next.stack.emission;
            carried += throughput * gatherLight(next, tid, sample, bounce + 1u);
            sh = next;
        }
        total += carried;
    }
    const float3 mean = total / float(samples);
    const float4 added = float4(mean * (alpha / float(samples)), alpha / float(samples));
    const float before = float(path.accumulated);
    const float4 kept = path.accumulated != 0 ? sum[at] : float4(0.0);
    const float4 now = kept + added * float(samples);
    sum[at] = now;
    const float total_samples = before + float(samples);
    colour[at] = total_samples > 0.0 ? now / total_samples : float4(0.0);
    depth[at] = hitDepth;
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
    auto program = library_->loadSource(name, source, {"tracePaths"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "tracePaths");
    if (!kernel) return std::move(kernel).error();
    kernel_.emplace(std::move(*kernel));
    module_ = programs.module();
    accumulated_ = 0;
    return ok();
}

Result<void> PathTracer::trace(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                               const render::Projection& projection, const MaterialFrame& frame,
                               const PathSettings& settings, render::RenderTargets& out) {
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
        width_ = targets.width;
        height_ = targets.height;
        accumulated_ = 0;
    }
    if (!settings.accumulate) {
        accumulated_ = 0;
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
        setCamera(cursor["camera"], projection, targets.width, targets.height);
        cursor["path"]["samples"].setData(samples);
        cursor["path"]["bounces"].setData(settings.bounces);
        cursor["path"]["seed"].setData(settings.seed);
        cursor["path"]["accumulated"].setData(already);
        cursor["path"]["chooseLights"].setData(uint32_t{frame.chooseLights ? 1u : 0u});
        cursor["path"]["power"].setData(frame.lights != nullptr ? frame.lights->power() : 0.0F);
    });
    accumulated_ = already + samples;
    return ok();
}

}   // namespace lrt::technique
