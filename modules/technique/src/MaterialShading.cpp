// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/MaterialShading.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

namespace {

const char* kKernelPrelude = R"(
import lrt.light.lights;

struct LightingParams {
    uint samples;
    uint pad0; uint pad1; uint pad2;
};

Texture2D<uint4>              visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>    colour;
RWStructuredBuffer<float>     depth;
ConstantBuffer<CameraParams>  camera;
StructuredBuffer<LightRecord> lights;
uniform uint                  lightCount;
ConstantBuffer<LightingParams> lighting;

/// Two numbers for the i-th sample of light k at a pixel. Stratification and
/// a frame's worth of decorrelation are the path tracer's (M6); here the
/// samples only have to be spread.
float2 sampleAt(uint2 pixel, uint light, uint index) {
    uint h = pixel.x * 73856093u ^ pixel.y * 19349663u ^ light * 83492791u ^ index * 2654435761u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const uint g = h * 1664525u + 1013904223u;
    return float2(float(h >> 8) * (1.0 / 16777216.0), float(g >> 8) * (1.0 / 16777216.0));
}
)";

/// Where the device traces rays, a light is occluded by anything between the
/// shading point and the sample. Cutouts do not open a shadow yet: a sample
/// cut out of visibility still stops a shadow ray.
const char* kShadowRay = R"(
uniform RaytracingAccelerationStructure shadowScene;

bool occluded(float3 p, float3 n, float3 wi, float distance) {
    // Off the surface it sits on, by a distance that grows with the scene, so
    // a grazing ray does not find the triangle it started from. Along the ray
    // as well as the normal, since a normal only says which side the surface
    // faces, not which side the light is on. What this costs is contact: an
    // occluder within the offset is not seen.
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
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(shadowScene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    query.Proceed();
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}
)";

const char* kNoShadowRay = R"(
bool occluded(float3 p, float3 n, float3 wi, float distance) {
    return false;
}
)";

const char* kKernelBody = R"(
[shader("compute")]
[numthreads(16, 16, 1)]
void shadeMaterials(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    // In quad order, so the nodes that want screen derivatives (bump) find
    // their neighbours in the thread's quad.
    const uint2 tid = lrtQuadPixel(group.xy, index);
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        colour[at] = float4(0.0);
        depth[at] = 0.0;
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time);
    const MaterialRecord m = materials[materialRowOf(s)];
    evaluateMaterial(m.function, inputs, m.blob);
    const LobeStack stack = gLrtResult;
    const float3 toEye = normalize(inputs.viewPosition - inputs.positionWorld);
    float3 radiance = stack.emission;
    if (lightCount == 0) {
        // The headlight: unit radiance from the eye, so a white Lambert
        // surface facing it shows 1. What meshes were lit by before there
        // were lights.
        radiance += kPi * stackEval(stack, toEye, toEye);
    } else {
        const uint samples = max(lighting.samples, 1u);
        for (uint k = 0; k < lightCount; ++k) {
            const LightRecord light = lights[k];
            // Light linking: a light reaches only the categories its
            // collection resolved to, and one with no collection reaches all.
            if (!lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi)) {
                continue;
            }
            const bool shadow = (light.flags & kLightShadow) != 0;
            float3 sum = float3(0.0);
            for (uint i = 0; i < samples; ++i) {
                const LightSample ls =
                    sampleLight(light, inputs.positionWorld, inputs.normalWorld, sampleAt(tid, k, i));
                if (!ls.valid) {
                    continue;
                }
                // stackEval carries the cosine, so the estimator is the
                // response times the arriving radiance over the density.
                const float3 f = stackEval(stack, toEye, ls.wi);
                if (!any(f > float3(0.0))) {
                    continue;
                }
                if (shadow && occluded(inputs.positionWorld, inputs.normalWorld, ls.wi, ls.distance)) {
                    continue;
                }
                sum += f * ls.radiance / ls.pdf;
            }
            radiance += sum / float(samples);
        }
    }
    colour[at] = float4(radiance * stack.opacity, stack.opacity);
    depth[at] = s.depth;
}
)";

}   // namespace

Result<MaterialShading> MaterialShading::create(gpu::ShaderLibrary& library) {
    MaterialShading shading;
    shading.library_ = &library;
    shading.device_ = &library.device();
    return shading;
}

Result<void> MaterialShading::setPrograms(const MaterialPrograms& programs) {
    if (programs.module() == module_ && kernel_.has_value()) {
        return ok();
    }
    const bool shadows = device_->caps().rayQuery && device_->caps().accelerationStructure;
    const std::string name = programs.module() + (shadows ? "_shade_shadowed" : "_shade");
    const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude +
                               (shadows ? kShadowRay : kNoShadowRay) + kKernelBody;
    auto program = library_->loadSource(name, source, {"shadeMaterials"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "shadeMaterials");
    if (!kernel) return std::move(kernel).error();
    kernel_.emplace(std::move(*kernel));
    module_ = programs.module();
    return ok();
}

Result<void> MaterialShading::shade(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                    const render::Projection& projection, const MaterialFrame& frame,
                                    render::RenderTargets& out) {
    if (!kernel_.has_value()) {
        return Error(ErrorCode::InvalidArgument, "material shading: no materials set");
    }
    const uint64_t pixels = uint64_t{targets.width} * targets.height;
    if (out.width != targets.width || out.height != targets.height || !out.colour.valid()) {
        gpu::BufferDesc colour;
        colour.bytes = pixels * 16;
        colour.elementBytes = 16;
        colour.label = "materials.colour";
        auto madeColour = gpu::Buffer::create(*device_, colour);
        if (!madeColour) return std::move(madeColour).error();
        gpu::BufferDesc depth;
        depth.bytes = pixels * 4;
        depth.elementBytes = 4;
        depth.label = "materials.depth";
        auto madeDepth = gpu::Buffer::create(*device_, depth);
        if (!madeDepth) return std::move(madeDepth).error();
        out.colour = std::move(*madeColour);
        out.depth = std::move(*madeDepth);
        out.width = targets.width;
        out.height = targets.height;
    }
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    kernel_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
        bindMaterialFrame(cursor, frame, projection);
        // The lights are this kernel's alone.
        if (frame.lights != nullptr) {
            frame.lights->bind(cursor);
            cursor["lighting"]["samples"].setData(frame.samples);
        }
        if (frame.shadows != nullptr) {
            cursor["shadowScene"].setBinding(frame.shadows);
        }
        cursor["visibility"].setBinding((*ids).get());
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        setCamera(cursor["camera"], projection, targets.width, targets.height);
    });
    return ok();
}

}   // namespace lrt::technique
