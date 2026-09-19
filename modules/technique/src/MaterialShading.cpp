// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/MaterialShading.h"

#include <algorithm>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

namespace {

const char* kKernelPrelude = R"(
import lrt.light.lights_image;
import lrt.light.light_bvh;

struct LightingParams {
    uint  samples;
    uint  chooseLights;
    uint  pad0;
    uint  pad1;
};

Texture2D<uint4>              visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>    colour;
RWStructuredBuffer<float>     depth;
ConstantBuffer<CameraParams>  camera;
StructuredBuffer<LightRecord> lights;
uniform uint                  lightNodeBase;
uniform uint                  lightTreeNodes;
uniform uint                  lightUnboundedCount;
uniform uint                  lightCount;
ConstantBuffer<LightingParams> lighting;

/// Two numbers for the i-th sample of light k at a pixel. Stratification and
/// a frame's worth of decorrelation are the path tracer's (M6); here the
/// samples only have to be spread.
uint pcgHashShade(uint input) {
    const uint state = input * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Pixel, light, sample and dimension folded through the hash in turn -- the
/// path tracer's construction. The second number used to be one LCG step of
/// the first, which ties the pair to a lattice.
float2 sampleAt(uint2 pixel, uint light, uint index) {
    uint key = pcgHashShade(pixel.y * 65536u + pixel.x);
    key = pcgHashShade(key + light);
    key = pcgHashShade(key + index);
    const uint a = pcgHashShade(key + 0u);
    const uint b = pcgHashShade(key + 1u);
    return float2(float(a >> 8) * (1.0 / 16777216.0), float(b >> 8) * (1.0 / 16777216.0));
}
)";

/// Where the device traces rays, a light is occluded by anything between the
/// shading point and the sample. Cutouts do not open a shadow yet: a sample
/// cut out of visibility still stops a shadow ray.
/// The light groups: declared only in the kernel of a frame that has them,
/// so a frame without compiles exactly the kernel it always did.
const char* kGroups = R"(
static const bool kLightGroups = true;
RWStructuredBuffer<float4>    groupColour;   // groupCount planes, a pixel each
uniform uint                  groupCount;
void writeGroups(uint at, uint pixels, float3 groups[8], float opacity) {
    for (uint g = 0; g < groupCount && g < 8; ++g) {
        groupColour[g * pixels + at] = float4(groups[g] * opacity, opacity);
    }
}
)";

const char* kNoGroups = R"(
static const bool kLightGroups = false;
void writeGroups(uint at, uint pixels, float3 groups[8], float opacity) {}
)";

const char* kShadowRay = R"(
uniform RaytracingAccelerationStructure shadowScene;

/// Whether the instance a shadow ray found casts this light's shadow: shadow
/// linking, resolved the same way light linking is, by one bit.
bool castsShadow(uint category, uint instance) {
    return lightLinked(category, instances[instance].categoriesLo, instances[instance].categoriesHi);
}

bool occluded(float3 p, float3 n, float3 wi, float distance, uint shadowCategory) {
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
    if (shadowCategory == kLightUnlinked) {
        // Nothing to ask of the occluder: the first hit is the answer.
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(shadowScene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF,
                             ray);
        query.Proceed();
        return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    }
    // Linked: walk on past whatever does not cast this light's shadow, as a
    // cutout walks past what its opacity removed. Sixteen is the same bound.
    for (uint step = 0; step < 16; ++step) {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(shadowScene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
        query.Proceed();
        if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            return false;
        }
        if (castsShadow(shadowCategory, query.CommittedInstanceID())) {
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
)";

const char* kNoShadowRay = R"(
bool occluded(float3 p, float3 n, float3 wi, float distance, uint shadowCategory) {
    return false;
}
)";

const char* kKernelBody = R"(
/// Lights at infinity -- a dome, a distant light -- are the ones the raster
/// can also meet along a lobe's own sample, since nothing but occlusion stands
/// between: with the same shadow ray, or none where the light casts none, the
/// two strategies see the same light and are weighed against each other. A
/// light whose shadow links leave some occluders out keeps light sampling
/// alone, since a lobe sample would have to ask the same of every occluder.
bool atInfinity(LightRecord l) {
    return l.kind == kLightDome || l.kind == kLightDistant;
}
bool rasterWeighs(LightRecord l) {
    return atInfinity(l) && ((l.flags & kLightShadow) == 0 || l.shadowCategory == kLightUnlinked);
}
/// Light k's probability of being chosen, as the frame chooses: one for every
/// light at every pixel, its power share, or the light tree's.
float rasterChoice(uint k, float3 p, float3 n) {
    if (lighting.chooseLights == 2) {
        return lightPdfChoiceAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount, lights, k, p, n);
    }
    if (lighting.chooseLights == 1) {
        const float power = lights[lightCount - 1].cumulative;
        if (!(power > 0.0)) {
            return 0.0;
        }
        const float before = k == 0 ? 0.0 : lights[k - 1].cumulative;
        return max(lights[k].cumulative - before, 1.0e-9) / power;
    }
    return 1.0;
}
/// The power heuristic between n_a samples at density a and n_b at density b.
float rasterMis(float na, float a, float nb, float b) {
    const float x = na * a;
    const float y = nb * b;
    return x * x / max(x * x + y * y, 1.0e-30);
}
static const uint kLobeLookups = 32;
/// Above this peak density (per steradian) a lobe is narrow enough for the
/// raster to meet lights at infinity along its samples as well as by light
/// sampling. A Phong lobe peaks at (e + 1) / 2 pi, so 4 is an exponent near 24,
/// roughly a GGX alpha of 0.35. Below it light sampling keeps the whole
/// weight, and the shadow ray it traces: a diffuse floor's dome shadows are
/// light sampling's alone.
static const float kNarrowPeak = 4.0;
/// The density both strategies are weighed by: a Phong lobe about the mirror
/// direction with the stack's own peak density, zero for a broad stack and
/// below the surface. Weights only have to sum to one where both can sample,
/// not to use the true density -- and the true density is not asked for
/// near a shadow ray (the note in the kernel).
float lobeProxyPdf(float3 wi, float3 mirror, float3 up, float peak) {
    if (!(peak > kNarrowPeak) || dot(wi, up) <= 0.0) {
        return 0.0;
    }
    const float exponent = max(2.0 * kPi * peak - 1.0, 0.0);
    return (exponent + 1.0) / (2.0 * kPi) * pow(max(dot(wi, mirror), 0.0), exponent);
}

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
    const uint pixels = camera.width * camera.height;
    float3 groups[8];
    for (uint g = 0; g < 8; ++g) groups[g] = float3(0.0);
    if (seen.x == 0) {
        colour[at] = float4(0.0);
        depth[at] = 0.0;
        writeGroups(at, pixels, groups, 0.0);
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time);
    const MaterialRecord m = materials[materialRowOf(s)];
    evaluateMaterial(m.function, inputs, m.blob);
    // The lobe stack is read where the material left it, in thread memory,
    // and never copied into a local: with a local copy live across the
    // shadow ray's intersector call, this kernel wrote rows of garbage in
    // blocks of half a threadgroup on an Apple M5 Pro -- clean under Metal
    // shader validation, clean without the trace, clean with the copy gone.
    // The path tracer, which keeps its stack in a struct it passes on, never
    // showed it. Measured, not understood: a live-state problem across the
    // intersector in the Metal compiler is the reading that fits.
#define stack gLrtResult
    const float3 toEye = normalize(inputs.viewPosition - inputs.positionWorld);
    float3 radiance = stack.emission;
    // What light sampling cannot reach, or reaches badly, is met along the
    // lobes' own samples instead: a dome is sampled over the hemisphere above
    // the normal, so light through the surface (glass) never meets it; a
    // delta lobe (smooth glass, a mirror) answers no light sample; a glossy
    // metal's narrow lobe is almost never found by a dome's samples. Without
    // this, the chess set's glass pawn heads and polished rims drew black.
    //
    // All of it here, before the first shadow ray, and none of it live after:
    // this kernel writes garbage on Metal when material state is live across
    // the intersector (the note below). Measured twice more with this very
    // loop -- drawn after the light loops, a raster frame after a path traced
    // one differed from a fresh one in 3 runs of 4; drawn before but kept in
    // arrays for shadow rays after, the unoccluded floor's shadowed frame
    // differed from its unshadowed one by a thousand words, differently each
    // run. So a lobe's sample traces no shadow ray of its own: a reflection
    // sees the sky whether or not something stands in the way, as an
    // environment map's does. Weighed against light sampling, which does
    // trace, that shows only where the lobe is narrow enough to take the
    // weight: glass, mirrors, polished metal.
    const uint lightSampleCount = max(lighting.samples, 1u);
    const uint lobeCount = lightCount != 0 ? min(lightSampleCount, kLobeLookups) : 0u;
    const float3 lobeUp = dot(toEye, inputs.normalWorld) < 0.0 ? -inputs.normalWorld : inputs.normalWorld;
    const float3 lobeMirror = reflect(-toEye, lobeUp);
    const float lobePeak = lobeCount != 0 ? stackPdf(stack, toEye, lobeMirror) : 0.0;
    if (lobeCount != 0) {
        const float3 n0 = inputs.normalWorld;
        const float eyeSide = dot(toEye, n0);
        float3 lobeSum = float3(0.0);
        for (uint j = 0; j < lobeCount; ++j) {
            const float2 ua = sampleAt(tid, lightCount + 1, j);
            const float2 ub = sampleAt(tid, lightCount + 2, j);
            const LobeSample ms = stackSample(stack, toEye, float3(ua, ub.x));
            if (!ms.valid || !any(ms.weight > float3(0.0))) {
                continue;
            }
            const bool through = dot(ms.wi, n0) * eyeSide < 0.0;
            for (uint k = 0; k < lightCount; ++k) {
                const LightRecord light = lights[k];
                if (!atInfinity(light) ||
                    !lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi)) {
                    continue;
                }
                const bool weighs = rasterWeighs(light);
                // Light sampling has a light it weighs, and a delta lobe never;
                // through the surface only the dome, which light sampling does
                // not reach there, and a distant light through a delta lobe.
                if (through) {
                    if (light.kind != kLightDome && !ms.delta) {
                        continue;
                    }
                } else if (!weighs && !ms.delta) {
                    continue;
                }
                const LightHit lh = lightHitImaged(light, inputs.positionWorld, ms.wi);
                if (!lh.valid) {
                    continue;
                }
                float weight = 1.0;
                if (!ms.delta && !through) {
                    const float proxy = lobeProxyPdf(ms.wi, lobeMirror, lobeUp, lobePeak);
                    if (!(proxy > 0.0)) {
                        continue;   // a broad lobe: light sampling has all of it
                    }
                    const float lightDensity = rasterChoice(k, inputs.positionWorld, n0) *
                                               lightPdfImaged(light, inputs.positionWorld, n0, ms.wi);
                    weight = rasterMis(float(lobeCount), proxy, float(lightSampleCount), lightDensity);
                }
                const float3 arrived = weight * ms.weight * lh.radiance;
                lobeSum += arrived;
                if (kLightGroups && light.group != 0 && light.group <= 8) {
                    groups[light.group - 1] += arrived / float(lobeCount);
                }
            }
        }
        radiance += lobeSum / float(lobeCount);
    }
    if (lightCount == 0) {
        // The headlight: unit radiance from the eye, so a white Lambert
        // surface facing it shows 1. What meshes were lit by before there
        // were lights.
        radiance += kPi * stackEval(stack, toEye, toEye);
    } else if (lighting.chooseLights != 0) {
        // One light a sample, in proportion to its power: the cost of a pixel
        // stops growing with the number of lights, and the density it was
        // chosen with is divided back out.
        const uint samples = max(lighting.samples, 1u);
        float3 sum = float3(0.0);
        for (uint i = 0; i < samples; ++i) {
            const float2 u = sampleAt(tid, 0, i);
            const float pick = sampleAt(tid, 1, i).x;
            const LightChoice choice =
                lighting.chooseLights == 2
                    ? chooseLightAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount,
                                     inputs.positionWorld, inputs.normalWorld, pick)
                    : chooseLight(lights, lightCount, pick);
            if (!choice.valid) {
                continue;
            }
            const LightRecord light = lights[choice.index];
            if (!lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi)) {
                continue;
            }
            const LightSample ls = sampleLightImaged(light, inputs.positionWorld, inputs.normalWorld, u);
            if (!ls.valid) {
                continue;
            }
            const float3 f = stackEval(stack, toEye, ls.wi);
            if (!any(f > float3(0.0))) {
                continue;
            }
            // Weighed by the proxy, never the stack: asked for the stack's
            // density in a loop that traces, this kernel writes garbage.
            float weight = 1.0;
            if (!ls.delta && rasterWeighs(light)) {
                weight = rasterMis(float(samples), ls.pdf * choice.probability, float(lobeCount),
                                   lobeProxyPdf(ls.wi, lobeMirror, lobeUp, lobePeak));
            }
            if ((light.flags & kLightShadow) != 0 &&
                occluded(inputs.positionWorld, inputs.normalWorld, ls.wi, ls.distance, light.shadowCategory)) {
                continue;
            }
            const float3 arrived = weight * f * ls.radiance / (ls.pdf * choice.probability);
            sum += arrived;
            if (kLightGroups && light.group != 0 && light.group <= 8) {
                groups[light.group - 1] += arrived / float(samples);
            }
        }
        radiance += sum / float(samples);
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
                    sampleLightImaged(light, inputs.positionWorld, inputs.normalWorld, sampleAt(tid, k, i));
                if (!ls.valid) {
                    continue;
                }
                // stackEval carries the cosine, so the estimator is the
                // response times the arriving radiance over the density.
                const float3 f = stackEval(stack, toEye, ls.wi);
                if (!any(f > float3(0.0))) {
                    continue;
                }
                // Weighed by the proxy, as above.
                float weight = 1.0;
                if (!ls.delta && rasterWeighs(light)) {
                    weight = rasterMis(float(samples), ls.pdf, float(lobeCount),
                                       lobeProxyPdf(ls.wi, lobeMirror, lobeUp, lobePeak));
                }
                if (shadow && occluded(inputs.positionWorld, inputs.normalWorld, ls.wi, ls.distance,
                                       light.shadowCategory)) {
                    continue;
                }
                sum += weight * f * ls.radiance / ls.pdf;
            }
            radiance += sum / float(samples);
            if (kLightGroups && light.group != 0 && light.group <= 8) {
                groups[light.group - 1] += sum / float(samples);
            }
        }
    }
    // A cutout's sample survived its lot in the visibility pass: it is there
    // whole. Any other opacity is still blended, as displayOpacity is.
    const float coverage = (m.flags & kMaterialCutout) != 0 ? 1.0 : stack.opacity;
    colour[at] = float4(radiance * coverage, coverage);
    depth[at] = s.depth;
    writeGroups(at, pixels, groups, coverage);
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
    return setPrograms(programs, groups_);
}

Result<void> MaterialShading::setPrograms(const MaterialPrograms& programs, bool groups) {
    if (programs.module() == module_ && groups == groups_ && kernel_.has_value()) {
        return ok();
    }
    const bool shadows = device_->caps().rayQuery && device_->caps().accelerationStructure;
    const std::string name =
        programs.module() + (shadows ? "_shade_shadowed" : "_shade") + (groups ? "_groups" : "");
    const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude +
                               (groups ? kGroups : kNoGroups) + (shadows ? kShadowRay : kNoShadowRay) + kKernelBody;
    auto program = library_->loadSource(name, source, {"shadeMaterials"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "shadeMaterials");
    if (!kernel) return std::move(kernel).error();
    kernel_.emplace(std::move(*kernel));
    module_ = programs.module();
    groups_ = groups;
    return ok();
}

Result<void> MaterialShading::shade(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                    const render::Projection& projection, const MaterialFrame& frame,
                                    render::RenderTargets& out) {
    if (!kernel_.has_value()) {
        return Error(ErrorCode::InvalidArgument, "material shading: no materials set");
    }
    const bool groups = frame.groups.count > 0;
    if (groups && (frame.groups.colour == nullptr || !frame.groups.colour->valid())) {
        return Error(ErrorCode::InvalidArgument, "material shading: light groups asked for without their buffer");
    }
    if (groups != groups_ && frame.programs != nullptr) {
        LRT_TRY(setPrograms(*frame.programs, groups));
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
            const bool tree = frame.chooseLights && frame.lightBvh && frame.lights->hasBvh();
            cursor["lighting"]["chooseLights"].setData(uint32_t{tree ? 2u : frame.chooseLights ? 1u : 0u});
        }
        if (frame.shadows != nullptr) {
            cursor["shadowScene"].setBinding(frame.shadows);
        }
        cursor["visibility"].setBinding((*ids).get());
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        if (groups) {
            cursor["groupColour"].setBinding(frame.groups.colour->rhi());
            cursor["groupCount"].setData(std::min(frame.groups.count, kMaxLightGroups));
        }
        setCamera(cursor["camera"], projection, targets.width, targets.height);
    });
    return ok();
}

}   // namespace lrt::technique
