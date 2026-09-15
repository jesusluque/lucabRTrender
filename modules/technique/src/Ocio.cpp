// Copyright (c) 2026 lucabRTrender contributors.
//
// OpenColorIO as a compiler. A config's display and view become a processor,
// the processor's GPU shader becomes a generated Slang module that imports
// the display kernel's pieces, and its LUTs become textures a kernel fills.
// Every pixel is transformed on the device; the LUT values are computed by
// OCIO on the host, the one exception this route has (docs/decisions.md).
#include "lrt/technique/DisplayTransform.h"

#include <cstdio>
#include <vector>

#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/gpu/Texture.h"

#if LRT_HAVE_OCIO
#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;
#endif

namespace lrt::technique {

struct DisplayTransform::OcioState {
    gpu::ComputeKernel kernel;
    std::string        description;
    struct Lut {
        std::string  name;       ///< the texture as the shader names it
        std::string  sampler;    ///< and its sampler
        gpu::Texture texture;
        gpu::Sampler filter;
    };
    std::vector<Lut> luts;
#if LRT_HAVE_OCIO
    OCIO::ConstProcessorRcPtr processor;   ///< keeps the dynamic properties the uniforms read alive
    OCIO::GpuShaderDescRcPtr  shader;
#endif
};

bool ocioBuilt() noexcept {
    return LRT_HAVE_OCIO != 0;
}

const gpu::ComputeKernel& ocioKernel(const DisplayTransform::OcioState& state) {
    return state.kernel;
}

void bindOcio(const DisplayTransform::OcioState& state, rhi::ShaderCursor cursor) {
    for (const DisplayTransform::OcioState::Lut& lut : state.luts) {
        cursor[lut.name.c_str()].setBinding(lut.texture.rhi());
        cursor[lut.sampler.c_str()].setBinding(lut.filter.rhi());
    }
#if LRT_HAVE_OCIO
    // Dynamic properties, read as they stand now (exposure, contrast, gamma,
    // a grading's values): whatever the processor was built with, by name.
    for (unsigned i = 0; i < state.shader->getNumUniforms(); ++i) {
        OCIO::GpuShaderDesc::UniformData data;
        const char* name = state.shader->getUniform(i, data);
        rhi::ShaderCursor field = cursor[name];
        if (!field.isValid()) {
            continue;
        }
        switch (data.m_type) {
        case OCIO::UNIFORM_DOUBLE: field.setData(static_cast<float>(data.m_getDouble())); break;
        case OCIO::UNIFORM_BOOL: field.setData(int32_t{data.m_getBool() ? 1 : 0}); break;
        case OCIO::UNIFORM_FLOAT3: {
            const OCIO::Float3& v = data.m_getFloat3();
            field.setData(v.data(), sizeof(float) * 3);
            break;
        }
        default: break;   // vectors: refused when the view was compiled
        }
    }
#endif
}

const std::string& DisplayTransform::ocioDescription() const noexcept {
    static const std::string none;
    return ocio_ != nullptr ? ocio_->description : none;
}

#if !LRT_HAVE_OCIO

Result<void> DisplayTransform::setOcio(const OcioView&) {
    return Error(ErrorCode::Unsupported, "no OpenColorIO in this build (scripts/build-ocio.sh)");
}

#else

namespace {

/// OCIO's HLSL samples its LUTs with `Sample`, which takes derivatives a
/// compute kernel does not have; the tables are single level, so an explicit
/// level zero samples the same texels.
std::string explicitLevels(const std::string& text) {
    const std::string needle = ".Sample(";
    std::string out;
    size_t at = 0;
    for (;;) {
        const size_t found = text.find(needle, at);
        if (found == std::string::npos) {
            out.append(text, at, std::string::npos);
            break;
        }
        out.append(text, at, found - at);
        size_t end = found + needle.size();
        int depth = 1;
        while (end < text.size() && depth > 0) {
            depth += text[end] == '(' ? 1 : text[end] == ')' ? -1 : 0;
            ++end;
        }
        out += ".SampleLevel(";
        out.append(text, found + needle.size(), end - 1 - (found + needle.size()));
        out += ", 0.0)";
        at = end;
    }
    return out;
}

uint64_t fnv1a(const std::string& text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char c : text) {
        hash = (hash ^ c) * 1099511628211ULL;
    }
    return hash;
}

const char* kEntry = R"(
[shader("compute")]
[numthreads(16, 16, 1)]
void ocioDisplayTransform(uint3 tid: SV_DispatchThreadID) {
    uint pixel;
    if (!displayPixel(tid, pixel)) {
        return;
    }
    const float3 shown = displayingColour() ? lrtOcio(float4(displayScene(pixel), 1.0)).rgb
                                            : displayOther(pixel, lrtOcio(float4(displayBackground(), 1.0)).rgb);
    displayWrite(tid, shown);
}
)";

const char* kFill = R"(
module lrt_ocio_fill;

StructuredBuffer<float> values;
RWTexture2D<float4>     target2;
RWTexture3D<float4>     target3;
uniform uint            width;
uniform uint            height;
uniform uint            depth;
uniform uint            channels;   // 1 red, 3 rgb

float4 texel(uint i) {
    return channels == 1 ? float4(values[i], 0.0, 0.0, 1.0)
                         : float4(values[i * 3], values[i * 3 + 1], values[i * 3 + 2], 1.0);
}

// OCIO's rows, as it lays them out, into a texture of four channels.
[shader("compute")]
[numthreads(16, 16, 1)]
void ocioFill2(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    target2[tid.xy] = texel(tid.y * width + tid.x);
}

// A 3D LUT's values as OCIO's own OpenGL upload takes them: x fastest, then
// y, then z -- the order its shader text samples in.
[shader("compute")]
[numthreads(8, 8, 8)]
void ocioFill3(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height || tid.z >= depth) {
        return;
    }
    target3[tid] = texel((tid.z * height + tid.y) * width + tid.x);
}
)";

rhi::SamplerDesc samplerFor(OCIO::Interpolation interpolation) {
    rhi::SamplerDesc desc;
    const rhi::TextureFilteringMode mode = interpolation == OCIO::INTERP_NEAREST ? rhi::TextureFilteringMode::Point
                                                                                  : rhi::TextureFilteringMode::Linear;
    desc.minFilter = mode;
    desc.magFilter = mode;
    desc.mipFilter = rhi::TextureFilteringMode::Point;
    desc.addressU = rhi::TextureAddressingMode::ClampToEdge;
    desc.addressV = rhi::TextureAddressingMode::ClampToEdge;
    desc.addressW = rhi::TextureAddressingMode::ClampToEdge;
    return desc;
}

}   // namespace

Result<void> DisplayTransform::setOcio(const OcioView& view) {
    auto state = std::make_shared<OcioState>();
    std::string shaderText;
    std::string label;
    try {
        OCIO::ConstConfigRcPtr config = OCIO::Config::CreateFromFile(view.config.c_str());
        const std::string display = view.display.empty() ? config->getDefaultDisplay() : view.display;
        const std::string viewName = view.view.empty() ? config->getDefaultView(display.c_str()) : view.view;
        OCIO::DisplayViewTransformRcPtr transform = OCIO::DisplayViewTransform::Create();
        transform->setSrc(view.source.c_str());
        transform->setDisplay(display.c_str());
        transform->setView(viewName.c_str());
        OCIO::ConstTransformRcPtr applied = transform;
        if (!view.look.empty()) {
            OCIO::LegacyViewingPipelineRcPtr pipeline = OCIO::LegacyViewingPipeline::Create();
            pipeline->setDisplayViewTransform(transform);
            pipeline->setLooksOverrideEnabled(true);
            pipeline->setLooksOverride(view.look.c_str());
            state->processor = pipeline->getProcessor(config);
        } else {
            state->processor = config->getProcessor(applied);
        }
        state->shader = OCIO::GpuShaderDesc::CreateShaderDesc();
        state->shader->setLanguage(OCIO::GPU_LANGUAGE_HLSL_DX11);
        state->shader->setFunctionName("lrtOcio");
        state->shader->setResourcePrefix("lrt_ocio_");
        state->shader->setAllowTexture1D(false);
        state->processor->getDefaultGPUProcessor()->extractGpuShaderInfo(state->shader);
        shaderText = state->shader->getShaderText();
        label = std::string(config->getName()) + " / " + display + " / " + viewName;
        for (unsigned i = 0; i < state->shader->getNumUniforms(); ++i) {
            OCIO::GpuShaderDesc::UniformData data;
            const char* name = state->shader->getUniform(i, data);
            if (data.m_type != OCIO::UNIFORM_DOUBLE && data.m_type != OCIO::UNIFORM_BOOL &&
                data.m_type != OCIO::UNIFORM_FLOAT3) {
                return Error::make(ErrorCode::Unsupported, "OCIO: uniform '{}' is an array, which the display does not bind",
                                   name);
            }
        }
    } catch (const OCIO::Exception& e) {
        // OCIO reports by throwing; the engine does not, so this is where
        // its exceptions stop.
        return Error::make(ErrorCode::InvalidArgument, "OCIO: {}", e.what());
    }
    state->description = std::string("OCIO ") + OCIO::GetVersion() + ": " + label;

    // The module: the display kernel's pieces, OCIO's function, the entry.
    const std::string source = "import lrt.technique.display;\n" + explicitLevels(shaderText) + kEntry;
    char name[40];
    std::snprintf(name, sizeof(name), "lrt_ocio_%016llx", static_cast<unsigned long long>(fnv1a(source)));
    auto program = library_->loadSource(name, "module " + std::string(name) + ";\n" + source,
                                        {"ocioDisplayTransform"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "ocioDisplayTransform");
    if (!kernel) return std::move(kernel).error();
    state->kernel = std::move(*kernel);

    // The LUTs: OCIO's values uploaded as they are, laid into textures by a kernel.
    const unsigned flat = state->shader->getNumTextures();
    const unsigned cubes = state->shader->getNum3DTextures();
    if (flat + cubes > 0) {
        auto fillProgram = library_->loadSource("lrt_ocio_fill", kFill, {"ocioFill2", "ocioFill3"});
        if (!fillProgram) return std::move(fillProgram).error();
        auto fill2 = gpu::ComputeKernel::create(*library_, "lrt_ocio_fill", "ocioFill2");
        if (!fill2) return std::move(fill2).error();
        auto fill3 = gpu::ComputeKernel::create(*library_, "lrt_ocio_fill", "ocioFill3");
        if (!fill3) return std::move(fill3).error();
        gpu::CommandBatch batch(*device_);
        std::vector<gpu::Buffer> uploads;   // alive until the batch has run
        const auto lutOf = [&](const char* texture, const char* sampler, uint32_t w, uint32_t h, uint32_t d,
                               uint32_t channels, OCIO::Interpolation interpolation,
                               const float* values) -> Result<void> {
            OcioState::Lut lut;
            lut.name = texture;
            lut.sampler = sampler;
            gpu::TextureDesc desc;
            desc.type = d > 1 ? rhi::TextureType::Texture3D : rhi::TextureType::Texture2D;
            desc.width = w;
            desc.height = h;
            desc.depth = d;
            desc.format = rhi::Format::RGBA32Float;
            desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::UnorderedAccess;
            desc.label = texture;
            auto made = gpu::Texture::create(*device_, desc);
            if (!made) return std::move(made).error();
            lut.texture = std::move(*made);
            auto filter = gpu::Sampler::create(*device_, samplerFor(interpolation));
            if (!filter) return std::move(filter).error();
            lut.filter = std::move(*filter);
            gpu::BufferDesc upload;
            upload.bytes = uint64_t{w} * h * d * channels * sizeof(float);
            upload.elementBytes = sizeof(float);
            upload.label = "ocio.lut";
            auto buffer = gpu::Buffer::create(*device_, upload, values);
            if (!buffer) return std::move(buffer).error();
            auto level = lut.texture.view(0);
            if (!level) return std::move(level).error();
            const gpu::ComputeKernel& fill = d > 1 ? *fill3 : *fill2;
            fill.dispatch(batch, {w, h, d}, [&](rhi::ShaderCursor cursor) {
                cursor["values"].setBinding(buffer->rhi());
                cursor[d > 1 ? "target3" : "target2"].setBinding((*level).get());
                cursor["width"].setData(w);
                cursor["height"].setData(h);
                cursor["depth"].setData(d);
                cursor["channels"].setData(channels);
            });
            uploads.push_back(std::move(*buffer));
            state->luts.push_back(std::move(lut));
            return ok();
        };
        for (unsigned i = 0; i < flat; ++i) {
            const char* texture = nullptr;
            const char* sampler = nullptr;
            unsigned w = 0, h = 0;
            OCIO::GpuShaderDesc::TextureType channel = OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL;
            OCIO::GpuShaderDesc::TextureDimensions dimensions = OCIO::GpuShaderDesc::TEXTURE_2D;
            OCIO::Interpolation interpolation = OCIO::INTERP_LINEAR;
            state->shader->getTexture(i, texture, sampler, w, h, channel, dimensions, interpolation);
            const float* values = nullptr;
            state->shader->getTextureValues(i, values);
            LRT_TRY(lutOf(texture, sampler, w, h, 1, channel == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL ? 1u : 3u,
                          interpolation, values));
        }
        for (unsigned i = 0; i < cubes; ++i) {
            const char* texture = nullptr;
            const char* sampler = nullptr;
            unsigned edge = 0;
            OCIO::Interpolation interpolation = OCIO::INTERP_LINEAR;
            state->shader->get3DTexture(i, texture, sampler, edge, interpolation);
            const float* values = nullptr;
            state->shader->get3DTextureValues(i, values);
            LRT_TRY(lutOf(texture, sampler, edge, edge, edge, 3, interpolation, values));
        }
        LRT_TRY(batch.submit(true));
    }
    ocio_ = std::move(state);
    return ok();
}

#endif

}   // namespace lrt::technique
