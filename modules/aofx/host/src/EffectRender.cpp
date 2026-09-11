// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/aofx/EffectRender.h"

#include <algorithm>

#include "aofx/Effect.h"
#include "gpe/args.h"
#include "gpe/pool.h"
#include "lrt/aofx/EffectRegistry.h"
#include "lrt/aofx/EffectRunner.h"
#include "lrt/gpu_host/Context.h"
#include "lrt/gpu_host/ImageStorage.h"

namespace lrt::aofx_host {
namespace {

/// Matches `ChannelParams` in kernels/channels.slang.
struct ChannelUniforms {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t dstStride = 0;
    uint32_t srcWidth = 0;
    uint32_t srcHeight = 0;
    uint32_t srcStride = 0;
    int32_t  srcOffsetX = 0;
    int32_t  srcOffsetY = 0;
    uint32_t keepMask = 0;
    uint32_t pad0 = 0;
};
static_assert(sizeof(ChannelUniforms) == 40, "must match ChannelParams");

aofx::Rect toRect(const image::PixelRect& rect) {
    return aofx::Rect{rect.x1, rect.y1, rect.x2, rect.y2};
}

double given(const std::vector<aofx::ParamValue>& params, const char* name, double fallback) {
    for (const aofx::ParamValue& param : params) {
        if (param.name == name) {
            return param.number(fallback);
        }
    }
    return fallback;
}

}   // namespace

Result<image::ImagePtr> renderEffect(gpu_host::Context& context, aofx::Effect& effect,
                                     const EffectJob& job) {
    gpe::PooledDevice* device = context.compute();
    gpu_host::ImageStorage* storage = context.sharedStorage();
    if (device == nullptr || storage == nullptr) {
        return Error(ErrorCode::Unsupported, "AOFX effects need a gpe device");
    }

    aofx::EffectDesc desc;
    effect.describe(desc);

    image::PixelRect bounds = job.bounds;
    if (bounds.isEmpty()) {
        if (job.inputs.empty() || job.inputs.front().image == nullptr) {
            return Error(ErrorCode::InvalidArgument, "no output bounds and no input to take them from");
        }
        bounds = job.inputs.front().image->bounds();
    }

    const auto bufferOf = [&](const image::Image& picture) -> Result<aofx::Buffer> {
        const uint64_t id = storage->bufferFor(picture.address());
        if (id == 0) {
            return Error(ErrorCode::InvalidArgument,
                         "an image is on the heap, not on the device: install the context "
                         "before making images");
        }
        aofx::Buffer buffer;
        buffer.device = id;
        buffer.width = picture.bounds().width();
        buffer.height = picture.bounds().height();
        buffer.stride = picture.stride();
        buffer.rect = toRect(picture.bounds());
        return buffer;
    };

    auto output = image::Image::create(bounds);
    if (!output) {
        return std::move(output).error();
    }
    image::ImagePtr out = *output;

    aofx::RenderRequest request;
    request.time = job.time;
    request.instance = job.instance;
    request.renderWindow = toRect(bounds);
    request.outputRod = toRect(bounds);

    const aofx::InputPlane* passThrough = nullptr;
    for (const EffectInput& input : job.inputs) {
        if (input.image == nullptr) {
            continue;
        }
        auto buffer = bufferOf(*input.image);
        if (!buffer) {
            return std::move(buffer).error();
        }
        input.image->syncDevice();
        aofx::InputPlane plane;
        plane.clip = input.clip;
        plane.time = job.time;
        plane.buffer = *buffer;
        plane.rod = buffer->rect;
        plane.values = input.image->attachments();
        request.inputs.push_back(std::move(plane));
    }
    for (const aofx::ClipDesc& clip : desc.inputs) {
        if (!clip.optional && request.input(clip.name) == nullptr) {
            return Error::make(ErrorCode::InvalidArgument, "'{}' needs its '{}' input",
                               desc.identifier, clip.name);
        }
    }
    for (const aofx::ClipDesc& clip : desc.inputs) {
        if (clip.passThrough) {
            passThrough = request.input(clip.name);
        }
    }
    if (passThrough == nullptr && !request.inputs.empty()) {
        passThrough = &request.inputs.front();
    }

    auto outBuffer = bufferOf(*out);
    if (!outBuffer) {
        return std::move(outBuffer).error();
    }
    request.outputs.push_back(aofx::OutputPlane{"Color", *outBuffer});

    // Every declared parameter, given or defaulted, in declaration order.
    for (const aofx::ParamDesc& param : desc.params) {
        aofx::ParamValue value;
        value.name = param.name;
        const auto found = std::find_if(job.params.begin(), job.params.end(),
                                        [&](const aofx::ParamValue& p) { return p.name == param.name; });
        if (found != job.params.end()) {
            value = *found;
        } else if (param.type == aofx::ParamType::String) {
            value.text = param.textDefault;
        } else {
            value.numbers = param.defaults;
        }
        request.params.push_back(std::move(value));
    }

    uint32_t keepMask = 0;
    keepMask |= given(job.params, kChannelRedParam, 1.0) >= 0.5 ? 1u : 0u;
    keepMask |= given(job.params, kChannelGreenParam, 1.0) >= 0.5 ? 2u : 0u;
    keepMask |= given(job.params, kChannelBlueParam, 1.0) >= 0.5 ? 4u : 0u;
    keepMask |= given(job.params, kChannelAlphaParam, 1.0) >= 0.5 ? 8u : 0u;
    if (keepMask == 0) {
        keepMask = 15u;   // none selected means all (openFXplayer Channels.h)
    }

    bool processed = false;
    std::string complaint;
    LRT_TRY(context.run([&] {
        out->clear();
        EffectRunner runner(context);
        request.gpu = &runner;
        processed = effect.process(request);
        if (processed && keepMask != 15u && passThrough != nullptr) {
            ChannelUniforms uniforms;
            uniforms.width = static_cast<uint32_t>(outBuffer->width);
            uniforms.height = static_cast<uint32_t>(outBuffer->height);
            uniforms.dstStride = static_cast<uint32_t>(outBuffer->stride);
            uniforms.srcWidth = static_cast<uint32_t>(passThrough->buffer.width);
            uniforms.srcHeight = static_cast<uint32_t>(passThrough->buffer.height);
            uniforms.srcStride = static_cast<uint32_t>(passThrough->buffer.stride);
            uniforms.srcOffsetX = outBuffer->rect.x1 - passThrough->buffer.rect.x1;
            uniforms.srcOffsetY = outBuffer->rect.y1 - passThrough->buffer.rect.y1;
            uniforms.keepMask = keepMask;
            const aofx::KernelId restore = runner.load(kChannelsKernel);
            processed = runner.run(restore, aofx::Grid{uniforms.width, uniforms.height, 1},
                                   {passThrough->buffer, *outBuffer}, &uniforms, sizeof(uniforms));
        }
        device->sync();
        if (!processed) {
            complaint = EffectRunner::lastComplaint();
        }
    }));
    out->deviceWrote();
    if (!processed) {
        return Error::make(ErrorCode::PluginFailure, "'{}' did not render{}{}", desc.identifier,
                           complaint.empty() ? "" : ": ", complaint);
    }
    for (auto& [id, values] : request.produced) {
        out->attach(id, values);
    }
    return out;
}

}   // namespace lrt::aofx_host
