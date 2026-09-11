// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/DisplayTransform.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

Result<DisplayTransform> DisplayTransform::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "lrt/technique/display", "displayTransform");
    if (!kernel) return std::move(kernel).error();
    DisplayTransform display;
    display.device_ = &library.device();
    display.kernel_ = std::move(*kernel);
    gpu::BufferDesc four;
    four.bytes = 16;
    four.elementBytes = 16;
    four.label = "display.placeholder4";
    auto a = gpu::Buffer::create(*display.device_, four);
    if (!a) return std::move(a).error();
    gpu::BufferDesc word;
    word.bytes = 4;
    word.elementBytes = 4;
    word.label = "display.placeholder";
    auto b = gpu::Buffer::create(*display.device_, word);
    if (!b) return std::move(b).error();
    display.placeholderFloat4_ = std::move(*a);
    display.placeholderWord_ = std::move(*b);
    return display;
}

Result<void> DisplayTransform::run(gpu::CommandBatch& batch, const DisplaySource& source,
                                   const DisplaySettings& settings, rhi::ITexture* output, uint32_t outputWidth,
                                   uint32_t outputHeight) {
    if (output == nullptr || source.width == 0 || source.height == 0) {
        return Error(ErrorCode::InvalidArgument, "display: nothing to show, or nowhere to show it");
    }
    const bool floats = source.kind == DisplaySource::Kind::Colour || source.kind == DisplaySource::Kind::Vector;
    const bool valid = source.buffer != nullptr && source.buffer->valid();
    // An absent AOV shows as the background: the id mode with nothing drawn.
    const DisplaySource::Kind kind = valid ? source.kind : DisplaySource::Kind::Ids;
    const uint32_t ow = outputWidth != 0 ? outputWidth : source.width;
    const uint32_t oh = outputHeight != 0 ? outputHeight : source.height;
    kernel_.dispatch(batch, {ow, oh, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(valid && floats ? source.buffer->rhi() : placeholderFloat4_.rhi());
        cursor["depth"].setBinding(valid && kind == DisplaySource::Kind::Depth ? source.buffer->rhi()
                                                                              : placeholderWord_.rhi());
        cursor["ids"].setBinding(valid && kind == DisplaySource::Kind::Ids ? source.buffer->rhi()
                                                                          : placeholderWord_.rhi());
        cursor["output"].setBinding(output);
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(source.width);
        p["height"].setData(source.height);
        p["mode"].setData(valid ? static_cast<uint32_t>(kind) : uint32_t{4});
        p["stride"].setData(source.stride);
        p["offset"].setData(source.offset);
        p["view"].setData(static_cast<uint32_t>(settings.view));
        p["display"].setData(static_cast<uint32_t>(settings.display));
        p["flipRows"].setData(uint32_t{source.bottomRowFirst ? 1u : 0u});
        p["exposure"].setData(settings.exposure);
        p["nearZ"].setData(settings.nearZ);
        p["farZ"].setData(settings.farZ);
        p["vectorScale"].setData(settings.vectorScale);
        p["vectorBias"].setData(settings.vectorBias);
        p["backgroundR"].setData(settings.background[0]);
        p["backgroundG"].setData(settings.background[1]);
        p["backgroundB"].setData(settings.background[2]);
        p["outputWidth"].setData(ow);
        p["outputHeight"].setData(oh);
    });
    return ok();
}

}   // namespace lrt::technique
