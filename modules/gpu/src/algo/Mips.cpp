// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/algo/Mips.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/gpu/Texture.h"

namespace lrt::gpu {

Result<MipGenerator> MipGenerator::create(ShaderLibrary& library) {
    auto kernel = ComputeKernel::create(library, "lrt/algo/mips", "mipDownsample");
    if (!kernel) return std::move(kernel).error();
    MipGenerator mips;
    mips.downsample_ = std::move(*kernel);
    return mips;
}

Result<void> MipGenerator::generate(CommandBatch& batch, const Texture& texture, bool srgb) const {
    if (texture.desc().type != rhi::TextureType::Texture2D || texture.desc().arrayLength != 1) {
        return Error::make(ErrorCode::Unsupported, "mips of '{}': 2D textures only", texture.desc().label);
    }
    const rhi::Format format = texture.desc().format;
    const bool eightBit = format == rhi::Format::RGBA8Unorm || format == rhi::Format::BGRA8Unorm ||
                          format == rhi::Format::R8Unorm || format == rhi::Format::RG8Unorm ||
                          format == rhi::Format::RGBA8UnormSrgb || format == rhi::Format::BGRA8UnormSrgb;
    for (uint32_t mip = 1; mip < texture.mipCount(); ++mip) {
        auto source = texture.view(mip - 1);
        if (!source) return std::move(source).error();
        auto target = texture.view(mip);
        if (!target) return std::move(target).error();
        downsample_.dispatch(batch, {texture.width(mip), texture.height(mip), 1}, [&](rhi::ShaderCursor cursor) {
            cursor["source"].setBinding((*source).get());
            cursor["target"].setBinding((*target).get());
            cursor["params"]["sourceWidth"].setData(texture.width(mip - 1));
            cursor["params"]["sourceHeight"].setData(texture.height(mip - 1));
            cursor["params"]["width"].setData(texture.width(mip));
            cursor["params"]["height"].setData(texture.height(mip));
            cursor["params"]["srgb"].setData(uint32_t{srgb ? 1u : 0u});
            cursor["params"]["levels"].setData(eightBit ? uint32_t{255} : uint32_t{0});
        });
    }
    return ok();
}

}   // namespace lrt::gpu
