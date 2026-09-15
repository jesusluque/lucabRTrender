// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/algo/Mips.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/gpu/Texture.h"

namespace lrt::gpu {

Result<MipGenerator> MipGenerator::create(ShaderLibrary& library) {
    auto kernel = ComputeKernel::create(library, "lrt/algo/mips", "mipDownsample");
    if (!kernel) return std::move(kernel).error();
    MipGenerator mips;
    mips.device_ = &library.device();
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
    // Where a store into the texture does not convert (CUDA), each level is
    // packed into a buffer by the kernel and the buffer copied into the level.
    const uint32_t packedKind = device_ != nullptr && !device_->caps().convertingStores ? packedKindOf(format) : 0u;
    if (device_ != nullptr && !device_->caps().convertingStores && packedKind == 0) {
        return Error::make(ErrorCode::Unsupported, "mips of '{}': this device's stores do not convert and no kernel "
                                                   "packs its format", texture.desc().label);
    }
    Buffer scratch;
    if (packedKind != 0 && texture.mipCount() > 1) {
        BufferDesc desc;
        desc.bytes = uint64_t{texture.width(1)} * texture.height(1) * texture.texelBytes();
        desc.elementBytes = 4;
        desc.label = "mips.packed";
        auto made = Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        scratch = std::move(*made);
    }
    for (uint32_t mip = 1; mip < texture.mipCount(); ++mip) {
        auto source = texture.view(mip - 1);
        if (!source) return std::move(source).error();
        auto target = texture.view(mip);
        if (!target) return std::move(target).error();
        downsample_.dispatch(batch, {texture.width(mip), texture.height(mip), 1}, [&](rhi::ShaderCursor cursor) {
            cursor["source"].setBinding((*source).get());
            cursor["target"].setBinding((*target).get());
            cursor["packed"].setBinding(packedKind != 0 ? scratch.rhi() : nullptr);
            cursor["params"]["packedKind"].setData(packedKind);
            cursor["params"]["sourceWidth"].setData(texture.width(mip - 1));
            cursor["params"]["sourceHeight"].setData(texture.height(mip - 1));
            cursor["params"]["width"].setData(texture.width(mip));
            cursor["params"]["height"].setData(texture.height(mip));
            cursor["params"]["srgb"].setData(uint32_t{srgb ? 1u : 0u});
            cursor["params"]["levels"].setData(eightBit ? uint32_t{255} : uint32_t{0});
        });
        if (packedKind != 0) {
            const uint32_t rowPitch = texture.width(mip) * texture.texelBytes();
            batch.encoder()->copyBufferToTexture(texture.rhi(), 0, mip, {0, 0, 0}, scratch.rhi(), 0,
                                                 uint64_t{rowPitch} * texture.height(mip), rowPitch,
                                                 {texture.width(mip), texture.height(mip), 1});
            batch.markDirty();
        }
    }
    return ok();
}

}   // namespace lrt::gpu
