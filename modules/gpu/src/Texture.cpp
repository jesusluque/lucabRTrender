// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/Texture.h"

#include <algorithm>
#include <cstring>

#include "lrt/gpu/Device.h"

namespace lrt::gpu {

uint32_t mipChain(uint32_t width, uint32_t height, uint32_t depth) noexcept {
    uint32_t levels = 1;
    uint32_t largest = std::max({width, height, depth});
    while (largest > 1) {
        largest /= 2;
        ++levels;
    }
    return levels;
}

Result<Texture> Texture::create(Device& device, const TextureDesc& desc) {
    if (desc.width == 0 || desc.height == 0 || desc.depth == 0 || desc.arrayLength == 0) {
        return Error::make(ErrorCode::InvalidArgument, "texture '{}' with no texels", desc.label);
    }
    Texture texture;
    texture.device_ = device.rhi();
    texture.desc_ = desc;
    texture.desc_.mipCount = desc.mipCount == 0 ? mipChain(desc.width, desc.height, desc.depth)
                                                : std::min(desc.mipCount, mipChain(desc.width, desc.height, desc.depth));
    rhi::TextureDesc r;
    r.type = desc.type;
    r.size = {desc.width, desc.height, desc.depth};
    r.arrayLength = desc.arrayLength;
    r.mipCount = texture.desc_.mipCount;
    r.format = desc.format;
    r.usage = desc.usage | rhi::TextureUsage::CopyDestination | rhi::TextureUsage::CopySource;
    r.label = texture.desc_.label.c_str();
    if (SLANG_FAILED(device.rhi()->createTexture(r, nullptr, texture.texture_.writeRef())) ||
        texture.texture_ == nullptr) {
        return Error::make(ErrorCode::DeviceFailure, "cannot create texture '{}' ({}x{}x{}, {} mips)", desc.label,
                           desc.width, desc.height, desc.depth, texture.desc_.mipCount);
    }
    return texture;
}

uint32_t Texture::width(uint32_t mip) const noexcept {
    return std::max<uint32_t>(desc_.width >> mip, 1);
}
uint32_t Texture::height(uint32_t mip) const noexcept {
    return std::max<uint32_t>(desc_.height >> mip, 1);
}
uint32_t Texture::depth(uint32_t mip) const noexcept {
    return desc_.type == rhi::TextureType::Texture3D ? std::max<uint32_t>(desc_.depth >> mip, 1) : 1;
}

uint32_t Texture::texelBytes() const noexcept {
    const rhi::FormatInfo& info = rhi::getFormatInfo(desc_.format);
    return info.blockSizeInBytes / std::max<uint32_t>(info.pixelsPerBlock, 1);
}

Result<rhi::ComPtr<rhi::ITextureView>> Texture::view(uint32_t mip, uint32_t count) const {
    rhi::TextureViewDesc desc;
    desc.format = desc_.format;
    desc.subresourceRange.layer = 0;
    desc.subresourceRange.layerCount = desc_.arrayLength;
    desc.subresourceRange.mip = mip;
    desc.subresourceRange.mipCount = count;
    rhi::ComPtr<rhi::ITextureView> view;
    if (SLANG_FAILED(device_->createTextureView(texture_.get(), desc, view.writeRef()))) {
        return Error::make(ErrorCode::DeviceFailure, "cannot view mip {} of texture '{}'", mip, desc_.label);
    }
    return view;
}

Result<void> Texture::upload(Device& device, uint32_t mip, uint32_t layer, std::span<const std::byte> texels) {
    const uint64_t rowPitch = uint64_t{width(mip)} * texelBytes();
    const uint64_t slicePitch = rowPitch * height(mip);
    if (mip >= desc_.mipCount || layer >= desc_.arrayLength || texels.size() < slicePitch * depth(mip)) {
        return Error::make(ErrorCode::InvalidArgument, "upload outside texture '{}'", desc_.label);
    }
    rhi::SubresourceData data;
    data.data = texels.data();
    data.rowPitch = rowPitch;
    data.slicePitch = slicePitch;
    rhi::SubresourceRange range;
    range.layer = layer;
    range.layerCount = 1;
    range.mip = mip;
    range.mipCount = 1;
    rhi::ComPtr<rhi::ICommandEncoder> encoder = device.queue()->createCommandEncoder();
    if (SLANG_FAILED(encoder->uploadTextureData(texture_.get(), range, {0, 0, 0},
                                                {width(mip), height(mip), depth(mip)}, &data, 1))) {
        return Error::make(ErrorCode::DeviceFailure, "cannot upload to texture '{}'", desc_.label);
    }
    device.beforeSubmit();
    if (SLANG_FAILED(device.queue()->submit(encoder->finish()))) {
        return Error(ErrorCode::DeviceFailure, "texture upload submit failed");
    }
    return ok();
}

Result<std::vector<std::byte>> Texture::read(Device& device, uint32_t mip, uint32_t layer) const {
    device.waitIdle();
    rhi::ComPtr<ISlangBlob> blob;
    rhi::SubresourceLayout layout;
    if (SLANG_FAILED(device.rhi()->readTexture(texture_.get(), layer, mip, blob.writeRef(), &layout)) ||
        blob == nullptr) {
        return Error::make(ErrorCode::DeviceFailure, "cannot read texture '{}'", desc_.label);
    }
    const uint64_t rowBytes = uint64_t{width(mip)} * texelBytes();
    std::vector<std::byte> out(rowBytes * height(mip) * depth(mip));
    const auto* from = static_cast<const std::byte*>(blob->getBufferPointer());
    for (uint64_t z = 0; z < depth(mip); ++z) {
        for (uint64_t y = 0; y < height(mip); ++y) {
            // Rows at the layout's pitch, which may pad past the texels.
            std::memcpy(out.data() + (z * height(mip) + y) * rowBytes,
                        from + z * layout.slicePitch + y * layout.rowPitch, rowBytes);
        }
    }
    return out;
}

Result<Sampler> Sampler::create(Device& device, const rhi::SamplerDesc& desc) {
    Sampler sampler;
    if (SLANG_FAILED(device.rhi()->createSampler(desc, sampler.sampler_.writeRef()))) {
        return Error(ErrorCode::DeviceFailure, "cannot create a sampler");
    }
    return sampler;
}

}   // namespace lrt::gpu
