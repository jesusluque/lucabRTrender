// Copyright (c) 2026 lucabRTrender contributors.
//
// Textures and samplers: slang-rhi's, owned, with results instead of codes.
// Engine images stay buffers (float4 per pixel, a kernel can index them); a
// texture is for what the pipeline or a sampler needs as one -- material
// images with mips, render targets, depth.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "lrt/core/Result.h"

namespace lrt::gpu {

class Device;

struct TextureDesc {
    rhi::TextureType  type = rhi::TextureType::Texture2D;
    uint32_t          width = 1;
    uint32_t          height = 1;
    uint32_t          depth = 1;
    uint32_t          arrayLength = 1;
    /// Mip levels; 0 is the whole chain down to 1x1.
    uint32_t          mipCount = 1;
    rhi::Format       format = rhi::Format::RGBA32Float;
    rhi::TextureUsage usage = rhi::TextureUsage::ShaderResource;
    std::string       label;
};

/// How a kernel writes a texel of `format` into a buffer (packing.slang's
/// `lrtPackTexel`): 1 four 8-bit unorms, 2 four halves, 3 four floats, 0 a
/// format no kernel packs. Only for a device whose stores do not convert.
[[nodiscard]] uint32_t packedKindOf(rhi::Format format) noexcept;

/// Levels from `width` x `height` x `depth` down to 1x1x1, each half the last
/// rounded down.
[[nodiscard]] uint32_t mipChain(uint32_t width, uint32_t height, uint32_t depth = 1) noexcept;

class Texture {
public:
    Texture() = default;

    [[nodiscard]] static Result<Texture> create(Device& device, const TextureDesc& desc);

    [[nodiscard]] bool valid() const noexcept { return texture_ != nullptr; }
    [[nodiscard]] rhi::ITexture* rhi() const noexcept { return texture_.get(); }
    [[nodiscard]] const TextureDesc& desc() const noexcept { return desc_; }
    [[nodiscard]] uint32_t width(uint32_t mip = 0) const noexcept;
    [[nodiscard]] uint32_t height(uint32_t mip = 0) const noexcept;
    [[nodiscard]] uint32_t depth(uint32_t mip = 0) const noexcept;
    [[nodiscard]] uint32_t mipCount() const noexcept { return desc_.mipCount; }
    /// Bytes per texel of the format (uncompressed formats only).
    [[nodiscard]] uint32_t texelBytes() const noexcept;

    /// A view of `count` levels from `mip`, every layer.
    [[nodiscard]] Result<rhi::ComPtr<rhi::ITextureView>> view(uint32_t mip, uint32_t count = 1) const;

    /// Uploads one subresource; `texels` holds its rows packed, top row first
    /// as the file had them.
    [[nodiscard]] Result<void> upload(Device& device, uint32_t mip, uint32_t layer,
                                      std::span<const std::byte> texels);

    /// Reads one subresource back, rows packed. For output and tests' counters,
    /// never a frame path.
    [[nodiscard]] Result<std::vector<std::byte>> read(Device& device, uint32_t mip, uint32_t layer) const;

private:
    rhi::ComPtr<rhi::ITexture> texture_;
    rhi::IDevice*              device_ = nullptr;   ///< the device that made it, which outlives it
    TextureDesc                desc_;
};

class Sampler {
public:
    Sampler() = default;
    [[nodiscard]] static Result<Sampler> create(Device& device, const rhi::SamplerDesc& desc);
    [[nodiscard]] bool valid() const noexcept { return sampler_ != nullptr; }
    [[nodiscard]] rhi::ISampler* rhi() const noexcept { return sampler_.get(); }

private:
    rhi::ComPtr<rhi::ISampler> sampler_;
};

}   // namespace lrt::gpu
