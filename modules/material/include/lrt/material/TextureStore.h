// Copyright (c) 2026 lucabRTrender contributors.
//
// The textures materials sample, on the device. A file is decoded by Hio on
// the host (decompression, as the rules allow) and its bytes go up whole; a
// kernel expands its channels and flips it to v-up (texture_decode.slang), and
// its mip chain is made on the device -- averaged as light for sRGB colour,
// which is sampled through an sRGB view. UDIM sets resolve to a slot per tile.
// Materials reach all of it through one table (lrt/material/texture_table).
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/Texture.h"
#include "lrt/gpu/algo/Mips.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::material {

/// How a file's components are to be read: MaterialX's and UsdUVTexture's
/// `colorSpace`/`sourceColorSpace`.
enum class ColourSpace : uint8_t {
    Auto,   ///< the file says: 8-bit images are sRGB unless tagged otherwise, float ones linear
    Raw,    ///< data (normals, roughness): no decoding
    Srgb,   ///< sRGB-encoded colour
};

enum class Wrap : uint8_t { Repeat, Clamp, Mirror, Black };
enum class Filter : uint8_t { Linear, Nearest };

struct TextureInfo {
    std::string path;
    ColourSpace space = ColourSpace::Auto;
    bool        loaded = false;
    bool        udim = false;
    uint32_t    width = 0;    ///< the first tile's, for a UDIM set
    uint32_t    height = 0;
    std::string error;        ///< why it did not load
};

class TextureStore {
public:
    [[nodiscard]] static Result<std::unique_ptr<TextureStore>> create(gpu::ShaderLibrary& library);

    /// The id of `path` -- a resolved file path, or one holding "<UDIM>" --
    /// read as `space`. Loaded at the next commit; until then, and if it
    /// cannot be, samples report it missing and materials use their defaults.
    uint32_t request(const std::string& path, ColourSpace space = ColourSpace::Auto);

    /// The sampler slot for these modes, shared by every texture asking the same.
    uint32_t sampler(Wrap s, Wrap t, Filter filter = Filter::Linear);

    /// Decodes and uploads what was requested since the last commit, mips
    /// made on the device. Returns how many files loaded.
    [[nodiscard]] Result<size_t> commit();

    /// Binds the table under `table` (a TextureTable parameter block).
    void bind(rhi::ShaderCursor table) const;

    [[nodiscard]] const TextureInfo& info(uint32_t id) const { return entries_[id].info; }
    [[nodiscard]] uint32_t count() const noexcept { return static_cast<uint32_t>(entries_.size()); }
    /// The texture in `slot`, for tests.
    [[nodiscard]] const gpu::Texture& slotTexture(uint32_t slot) const { return slots_[slot].texture; }

private:
    struct Slot {
        gpu::Texture                   texture;
        rhi::ComPtr<rhi::ITextureView> view;   ///< sRGB for 8-bit colour, else the texture's own
    };
    struct Entry {
        TextureInfo           info;
        std::vector<uint32_t> tiles;   ///< a UDIM set: slot + 1 per cell of 1001..1100, 0 where none
        uint32_t              slot = 0;
        uint32_t              udimBase = 0;
        bool                  pending = true;
    };

    TextureStore() = default;
    [[nodiscard]] Result<uint32_t> loadFile(const std::string& path, ColourSpace space, TextureInfo& info);
    [[nodiscard]] Result<void> writeRecords();

    gpu::Device*                                          device_ = nullptr;
    gpu::ComputeKernel                                    decode_;
    std::unique_ptr<gpu::MipGenerator>                    mips_;
    std::vector<Entry>                                    entries_;
    std::map<std::pair<std::string, ColourSpace>, uint32_t> ids_;
    std::vector<Slot>                                     slots_;
    std::vector<gpu::Sampler>                             samplers_;
    std::map<std::tuple<Wrap, Wrap, Filter>, uint32_t>    samplerIds_;
    gpu::Buffer                                           records_;
    gpu::Buffer                                           udim_;
};

}   // namespace lrt::material
