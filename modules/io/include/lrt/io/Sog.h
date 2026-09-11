// Copyright (c) 2026 lucabRTrender contributors.
//
// SOG -- PlayCanvas's "Spatially Ordered Gaussians": a zip (or a directory) of
// lossless WebP images, one or two per attribute, and a meta.json of ranges
// and 256-entry codebooks. Read here as far as the CPU should go: the archive
// opened, the images decoded to their bytes, the metadata parsed. What those
// bytes mean -- log-domain positions, codebook lookups, smallest-three
// quaternions, the harmonics palette -- is shaders/lrt/scene/sog_decode.slang.
//
// Formulas after PlayCanvas's reference (MIT: playcanvas/engine
// gsplat-sog-data.js, playcanvas/splat-transform), as openFXplayer's
// SogReader ported them; both SOG versions: 1 lerps min/max ranges, 2 looks
// up codebooks.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "lrt/core/Result.h"

namespace lrt::io {

/// An image's bytes exactly as stored: RGBA8 packed r | g << 8 | b << 16 |
/// a << 24, raster order. Never premultiplied -- these channels are indices
/// and packed integers, not colour.
struct SogImage {
    uint32_t              width = 0;
    uint32_t              height = 0;
    std::vector<uint32_t> texels;
    [[nodiscard]] bool empty() const noexcept { return texels.empty(); }
};

struct RawSog {
    std::string source;
    uint32_t    count = 0;
    uint32_t    version = 1;

    SogImage meansL, meansU, quats, scales, sh0;
    /// Higher harmonics: a palette of `shCoefficients` per entry, 64 entries a
    /// row, and each splat's entry index in the labels' red and green.
    SogImage shLabels, shCentroids;
    uint32_t shCoefficients = 0;   ///< 0, 3, 8 or 15

    std::array<float, 3> meansMin{}, meansMax{};
    std::array<float, 3> scalesMin{}, scalesMax{};   ///< version 1
    std::array<float, 4> sh0Min{}, sh0Max{};         ///< version 1
    float                shNMin = 0.0F, shNMax = 0.0F;   ///< version 1
    std::vector<float>   scalesBook, sh0Book, shNBook;   ///< version 2: 256 each
};

/// A .sog bundle, or the meta.json of an unbundled one.
[[nodiscard]] Result<RawSog> readSog(const std::filesystem::path& path);

/// Whether this build can decode SOG's WebP images.
[[nodiscard]] bool sogSupported() noexcept;

}   // namespace lrt::io
