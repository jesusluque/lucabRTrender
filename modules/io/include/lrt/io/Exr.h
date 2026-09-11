// Copyright (c) 2026 lucabRTrender contributors.
//
// OpenEXR in and out, through tinyexr. The file is top row first; the engine's
// images are bottom row first, and both functions turn one into the other.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "lrt/core/Result.h"

namespace lrt::io {

/// Writes linear premultiplied RGBA (bottom row first, as the engine renders)
/// and, when given, a Z channel, to an OpenEXR file with half or float pixels.
[[nodiscard]] Result<void> writeExr(const std::filesystem::path& path, uint32_t width,
                                    uint32_t height, std::span<const float> rgba,
                                    std::span<const float> depth = {}, bool half = true);

struct ExrPixels {
    uint32_t           width = 0;
    uint32_t           height = 0;
    std::vector<float> rgba;   ///< bottom row first, as the engine's images
};

/// Reads an OpenEXR file's R, G, B and A (missing channels: 0, alpha 1).
[[nodiscard]] Result<ExrPixels> readExr(const std::filesystem::path& path);

}   // namespace lrt::io
