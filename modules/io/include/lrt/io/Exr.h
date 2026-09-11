// Copyright (c) 2026 lucabRTrender contributors.
//
// OpenEXR in and out, through tinyexr. The file is top row first; the engine's
// images are bottom row first, and both functions turn one into the other.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "lrt/core/Result.h"

namespace lrt::io {

/// A header attribute beyond the ones the image itself implies.
struct ExrAttribute {
    std::string          name;
    std::string          type;    ///< OpenEXR's type name
    std::vector<uint8_t> value;   ///< its bytes, as the file stores them (little-endian)

    /// OpenEXR's TimeCode: SMPTE 12M's packed time and flags, and user bits.
    static ExrAttribute timecode(std::string name, uint32_t timeAndFlags, uint32_t userData = 0);
    static ExrAttribute rational(std::string name, int32_t numerator, uint32_t denominator);
    static ExrAttribute text(std::string name, const std::string& value);
    static ExrAttribute float64(std::string name, double value);
};

/// Writes linear premultiplied RGBA (bottom row first, as the engine renders)
/// and, when given, a Z channel, to an OpenEXR file with half or float pixels.
[[nodiscard]] Result<void> writeExr(const std::filesystem::path& path, uint32_t width,
                                    uint32_t height, std::span<const float> rgba,
                                    std::span<const float> depth = {}, bool half = true,
                                    std::span<const ExrAttribute> attributes = {});

struct ExrPixels {
    uint32_t                  width = 0;
    uint32_t                  height = 0;
    std::vector<float>        rgba;   ///< bottom row first, as the engine's images
    std::vector<ExrAttribute> attributes;   ///< those the reader does not interpret itself
};

/// Reads an OpenEXR file's R, G, B and A (missing channels: 0, alpha 1).
[[nodiscard]] Result<ExrPixels> readExr(const std::filesystem::path& path);

}   // namespace lrt::io
