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

/// How a channel's 32-bit words are stored in the file.
enum class ExrChannelType : uint8_t { Half, Float, Uint };

/// One named channel of an image to write: `words` is one 32-bit word a
/// pixel, bottom row first -- a float for Half and Float, an unsigned
/// integer for Uint (an id plane; -1 stays 0xFFFFFFFF).
struct ExrChannel {
    std::string               name;
    ExrChannelType            type = ExrChannelType::Float;
    std::span<const uint32_t> words;
};

/// Writes an image of named channels ("R", "Z", "Neye.x", "primId"...): what
/// a render product with several vars is. The channels are stored in the
/// order the format requires (by name), whatever order they are given in.
[[nodiscard]] Result<void> writeExrChannels(const std::filesystem::path& path, uint32_t width, uint32_t height,
                                            std::span<const ExrChannel> channels,
                                            std::span<const ExrAttribute> attributes = {});

/// A channel read back: its words as the file held them (a Half channel
/// comes back as floats), bottom row first.
struct ExrChannelData {
    std::string           name;
    ExrChannelType        type = ExrChannelType::Float;
    std::vector<uint32_t> words;
};

struct ExrChannels {
    uint32_t                    width = 0;
    uint32_t                    height = 0;
    std::vector<ExrChannelData> channels;   ///< in the file's order (by name)
};

/// Reads every channel of an OpenEXR file, by name.
[[nodiscard]] Result<ExrChannels> readExrChannels(const std::filesystem::path& path);

struct ExrPixels {
    uint32_t                  width = 0;
    uint32_t                  height = 0;
    std::vector<float>        rgba;   ///< bottom row first, as the engine's images
    std::vector<ExrAttribute> attributes;   ///< those the reader does not interpret itself
};

/// Reads an OpenEXR file's R, G, B and A (missing channels: 0, alpha 1).
[[nodiscard]] Result<ExrPixels> readExr(const std::filesystem::path& path);

}   // namespace lrt::io
