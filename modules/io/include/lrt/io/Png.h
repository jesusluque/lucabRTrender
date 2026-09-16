// Copyright (c) 2026 lucabRTrender contributors.
//
// A PNG of what a kernel already decided: eight-bit RGBA rows, top row first,
// straight into the file. The pixels are the display transform's -- tone
// mapping, the view transform and the encoding all happen on the device -- so
// what is left here is a container: a header, zlib's deflate, and a CRC. The
// same class of work as reading a PLY header, and no arithmetic on any pixel
// beyond the filter byte PNG puts at the start of each row.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

#include "lrt/core/Result.h"

namespace lrt::io {

/// `rgba` holds `width * height * 4` bytes, row 0 at the top.
[[nodiscard]] Result<void> writePng(const std::filesystem::path& path, uint32_t width, uint32_t height,
                                    std::span<const uint8_t> rgba);

/// The same bytes as a PNG in memory, for a caller that sends it rather than
/// saves it (the MCP server answers with the image).
[[nodiscard]] Result<std::vector<uint8_t>> encodePng(uint32_t width, uint32_t height,
                                                     std::span<const uint8_t> rgba);

}   // namespace lrt::io
