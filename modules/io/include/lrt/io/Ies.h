// Copyright (c) 2026 lucabRTrender contributors.
//
// IESNA LM-63 photometric profiles, read as authored: the vertical and
// horizontal angle lists and the candela table, the multiplier and the
// photometric type. Nothing is normalised, resampled or mirrored here --
// each of those is arithmetic on the data, and the shader does them where
// it samples. The CPU reads a file.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

#include "lrt/core/Result.h"

namespace lrt::io {

struct IesProfile {
    std::vector<float> vertical;     ///< degrees, ascending; type C: 0 is straight down
    std::vector<float> horizontal;   ///< degrees; one entry means rotational symmetry
    std::vector<float> candela;      ///< horizontal-major: for each horizontal angle, one value per vertical
    float              multiplier = 1.0F;
    uint32_t           photometricType = 1;   ///< 1 C, 2 B, 3 A, as LM-63 numbers them
};

[[nodiscard]] Result<IesProfile> readIes(const std::filesystem::path& path);
/// The same from the text, for a profile a test writes.
[[nodiscard]] Result<IesProfile> parseIes(std::string_view text);

}   // namespace lrt::io
