// Copyright (c) 2026 lucabRTrender contributors.
//
// What the CPU hands the GPU: records of float32, and a description of where
// each field is and how it is encoded. The CPU parses and arranges; it never
// decodes a number -- no sigmoid, no exp, no normalisation, no sRGB curve.
// shaders/lrt/scene/splat_decode.slang does all of that.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lrt::io {

struct SplatEncoding {
    uint32_t floatsPerRecord = 0;
    // Offsets, in floats, from the start of a record.
    uint32_t x = 0, y = 0, z = 0;
    uint32_t opacity = 0;
    uint32_t scale0 = 0, scale1 = 0, scale2 = 0;
    uint32_t rotW = 0, rotX = 0, rotY = 0, rotZ = 0;
    uint32_t dc0 = 0, dc1 = 0, dc2 = 0;
    /// First rest coefficient and how many basis functions per colour the file
    /// carries (0, 3, 8 or 15).
    uint32_t restBase = 0;
    uint32_t restPerColour = 0;
    /// 1: all red, then green, then blue (3DGS PLY). 0: rgb per basis (SPZ).
    uint32_t restColourOuter = 1;

    enum class Opacity : uint32_t { Logit = 0, Linear = 1, Byte = 2 };
    enum class Scale : uint32_t { Log = 0, Linear = 1 };
    enum class Colour : uint32_t { ShDc = 0, Linear = 1, Byte = 2 };
    enum class Rotation : uint32_t { Float = 0, Byte = 1 };   ///< byte: (v - 128) / 128
    Opacity  opacity_ = Opacity::Logit;
    Scale    scale_ = Scale::Log;
    Colour   colour = Colour::ShDc;
    Rotation rotation = Rotation::Float;
};

struct RawSplats {
    std::string        source;
    uint32_t           count = 0;
    SplatEncoding      encoding;
    std::vector<float> records;   ///< count * encoding.floatsPerRecord
};

/// Points: x y z, then r g b, as the file gave them.
struct RawPoints {
    std::string        source;
    uint32_t           count = 0;
    /// 0: no colour (white). 1: 8-bit sRGB values 0..255. 2: linear floats.
    uint32_t           colourKind = 0;
    std::vector<float> records;   ///< count * 6
};

}   // namespace lrt::io
