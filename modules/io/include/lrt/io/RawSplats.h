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
    /// An offset no record has: the field is not in this file.
    static constexpr uint32_t kNoField = 0xFFFFFFFFU;

    uint32_t floatsPerRecord = 0;
    // Offsets, in floats, from the start of a record.
    uint32_t x = 0, y = 0, z = 0;
    uint32_t opacity = 0;
    uint32_t scale0 = 0, scale1 = 0, scale2 = 0;
    uint32_t rotW = 0, rotX = 0, rotY = 0, rotZ = 0;
    uint32_t dc0 = 0, dc1 = 0, dc2 = 0;
    /// What a gaussian reflects with, where it has it: a cloud converted from
    /// a mesh knows its material's metallic and roughness, and a capture does
    /// not. `kNoField` for a file that carries neither, which is every file a
    /// trainer writes.
    uint32_t metallic = kNoField;
    uint32_t roughness = kNoField;
    /// First rest coefficient and how many basis functions per colour the file
    /// carries (0, 3, 8 or 15).
    uint32_t restBase = 0;
    uint32_t restPerColour = 0;
    /// 1: all red, then green, then blue (3DGS PLY). 0: rgb per basis (SPZ).
    uint32_t restColourOuter = 1;

    enum class Opacity : uint32_t { Logit = 0, Linear = 1, Byte = 2 };
    /// SpzByte: log scale = byte / 16 - 10.
    enum class Scale : uint32_t { Log = 0, Linear = 1, SpzByte = 2 };
    /// SpzByte: SH DC = (byte / 255 - 0.5) / 0.15.
    enum class Colour : uint32_t { ShDc = 0, Linear = 1, Byte = 2, SpzByte = 3 };
    /// Byte: w x y z bytes, (v - 128) / 128.
    /// FirstThree: x y z bytes at rotX..rotZ, v / 127.5 - 1, w from unit length (SPZ v2).
    /// SmallestThree: one uint32 split into its low 16 bits at rotX and high 16
    /// at rotY (a float holds 16 bits exactly and not 32): two bits name the
    /// largest component, three 10-bit signed magnitudes the rest (SPZ v3+).
    enum class Rotation : uint32_t { Float = 0, Byte = 1, FirstThree = 2, SmallestThree = 3 };
    /// Byte: (v - 128) / 128.
    enum class Rest : uint32_t { Float = 0, Byte = 1 };
    Opacity  opacity_ = Opacity::Logit;
    Scale    scale_ = Scale::Log;
    Colour   colour = Colour::ShDc;
    Rotation rotation = Rotation::Float;
    Rest     rest = Rest::Float;
    /// Positions are multiplied by this: 2^-fractionalBits for SPZ's fixed point.
    float    positionScale = 1.0F;
    /// The file is right-up-back (SPZ) and the engine reads splats as PLY
    /// writes them, right-down-front: y and z of positions, rotations and
    /// harmonics change sign.
    bool     flipYZ = false;
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
