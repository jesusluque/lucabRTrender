// Copyright (c) 2026 lucabRTrender contributors.
//
// The PLY header grammar, ported from openFXplayer plugins/splat/PlyHeader.h.
// Offsets are computed from the header, never assumed: a trained cloud has 45
// f_rest_* properties between the colour and the opacity, and a reader that
// assumed offsets would read harmonics as opacity and still show a scene.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "lrt/core/Result.h"

namespace lrt::io {

struct PlyProperty {
    std::string name;
    std::string type;
    size_t      offset = 0;   ///< bytes from the start of a record
    size_t      size = 0;
    bool        isFloat = false;
};

struct PlyElement {
    std::string              name;
    size_t                   count = 0;
    size_t                   stride = 0;
    bool                     hasList = false;
    std::vector<PlyProperty> properties;

    [[nodiscard]] const PlyProperty* find(std::string_view property) const;
};

struct PlyHeader {
    std::string             format;
    bool                    ascii = false;
    bool                    binaryLittleEndian = false;
    size_t                  bytes = 0;   ///< header length, end_header line included
    std::vector<PlyElement> elements;

    [[nodiscard]] const PlyElement* element(std::string_view name) const;
    /// Byte offset of `name`'s first record in a binary file.
    [[nodiscard]] Result<size_t> binaryOffsetOf(std::string_view name) const;
};

[[nodiscard]] Result<PlyHeader> parsePlyHeader(std::string_view text);

/// One property of a binary record as a double.
[[nodiscard]] double readPlyValue(const char* record, const PlyProperty& property) noexcept;

}   // namespace lrt::io
