// Copyright (c) 2026 lucabRTrender contributors.
//
// DecodeParams (shaders/lrt/scene/splat_encoding.slang), set by name from an
// io::SplatEncoding. One place, for every kernel that reads raw splat records:
// the loader's validate and decode, and the USD export.
#pragma once

#include <cstdint>

#include <slang-rhi/shader-cursor.h>

#include "lrt/io/RawSplats.h"

namespace lrt::scene {

/// Fills `cursor["params"]`.
void setDecodeParams(rhi::ShaderCursor cursor, const io::SplatEncoding& encoding, uint32_t count,
                     uint32_t base, uint32_t keepPerColour, uint32_t shWords);

}   // namespace lrt::scene
