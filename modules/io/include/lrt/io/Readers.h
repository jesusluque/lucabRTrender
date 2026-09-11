// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <filesystem>

#include "lrt/core/Result.h"
#include "lrt/io/RawSplats.h"

namespace lrt::io {

/// A Gaussian splat file by extension: .ply (3DGS), .splat, .spz.
[[nodiscard]] Result<RawSplats> readSplats(const std::filesystem::path& path);

[[nodiscard]] Result<RawSplats> readSplatPly(const std::filesystem::path& path);
[[nodiscard]] Result<RawSplats> readDotSplat(const std::filesystem::path& path);

/// A point file by name: .ply (ascii or binary), .xyz/.txt/.pts/.csv,
/// COLMAP points3D.txt / points3D.bin.
[[nodiscard]] Result<RawPoints> readPoints(const std::filesystem::path& path);

}   // namespace lrt::io
