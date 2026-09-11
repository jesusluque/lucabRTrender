// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <filesystem>

#include "lrt/core/Result.h"
#include "lrt/io/RawSplats.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::usd {

struct ExportOptions {
    uint32_t maxDegree = 3;
    /// Adds /World/Camera framing the cloud, so the stage renders as it is.
    bool     addCamera = true;
    /// COLMAP-trained clouds are y-down; turn them over on the prim's xform.
    double   rotateXDegrees = 0.0;
};

/// Writes `raw` as a UsdVolParticleField3DGaussianSplat at /World/Splats in a
/// new stage (.usda, .usdc or .usd). The values are computed on the GPU.
[[nodiscard]] Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library,
                                                   const io::RawSplats& raw,
                                                   const std::filesystem::path& path,
                                                   const ExportOptions& options = {});

}   // namespace lrt::usd
