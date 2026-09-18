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
    /// Writes `primvars:lrt:splat:relight = 1` (LrtSplatLightingAPI): the
    /// colours are an albedo the scene's lights are to light, not radiance
    /// somebody captured. True for a cloud converted from a mesh, false for a
    /// capture, which carries the light it was shot under.
    bool     relight = false;
    /// Writes `primvars:lrt:splat:litBody = 1`: the colours are the light on
    /// the material's body, not an albedo, so a frame that relights this
    /// cloud adds the polish and nothing else. What `lrt mesh2splat` bakes.
    bool     litBody = false;
};

/// Writes `raw` as a UsdVolParticleField3DGaussianSplat at /World/Splats in a
/// new stage (.usda, .usdc or .usd). The values are computed on the GPU.
[[nodiscard]] Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library,
                                                   const io::RawSplats& raw,
                                                   const std::filesystem::path& path,
                                                   const ExportOptions& options = {});

}   // namespace lrt::usd
