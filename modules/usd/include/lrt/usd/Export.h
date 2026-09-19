// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <array>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/io/RawSplats.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::usd {

/// WHAT A CLOUD CARRIED BY A SKELETON WRITES BESIDE ITS GAUSSIANS.
///
/// Four joints a gaussian and how much each carries it, the transform out of
/// the cloud's space into the one the joints are measured from, and the
/// joints' own transforms at each of a set of instants. That last is the only
/// thing that changes from frame to frame, and for sixty joints it is four
/// kilobytes a frame -- which is why a cloud that carries its rig is
/// megabytes where one that carries per-frame arrays is hundreds of them.
struct SplatSkinning {
    /// The Skeleton prim's path, for provenance.
    std::string           skeleton;
    /// `(joint, weight)` four times a gaussian, in the order the records are.
    std::vector<float>    influences;
    std::array<float, 16> geomBindTransform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    uint32_t              joints = 0;
    /// The time codes the transforms below were read at.
    std::vector<double>   times;
    /// What those codes are worth. Written on the stage, because a layer that
    /// is silent is read at 24 and stretched under a root that says otherwise.
    double                timeCodesPerSecond = 24.0;
    /// `times.size() * joints * 16` floats, row major as USD holds them.
    std::vector<float>    xforms;

    [[nodiscard]] bool valid() const noexcept {
        return joints > 0 && !times.empty() && !influences.empty() &&
               xforms.size() == times.size() * joints * 16;
    }
};

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
    /// Which way the stage the gaussians came from stood: 'y' or 'z'. They
    /// are in that stage's world space, so a cloud written as Y-up when they
    /// were laid out Z-up lies on its side -- which is what every asset out
    /// of Blender did.
    char     upAxis = 'y';
    /// The rig the cloud is carried by, or nothing. When it is there the
    /// stage takes a time range and the joints' transforms as time samples.
    const SplatSkinning* skinning = nullptr;
};

/// Writes `raw` as a UsdVolParticleField3DGaussianSplat at /World/Splats in a
/// new stage (.usda, .usdc or .usd). The values are computed on the GPU.
[[nodiscard]] Result<void> writeParticleFieldStage(gpu::ShaderLibrary& library,
                                                   const io::RawSplats& raw,
                                                   const std::filesystem::path& path,
                                                   const ExportOptions& options = {});

/// Writes a baked visibility onto the ParticleField at `prim` of the stage at
/// `path`, in place: `primvars:lrt:splat:visibilityParts` (12 floats a part)
/// and `primvars:lrt:splat:visibilityTexels` (two f16 a word), as
/// `lrt visibility` bakes them and `HdLrtParticleField` reads them back.
[[nodiscard]] Result<void> writeVisibility(const std::filesystem::path& path, const std::string& prim,
                                           std::span<const float> parts, std::span<const int32_t> texels,
                                           std::span<const int32_t> partOf, std::span<const int32_t> ambient);

}   // namespace lrt::usd
