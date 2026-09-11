// Copyright (c) 2026 lucabRTrender contributors.
//
// A texture's mip chain from its first level, on the device (slang-rhi has no
// mip generation). Area-weighted, so every level keeps level 0's mean; see
// shaders/lrt/algo/mips.slang.
#pragma once

#include "lrt/core/Result.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::gpu {

class CommandBatch;
class ShaderLibrary;
class Texture;

class MipGenerator {
public:
    [[nodiscard]] static Result<MipGenerator> create(ShaderLibrary& library);

    /// Levels 1.. of `texture` from level 0, queued into `batch`. The texture
    /// needs ShaderResource and UnorderedAccess usage and a writable format
    /// (float, or 8-bit linear; sRGB formats are not writable as storage).
    /// `srgb`: the texels hold sRGB-encoded colour (an 8-bit texture sampled
    /// through an sRGB view); levels average light and store it encoded.
    [[nodiscard]] Result<void> generate(CommandBatch& batch, const Texture& texture, bool srgb = false) const;

private:
    ComputeKernel downsample_;
};

}   // namespace lrt::gpu
