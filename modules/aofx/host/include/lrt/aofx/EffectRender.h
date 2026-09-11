// Copyright (c) 2026 lucabRTrender contributors.
//
// Rendering one AOFX effect over images: the request, the GPU thread, the
// channel restore, the attachments. What openFXplayer's RenderEngine does for
// an AOFX node, without the graph around it.
#pragma once

#include <string>
#include <vector>

#include "aofx/Types.h"
#include "lrt/core/Result.h"
#include "lrt/image/Image.h"

namespace aofx {
class Effect;
}

namespace lrt::gpu_host {
class Context;
}

namespace lrt::aofx_host {

struct EffectInput {
    std::string     clip;     ///< the ClipDesc name
    image::ImagePtr image;
};

struct EffectJob {
    std::vector<EffectInput>      inputs;
    /// Given values; anything the effect declares and is not given takes its
    /// default. Channel switches (aofx.channel.*) default to all on.
    std::vector<aofx::ParamValue> params;
    /// Empty means the first input's bounds.
    image::PixelRect              bounds;
    double                        time = 0.0;
    std::string                   instance = "lrt";
    /// Frames the host is prepared to be behind (RenderRequest::bufferFrames,
    /// ABI 22): the depth of a live jitter buffer. Zero for a render with none.
    int                           bufferFrames = 0;
};

/// Renders `effect`. Images must live on the context's device (the context's
/// storage installed); a heap image is refused rather than silently uploaded.
[[nodiscard]] Result<image::ImagePtr> renderEffect(gpu_host::Context& context,
                                                   aofx::Effect& effect, const EffectJob& job);

}   // namespace lrt::aofx_host
