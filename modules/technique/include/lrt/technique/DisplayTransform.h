// Copyright (c) 2026 lucabRTrender contributors.
//
// An engine image -- colour, depth, ids or a vector AOV -- onto a display
// texture, on the device (shaders/lrt/technique/display.slang). What lrt view
// shows: a window's surface texture is written directly, nothing read back.
#pragma once

#include <array>
#include <cstdint>

#include <slang-rhi.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::technique {

enum class ViewTransform : uint32_t { Standard = 0, AgX = 1 };
enum class DisplayEncoding : uint32_t { Srgb = 0, Rec709 = 1, DisplayP3 = 2 };

struct DisplaySource {
    enum class Kind : uint32_t { Colour = 0, Depth = 1, Ids = 2, Vector = 3 };
    Kind               kind = Kind::Colour;
    const gpu::Buffer* buffer = nullptr;   ///< float4 (Colour, Vector), float view z (Depth), uint (Ids)
    uint32_t           stride = 1;         ///< entries per pixel
    uint32_t           offset = 0;         ///< which one
    uint32_t           width = 0;
    uint32_t           height = 0;
    bool               bottomRowFirst = true;   ///< as the engine renders
};

struct DisplaySettings {
    ViewTransform           view = ViewTransform::AgX;
    DisplayEncoding         display = DisplayEncoding::Srgb;
    float                   exposure = 0.0F;   ///< stops
    std::array<float, 3>    background{0.0F, 0.0F, 0.0F};   ///< linear
    float                   nearZ = 0.1F;      ///< depth: white
    float                   farZ = 1000.0F;    ///< depth: black
    float                   vectorScale = 0.5F;   ///< vector: rgb * scale + bias (normals by default)
    float                   vectorBias = 0.5F;
};

class DisplayTransform {
public:
    [[nodiscard]] static Result<DisplayTransform> create(gpu::ShaderLibrary& library);

    /// `source` into `output` (a storage-writable texture, top row first) of
    /// `outputWidth` x `outputHeight` -- the source's size, or any other, each
    /// output pixel showing the source pixel under it -- encoded for
    /// `settings.display`. Zero sizes take the source's.
    [[nodiscard]] Result<void> run(gpu::CommandBatch& batch, const DisplaySource& source,
                                   const DisplaySettings& settings, rhi::ITexture* output,
                                   uint32_t outputWidth = 0, uint32_t outputHeight = 0);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel kernel_;
    gpu::Buffer        placeholderFloat4_, placeholderWord_;
};

}   // namespace lrt::technique
