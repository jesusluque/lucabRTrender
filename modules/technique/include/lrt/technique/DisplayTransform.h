// Copyright (c) 2026 lucabRTrender contributors.
//
// An engine image -- colour, depth, ids or a vector AOV -- onto a display
// texture, on the device (shaders/lrt/technique/display.slang). What lrt view
// shows: a window's surface texture is written directly, nothing read back.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <slang-rhi.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/technique/Aces2.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::technique {

/// Standard clips; AgX is Sobotka's sigmoid; Aces2 is the Academy's 2.0
/// output transform (aces2.slang), limited to the display's primaries; Ocio
/// is the OpenColorIO display and view last given to `setOcio`, which
/// encodes for its display itself (`DisplaySettings::display` is ignored).
enum class ViewTransform : uint32_t { Standard = 0, AgX = 1, Aces2 = 2, Ocio = 3 };
/// sRGB, BT.1886 and Display P3 encode for an 8-bit surface; LinearP3
/// leaves linear P3 with 1.0 at the display's reference white, for a
/// float surface with extended range (EDR): values above 1.0 are the
/// headroom, which ACES 2.0 fills up to `peakLuminance`.
enum class DisplayEncoding : uint32_t { Srgb = 0, Rec709 = 1, DisplayP3 = 2, LinearP3 = 3 };

/// An OpenColorIO config's display and view, applied to the engine's linear
/// Rec.709. OCIO is used as a compiler: its shader text becomes a generated
/// Slang module and its LUTs textures, so every pixel is still a kernel's --
/// the LUT values are the one thing the host computes (docs/decisions.md).
struct OcioView {
    std::string config = "ocio://studio-config-latest";   ///< a path, or one of OCIO's built-in configs
    std::string source = "lin_rec709_scene";              ///< the colour space the engine renders in, as the config names it
    std::string display;                                  ///< empty: the config's default
    std::string view;                                     ///< empty: the display's default
    std::string look;                                     ///< empty: none beyond the view's
};

/// Whether this build has OpenColorIO (LRT_HAVE_OCIO).
[[nodiscard]] bool ocioBuilt() noexcept;

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
    /// ACES 2.0's peak, in nits: 100 for a standard display; the reference
    /// white times the headroom for one with extended range.
    float                   peakLuminance = 100.0F;
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

    /// Compiles `view` for ViewTransform::Ocio: the config read, the
    /// processor's GPU shader generated, its module loaded and its LUTs
    /// uploaded. Unsupported without OpenColorIO; an error naming what the
    /// config refused otherwise. A second call replaces the first.
    [[nodiscard]] Result<void> setOcio(const OcioView& view);
    /// "OCIO 2.5.2: <config> / <display> / <view>", empty before setOcio.
    [[nodiscard]] const std::string& ocioDescription() const noexcept;

    struct OcioState;

private:
    gpu::ShaderLibrary* library_ = nullptr;
    std::shared_ptr<OcioState> ocio_;
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel kernel_;
    gpu::Buffer        placeholderFloat4_, placeholderWord_;
    Aces2Tables        aces_;   ///< built the first frame ACES 2.0 shows, and again when its peak or primaries change

public:
    /// The ACES 2.0 tables as the last run left them, for a check that reads them.
    [[nodiscard]] const Aces2Tables& aces() const noexcept { return aces_; }
};

}   // namespace lrt::technique
