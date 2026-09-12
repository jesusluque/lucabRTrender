// Copyright (c) 2026 lucabRTrender contributors.
//
// lrt view: a window onto a USD stage through the engine's Hydra delegate.
// Frames stay on the device -- the display transform writes the window's
// surface texture, Dear ImGui draws its panels over it -- and nothing comes
// back but a picked pixel's ids.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "lrt/core/Result.h"

namespace lrt::view {

struct ViewOptions {
    std::filesystem::path stage;
    std::string           camera;                   ///< a camera prim; empty, a free camera framing the stage
    std::string           technique = "raster";     ///< "raster" or "rt"
    std::string           visibility = "automatic";   ///< mesh visibility: automatic, raster, rays, bvh
    uint32_t              width = 1600;
    uint32_t              height = 900;
    uint32_t              frames = 0;               ///< stop after this many; 0, when the window closes
    uint32_t              lightSamples = 1;         ///< samples per light per pixel
    bool                  chooseLights = false;     ///< one light a sample, by power
    bool                  visible = true;
    /// Where the last frame goes as it was shown, panels included: an EXR of
    /// display-encoded values. Empty, nowhere.
    std::filesystem::path snapshot;
};

struct ViewStats {
    uint32_t frames = 0;
    uint64_t snapshotLitPixels = 0;   ///< of the snapshot: pixels brighter than the background
    double   medianDrawMs = 0.0;    ///< Hydra and the engine
    double   medianFrameMs = 0.0;   ///< the whole loop, events to present
};

[[nodiscard]] Result<ViewStats> runViewer(const ViewOptions& options);

}   // namespace lrt::view
