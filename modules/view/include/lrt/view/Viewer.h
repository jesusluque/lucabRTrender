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
    /// The path traced technique. A window keeps gathering paths while the
    /// camera is still and starts again when it moves; `pathTotal` is when
    /// the frame counts as converged, which is when a denoise runs.
    uint32_t              pathSamples = 1;          ///< paths a pixel each frame
    uint32_t              pathBounces = 4;          ///< bounces after the first hit
    uint32_t              pathTotal = 64;           ///< paths a pixel it counts as converged at
    bool                  denoise = false;          ///< denoise once converged
    /// A sky and a sun in the session layer when the stage authors no lights
    /// (StageRenderer::setDefaultLights). Without them a stage like the chess
    /// set or Kitchen_set is lit from the eye, and the path tracer has
    /// nothing to show that the raster does not.
    bool                  defaultLights = true;
    /// Extended dynamic range: a float surface in linear P3, ACES 2.0 with
    /// the screen's peak as its peak. On a standard display the same as off.
    bool                  edr = false;
    /// An OpenColorIO view to start with: any of these set compiles the
    /// config's display and view into the display kernel (empty config:
    /// OCIO's built-in studio config; empty display or view: its defaults).
    std::string           ocioConfig;
    std::string           ocioDisplay;
    std::string           ocioView;
    bool                  visible = true;
    /// Start with the timeline playing, as the Play button does.
    bool                  play = false;
    /// Where the last frame goes as it was shown, panels included: an EXR of
    /// display-encoded values. Empty, nowhere.
    std::filesystem::path snapshot;
};

struct ViewStats {
    uint32_t frames = 0;
    uint64_t snapshotLitPixels = 0;   ///< of the snapshot: pixels brighter than the background
    double   medianDrawMs = 0.0;    ///< Hydra and the engine
    double   medianFrameMs = 0.0;   ///< the whole loop, events to present
    double   lastTime = 0.0;        ///< the USD time the last frame drew
    uint32_t distinctTimes = 0;     ///< how many different times the frames drew
};

[[nodiscard]] Result<ViewStats> runViewer(const ViewOptions& options);

}   // namespace lrt::view
