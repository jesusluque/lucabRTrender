// Copyright (c) 2026 lucabRTrender contributors.
//
// A USD stage rendered headless: UsdImaging's scene indices feed a render
// index whose delegate is this engine, a task controller asks for colour and
// depth, and nothing touches a window or OpenGL.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/render/Camera.h"

namespace lrt::usd {

struct StageImage {
    uint32_t           width = 0;
    uint32_t           height = 0;
    std::vector<float> rgba;    ///< bottom row first, linear, premultiplied
    std::vector<float> depth;   ///< view z (distance along the view axis; 0 where nothing was drawn), bottom row first
};

class StageRenderer {
public:
    [[nodiscard]] static Result<std::unique_ptr<StageRenderer>> open(const std::filesystem::path& stage);
    ~StageRenderer();

    /// `camera` is a UsdGeomCamera prim path; empty takes the first camera.
    /// `technique` is the delegate's `lrt:technique` setting: "raster" or "rt".
    [[nodiscard]] Result<StageImage> render(const std::string& camera, double time,
                                            uint32_t width, uint32_t height,
                                            const std::string& technique = "raster");

    /// The same, from a camera that is not on the stage (the engine's
    /// convention: looking down its own -Z, as USD's cameras do).
    [[nodiscard]] Result<StageImage> render(const render::Camera& camera, double time, uint32_t width,
                                            uint32_t height, const std::string& technique = "raster");

    /// The Hydra outputs renders produce, colour and depth always among them
    /// ("primId", "instanceId", "elementId", "Neye", "normal", "primvars:st"...).
    void requestOutputs(const std::vector<std::string>& aovs);

    /// The last render's Hydra render buffer for `aov` ("color", "depth"), as
    /// a host mapping it reads it: the buffer's own format, top row first.
    [[nodiscard]] Result<std::vector<uint8_t>> mappedOutput(const std::string& aov);

    /// Every camera prim on the stage.
    [[nodiscard]] std::vector<std::string> cameras() const;

    /// The stage's timeCodesPerSecond and startTimeCode: how a frame on a
    /// clock maps to a USD time.
    [[nodiscard]] double timeCodesPerSecond() const;
    [[nodiscard]] double startTimeCode() const;

private:
    StageRenderer();
    [[nodiscard]] Result<StageImage> execute(uint32_t width, uint32_t height);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::usd
