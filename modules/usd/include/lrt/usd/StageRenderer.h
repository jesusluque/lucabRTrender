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

namespace lrt::usd {

struct StageImage {
    uint32_t           width = 0;
    uint32_t           height = 0;
    std::vector<float> rgba;    ///< bottom row first, linear, premultiplied
    std::vector<float> depth;   ///< Hydra depth in [0, 1], bottom row first
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

    /// Every camera prim on the stage.
    [[nodiscard]] std::vector<std::string> cameras() const;

    /// The stage's timeCodesPerSecond and startTimeCode: how a frame on a
    /// clock maps to a USD time.
    [[nodiscard]] double timeCodesPerSecond() const;
    [[nodiscard]] double startTimeCode() const;

private:
    StageRenderer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::usd
