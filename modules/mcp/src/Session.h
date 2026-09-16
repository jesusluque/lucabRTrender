// Copyright (c) 2026 lucabRTrender contributors.
//
// What the tools act on and what survives between calls: the stage that is
// open, the settings asked for, and the last frame's size -- so `pick` can
// answer about the frame the caller just rendered.
#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "lrt/core/Result.h"
#include "lrt/usd/StageRenderer.h"

namespace lrt::mcp {

struct Session {
    std::unique_ptr<usd::StageRenderer> renderer;
    std::filesystem::path               stage;
    uint32_t                            lastWidth = 0;
    uint32_t                            lastHeight = 0;
    double                              lastTime = 0.0;
    std::string                         lastCamera;

    [[nodiscard]] bool open() const noexcept { return renderer != nullptr; }
};

}   // namespace lrt::mcp
