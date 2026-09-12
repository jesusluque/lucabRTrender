// Copyright (c) 2026 lucabRTrender contributors.
//
// A window for lrt view: GLFW with no client API, since slang-rhi makes the
// surface on the engine's own device.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <slang-rhi.h>

#include "lrt/core/Result.h"

struct GLFWwindow;

namespace lrt::view {

class Window {
public:
    /// Whether this session can open windows at all (a macOS login session,
    /// an X11 or Wayland display).
    [[nodiscard]] static bool available();

    [[nodiscard]] static Result<std::unique_ptr<Window>> open(const std::string& title, uint32_t width,
                                                              uint32_t height, bool visible = true);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    /// What slang-rhi makes a surface from.
    [[nodiscard]] rhi::WindowHandle handle() const;
    /// Once the surface is made: its layer at the window's backing scale.
    void matchSurfaceToBacking() const;
    /// The screen's extended range headroom (1 on a standard display), and
    /// the surface asked to show it -- linear P3 in a float format.
    [[nodiscard]] double extendedRangeHeadroom() const;
    [[nodiscard]] bool enableExtendedRange() const;
    /// The drawable's size, in pixels.
    [[nodiscard]] std::pair<uint32_t, uint32_t> framebufferSize() const;
    [[nodiscard]] bool shouldClose() const;
    void requestClose();
    void pollEvents();
    void setTitle(const std::string& title);
    [[nodiscard]] GLFWwindow* glfw() const noexcept { return window_; }

private:
    Window() = default;
    GLFWwindow* window_ = nullptr;
};

}   // namespace lrt::view
