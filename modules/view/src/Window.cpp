// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/view/Window.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>

#include "lrt/core/Log.h"
#include "lrt/core/Platform.h"

namespace lrt::view {

namespace {

bool initialised() {
    static const bool kReady = [] {
        glfwSetErrorCallback([](int code, const char* text) { log::warn("glfw {}: {}", code, text); });
        return glfwInit() == GLFW_TRUE;
    }();
    return kReady;
}

}   // namespace

bool Window::available() {
    return initialised();
}

Result<std::unique_ptr<Window>> Window::open(const std::string& title, uint32_t width, uint32_t height,
                                             bool visible) {
    if (!initialised()) {
        return Error(ErrorCode::Unsupported, "no display to open a window on");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    auto window = std::unique_ptr<Window>(new Window());
    window->window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title.c_str(), nullptr,
                                       nullptr);
    if (window->window_ == nullptr) {
        return Error(ErrorCode::Unsupported, "cannot open a window");
    }
    return window;
}

Window::~Window() {
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
    }
}

rhi::WindowHandle Window::handle() const {
#if defined(__APPLE__)
    return rhi::WindowHandle::fromNSWindow(glfwGetCocoaWindow(window_));
#elif defined(_WIN32)
    return rhi::WindowHandle::fromHwnd(glfwGetWin32Window(window_));
#else
    return rhi::WindowHandle::fromXlibWindow(glfwGetX11Display(), static_cast<uint32_t>(glfwGetX11Window(window_)));
#endif
}

void Window::matchSurfaceToBacking() const {
#if defined(__APPLE__)
    platform::matchLayerToBacking(glfwGetCocoaWindow(window_));
#endif
}

double Window::extendedRangeHeadroom() const {
#if defined(__APPLE__)
    return platform::extendedRangeHeadroom(glfwGetCocoaWindow(window_));
#else
    return 1.0;
#endif
}

bool Window::enableExtendedRange() const {
#if defined(__APPLE__)
    return platform::enableExtendedRange(glfwGetCocoaWindow(window_));
#else
    return false;
#endif
}

std::pair<uint32_t, uint32_t> Window::framebufferSize() const {
    int w = 0;
    int h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    return {static_cast<uint32_t>(std::max(w, 0)), static_cast<uint32_t>(std::max(h, 0))};
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(window_) == GLFW_TRUE;
}

void Window::requestClose() {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

void Window::pollEvents() {
    glfwPollEvents();
}

void Window::setTitle(const std::string& title) {
    glfwSetWindowTitle(window_, title.c_str());
}

}   // namespace lrt::view
