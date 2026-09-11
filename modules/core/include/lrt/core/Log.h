// Copyright (c) 2026 lucabRTrender contributors.
//
// One line per event, to stderr, with a level. Nothing more: the engine is a
// library, and the application that embeds it (openFXplayer) has its own log to
// route these into through `setSink`.
#pragma once

#include <format>
#include <functional>
#include <string>
#include <string_view>

namespace lrt::log {

enum class Level { Debug, Info, Warn, Error };

using Sink = std::function<void(Level, std::string_view)>;

/// Replaces where lines go. Empty restores stderr. Not thread-safe to call:
/// it belongs at start-up.
void setSink(Sink sink);
void setMinimum(Level level) noexcept;
void write(Level level, std::string_view line);

template <typename... Args>
void debug(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Debug, std::format(fmt, std::forward<Args>(args)...));
}
template <typename... Args>
void info(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Info, std::format(fmt, std::forward<Args>(args)...));
}
template <typename... Args>
void warn(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Warn, std::format(fmt, std::forward<Args>(args)...));
}
template <typename... Args>
void error(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Error, std::format(fmt, std::forward<Args>(args)...));
}

}   // namespace lrt::log
