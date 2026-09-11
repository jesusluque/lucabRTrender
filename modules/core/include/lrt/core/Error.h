// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace lrt {

/// Broad failure categories, as in openFXplayer. Coarse on purpose: the
/// message carries the detail, the code exists so a caller can branch without
/// parsing a string.
enum class ErrorCode {
    Unknown,
    NotFound,
    InvalidArgument,
    Unsupported,
    IoFailure,
    DeviceFailure,   ///< the GPU or its driver refused
    ShaderFailure,   ///< Slang could not compile or link
    PluginFailure,
    OutOfMemory,
    Cancelled,
    InternalError,
};

[[nodiscard]] std::string_view toString(ErrorCode) noexcept;

/// A failure value. Cheap to move, never thrown.
class Error {
public:
    Error() = default;
    Error(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    template <typename... Args>
    [[nodiscard]] static Error make(ErrorCode code,
                                    std::format_string<Args...> fmt,
                                    Args&&... args) {
        return Error(code, std::format(fmt, std::forward<Args>(args)...));
    }

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    /// "NotFound: no kernel 'blend'"
    [[nodiscard]] std::string toString() const;

private:
    ErrorCode   code_ = ErrorCode::Unknown;
    std::string message_;
};

}   // namespace lrt
