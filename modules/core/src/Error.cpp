// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/core/Error.h"

namespace lrt {

std::string_view toString(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::Unknown: return "Unknown";
    case ErrorCode::NotFound: return "NotFound";
    case ErrorCode::InvalidArgument: return "InvalidArgument";
    case ErrorCode::Unsupported: return "Unsupported";
    case ErrorCode::IoFailure: return "IoFailure";
    case ErrorCode::DeviceFailure: return "DeviceFailure";
    case ErrorCode::ShaderFailure: return "ShaderFailure";
    case ErrorCode::PluginFailure: return "PluginFailure";
    case ErrorCode::OutOfMemory: return "OutOfMemory";
    case ErrorCode::Cancelled: return "Cancelled";
    case ErrorCode::InternalError: return "InternalError";
    }
    return "Unknown";
}

std::string Error::toString() const {
    return std::format("{}: {}", lrt::toString(code_), message_);
}

}   // namespace lrt
