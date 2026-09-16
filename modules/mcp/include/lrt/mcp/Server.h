// Copyright (c) 2026 lucabRTrender contributors.
//
// MCP over the renderer: open a stage, look at it, render it, measure it.
//
// The protocol is the only thing this class knows about I/O. It takes one
// message and returns the reply, so the transport -- stdio in the app, a
// string in the tests -- is somebody else's problem; every framing bug lives
// in a transport that cannot be tested when parsing and reading are the same
// function. The shape follows openFXplayer's server, which this project's
// aofx host already speaks to.
//
// **The device and the stage stay warm between calls.** A CLI shelled out to
// once a frame opens the GPU, compiles the kernels it needs and throws the
// lot away; here the second render of a stage costs what a second render
// should cost. That is the whole reason to drive the engine from outside
// rather than spawn `lrt stage` in a loop.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "lrt/core/Result.h"

namespace lrt::mcp {

/// The version this server speaks unless the client asks for another it knows.
inline constexpr std::string_view kPreferredProtocolVersion = "2025-06-18";

class Server {
public:
    Server();
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    /// One JSON-RPC message in, its reply out. No reply for a notification,
    /// which is what a client expects: an answered notification is a protocol
    /// violation, and the first message a client sends is one.
    [[nodiscard]] std::optional<std::string> handle(std::string_view message);

    [[nodiscard]] bool isInitialised() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::mcp
