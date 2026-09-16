// Copyright (c) 2026 lucabRTrender contributors.
//
// The protocol, and nothing else: JSON-RPC 2.0, the handshake, what a
// notification is. Every tool lives in the table (Tools.cpp) and acts on a
// session (Session.h), which is what holds the open stage and the device.
#include "lrt/mcp/Server.h"

#include <algorithm>

#include <nlohmann/json.hpp>

#include "Session.h"
#include "Tools.h"

namespace lrt::mcp {
namespace {

using nlohmann::json;

// JSON-RPC 2.0's reserved codes.
constexpr int kParseError = -32700;
constexpr int kInvalidRequest = -32600;
constexpr int kMethodNotFound = -32601;

/// Revisions this server understands. A client asking for one of these is
/// answered in its own version; anything else is answered in ours, which the
/// specification allows.
constexpr std::string_view kKnownProtocolVersions[] = {"2025-06-18", "2025-03-26", "2024-11-05"};

json rpcError(const json& id, int code, std::string message) {
    return json{{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", std::move(message)}}}};
}

json rpcResult(const json& id, json result) {
    return json{{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}};
}

}   // namespace

struct Server::Impl {
    Session session;
    bool    initialised = false;

    [[nodiscard]] json toolList() const {
        json tools = json::array();
        for (const ToolEntry& entry : toolTable()) {
            tools.push_back(json{{"name", entry.name}, {"description", entry.description},
                                 {"inputSchema", entry.schema}});
        }
        return json{{"tools", std::move(tools)}};
    }

    [[nodiscard]] json callTool(const std::string& name, const json& args) {
        for (const ToolEntry& entry : toolTable()) {
            if (entry.name == name) {
                return entry.handler(session, args);
            }
        }
        return textAnswer("Unknown tool '" + name + "'.", true);
    }
};

Server::Server() : impl_(std::make_unique<Impl>()) {}
Server::~Server() = default;

bool Server::isInitialised() const noexcept {
    return impl_->initialised;
}

std::optional<std::string> Server::handle(std::string_view message) {
    json request;
    try {
        request = json::parse(message);
    } catch (const json::exception& error) {
        return rpcError(nullptr, kParseError, error.what()).dump();
    }
    if (!request.is_object()) {
        return rpcError(nullptr, kInvalidRequest, "message is not an object").dump();
    }
    const json id = request.value("id", json(nullptr));
    // No id is a notification, and a notification gets no reply at all -- not
    // an empty one. `notifications/initialized` is the first message a client
    // sends, and answering it is a protocol violation on the handshake.
    const bool isNotification = !request.contains("id") || id.is_null();
    const auto methodField = request.find("method");
    if (methodField == request.end() || !methodField->is_string()) {
        return isNotification ? std::nullopt
                              : std::optional(rpcError(id, kInvalidRequest, "missing 'method'").dump());
    }
    const std::string method = methodField->get<std::string>();
    const json params = request.value("params", json::object());

    if (method == "initialize") {
        const std::string asked = params.value("protocolVersion", std::string(kPreferredProtocolVersion));
        const bool known = std::find(std::begin(kKnownProtocolVersions), std::end(kKnownProtocolVersions), asked) !=
                           std::end(kKnownProtocolVersions);
        impl_->initialised = true;
        return rpcResult(id,
                         json{{"protocolVersion", known ? asked : std::string(kPreferredProtocolVersion)},
                              {"capabilities", {{"tools", json::object()}}},
                              {"serverInfo", {{"name", "lucabRTrender"}, {"version", "0.1.0"}}},
                              {"instructions",
                               "A GPU renderer for USD stages and Gaussian splat captures: open a stage, look at "
                               "what is in it, render it by rasterisation or by path tracing, and read back what "
                               "a pixel saw. The device and the stage stay open between calls, so the second "
                               "render of a stage costs what a second render should cost.\n\nRenders are written "
                               "to EXR; ask for a smaller size while you are looking around and a larger one "
                               "when you know what you want."}})
            .dump();
    }
    if (isNotification) {
        // initialized, cancelled, progress: silence is the right answer.
        return std::nullopt;
    }
    if (method == "ping") {
        return rpcResult(id, json::object()).dump();
    }
    if (method == "tools/list") {
        return rpcResult(id, impl_->toolList()).dump();
    }
    if (method == "tools/call") {
        const std::string name = params.value("name", std::string{});
        const json args = params.value("arguments", json::object());
        return rpcResult(id, impl_->callTool(name, args)).dump();
    }
    return rpcError(id, kMethodNotFound, "unknown method '" + method + "'").dump();
}

}   // namespace lrt::mcp
