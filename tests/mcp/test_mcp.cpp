// Copyright (c) 2026 lucabRTrender contributors.
//
// The MCP server at the protocol's level: one message in, one reply out, with
// no transport in the way. Every interesting failure of a stdio server is a
// framing bug, and framing bugs are untestable when parsing and reading are
// the same function -- so the server takes a string and the app does the rest.
#include "../gpu/GpuTest.h"

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "lrt/mcp/Server.h"

using namespace lrt;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

json ask(mcp::Server& server, const json& message) {
    const std::optional<std::string> reply = server.handle(message.dump());
    REQUIRE(reply.has_value());
    return json::parse(*reply);
}

json call(mcp::Server& server, const std::string& tool, const json& arguments) {
    return ask(server, json{{"jsonrpc", "2.0"},
                            {"id", 7},
                            {"method", "tools/call"},
                            {"params", {{"name", tool}, {"arguments", arguments}}}});
}

std::string textOf(const json& answer) {
    std::string text;
    for (const json& block : answer["result"]["content"]) {
        if (block.value("type", std::string{}) == "text") {
            text += block.value("text", std::string{});
        }
    }
    return text;
}

fs::path scratch(const char* name) {
    const fs::path dir = fs::temp_directory_path() / "lrt-mcp-tests";
    fs::create_directories(dir);
    return dir / name;
}

}   // namespace

TEST_CASE("the server handshakes, lists its tools and refuses what it has not", "[mcp]") {
    mcp::Server server;
    CHECK_FALSE(server.isInitialised());
    const json hello = ask(server, json{{"jsonrpc", "2.0"},
                                        {"id", 1},
                                        {"method", "initialize"},
                                        {"params", {{"protocolVersion", "2025-06-18"}}}});
    CHECK(hello["result"]["protocolVersion"] == "2025-06-18");
    CHECK(hello["result"]["serverInfo"]["name"] == "lucabRTrender");
    CHECK(hello["result"]["capabilities"].contains("tools"));
    CHECK(server.isInitialised());

    // A version nobody implements is answered in ours, which the specification
    // allows and which lets an old client decide for itself.
    mcp::Server other;
    const json older = ask(other, json{{"jsonrpc", "2.0"},
                                       {"id", 1},
                                       {"method", "initialize"},
                                       {"params", {{"protocolVersion", "1999-01-01"}}}});
    CHECK(older["result"]["protocolVersion"] == std::string(mcp::kPreferredProtocolVersion));

    // A notification is answered with silence. The first message a client
    // sends is one, and answering it is a protocol violation.
    CHECK_FALSE(server.handle(R"({"jsonrpc":"2.0","method":"notifications/initialized"})").has_value());

    const json tools = ask(server, json{{"jsonrpc", "2.0"}, {"id", 2}, {"method", "tools/list"}});
    const json& listed = tools["result"]["tools"];
    CHECK(listed.size() >= 8);
    std::vector<std::string> names;
    for (const json& tool : listed) {
        CHECK(tool.contains("name"));
        CHECK(tool.contains("description"));
        CHECK(tool["inputSchema"]["type"] == "object");
        names.push_back(tool["name"].get<std::string>());
    }
    for (const char* wanted : {"open_stage", "render", "pick", "device_info", "stage_tree", "bounds"}) {
        CHECK(std::find(names.begin(), names.end(), wanted) != names.end());
    }

    // An unknown tool is the tool's failure, not the protocol's: a result with
    // isError, which is what a client shows the model.
    const json unknown = call(server, "no_such_tool", json::object());
    CHECK(unknown.contains("result"));
    CHECK(unknown["result"]["isError"] == true);
    CHECK(textOf(unknown).find("no_such_tool") != std::string::npos);

    // A method that is not a tool is the protocol's failure.
    const json bad = ask(server, json{{"jsonrpc", "2.0"}, {"id", 3}, {"method", "no/such/method"}});
    CHECK(bad.contains("error"));
    CHECK(bad["error"]["code"] == -32601);

    // And a message that is not JSON at all.
    const std::optional<std::string> garbage = server.handle("{not json");
    REQUIRE(garbage.has_value());
    CHECK(json::parse(*garbage)["error"]["code"] == -32700);
}

TEST_CASE("a tool that wants a stage says so, and says why one will not open", "[mcp]") {
    mcp::Server server;
    const json none = call(server, "render", json::object());
    CHECK(none["result"]["isError"] == true);
    CHECK(textOf(none).find("no stage is open") != std::string::npos);

    const json missing = call(server, "open_stage", json{{"stage", "/nowhere/at/all.usda"}});
    CHECK(missing["result"]["isError"] == true);
    CHECK(textOf(missing).find("all.usda") != std::string::npos);
}

TEST_CASE("the server opens a stage, renders it and answers with the image", "[mcp][gpu]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("square.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -4), (1, -1, -4), (1, 1, -4), (-1, 1, -4)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.3, 0.1)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    mcp::Server server;
    const json opened = call(server, "open_stage", json{{"stage", path.string()}});
    CHECK_FALSE(opened["result"].value("isError", false));
    CHECK(textOf(opened).find("/Camera") != std::string::npos);

    const json drawn = call(server, "render", json{{"width", 96}, {"height", 64}});
    CHECK_FALSE(drawn["result"].value("isError", false));
    const std::string text = textOf(drawn);
    std::printf("  %s\n", text.c_str());
    CHECK(text.find("96x64") != std::string::npos);
    // The image comes back with the numbers: a model that cannot see what it
    // rendered is guessing.
    bool sawImage = false;
    for (const json& block : drawn["result"]["content"]) {
        if (block.value("type", std::string{}) == "image") {
            sawImage = true;
            CHECK(block["mimeType"] == "image/png");
            const std::string data = block["data"].get<std::string>();
            CHECK(data.size() > 100);
            // base64 of a PNG: the signature is the first bytes, "iVBORw0KGgo".
            CHECK(data.rfind("iVBORw0KGgo", 0) == 0);
        }
    }
    CHECK(sawImage);

    // What a pixel saw, of the frame just drawn.
    const json picked = call(server, "pick", json{{"x", 48}, {"y", 32}});
    std::printf("  pick: %s\n", textOf(picked).c_str());
    CHECK(textOf(picked).find("/Square") != std::string::npos);

    // And where it is.
    const json where = call(server, "bounds", json::object());
    CHECK(textOf(where).find("min") != std::string::npos);

    // The device, so a caller knows what the frame can afford.
    const json device = call(server, "device_info", json::object());
    CHECK(textOf(device).find("backend") != std::string::npos);
}
