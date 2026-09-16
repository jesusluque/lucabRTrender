// Copyright (c) 2026 lucabRTrender contributors.
//
// The MCP server as a process: newline-delimited JSON-RPC over stdin and
// stdout. Everything else is in lrt::mcp::Server; what is here is the
// transport, and the transport has one rule.
//
// **stdout carries the protocol and nothing else.** One stray printf and the
// client sees a parse error on a message it cannot attribute to anything. The
// engine's logging already goes to stderr (core/Log.h), which is why this can
// afford to be short.
#include <cstdio>
#include <iostream>
#include <string>

#include "lrt/mcp/Server.h"

int main(int argc, char** argv) {
    for (int k = 1; k < argc; ++k) {
        const std::string arg = argv[k];
        if (arg == "-h" || arg == "--help") {
            std::fprintf(stderr,
                         "lrt-mcp -- lucabRTrender as an MCP server.\n"
                         "\n"
                         "Speaks JSON-RPC 2.0 over stdin/stdout; it is meant to be launched by\n"
                         "an MCP client, not typed at. To register it with Claude Code:\n"
                         "\n"
                         "  claude mcp add lrt -- %s\n"
                         "\n"
                         "The device and the stage stay open between calls, so the second render\n"
                         "of a stage costs what a second render should cost.\n",
                         argv[0]);
            return 0;
        }
    }
    lrt::mcp::Server server;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (std::optional<std::string> reply = server.handle(line); reply.has_value()) {
            std::cout << *reply << "\n" << std::flush;
        }
    }
    return 0;
}
