// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <CLI/CLI.hpp>

namespace lrt::cli {

/// Each subcommand registers itself on the app and returns the callback to run
/// when it was the one chosen.
void addInfo(CLI::App& app);
void addRender(CLI::App& app);
void addBench(CLI::App& app);
void addConvert(CLI::App& app);
void addStage(CLI::App& app);
void addAofx(CLI::App& app);
void addLive(CLI::App& app);

}   // namespace lrt::cli
