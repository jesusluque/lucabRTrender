// Copyright (c) 2026 lucabRTrender contributors.
//
// lrt: the engine from a terminal. Headless by design -- a render node has no
// window, and a test has no window either.
#include <CLI/CLI.hpp>

#include "Commands.h"
#include "lrt/core/Log.h"

int main(int argc, char** argv) {
    CLI::App app{"lucabRTrender: Gaussian splats and point clouds on the GPU"};
    app.require_subcommand(1);
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "debug logging");
    app.parse_complete_callback([&verbose] {
        if (verbose) {
            lrt::log::setMinimum(lrt::log::Level::Debug);
        }
    });

    lrt::cli::addInfo(app);
    lrt::cli::addRender(app);
    lrt::cli::addBench(app);
    lrt::cli::addConvert(app);
    lrt::cli::addStage(app);
    lrt::cli::addAofx(app);

    CLI11_PARSE(app, argc, argv);
    return 0;
}
