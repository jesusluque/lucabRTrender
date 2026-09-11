// Copyright (c) 2026 lucabRTrender contributors.
//
// lrt: the engine from a terminal. Headless but for `lrt view` -- a render node
// has no window, and nothing else here opens one.
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
    lrt::cli::addLive(app);
#if LRT_HAVE_VIEW
    lrt::cli::addView(app);
#endif

    CLI11_PARSE(app, argc, argv);
    return 0;
}
