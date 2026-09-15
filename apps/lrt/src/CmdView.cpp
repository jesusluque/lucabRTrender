// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt view`: a window onto a USD stage.
#include <cstdio>
#include <memory>
#include <string>

#include "Commands.h"
#include "lrt/view/Viewer.h"

namespace lrt::cli {

void addView(CLI::App& app) {
    auto options = std::make_shared<view::ViewOptions>();
    auto size = std::make_shared<std::string>("1600x900");
    auto* cmd = app.add_subcommand("view", "look at a USD stage in a window: free camera, stage cameras, outputs, picking");
    cmd->add_option("stage", options->stage, ".usd / .usda / .usdc")->required();
    cmd->add_option("--camera", options->camera, "start from this camera prim (default: a free camera framing the stage)");
    cmd->add_option("--technique", options->technique, "raster | rt");
    cmd->add_option("--visibility", options->visibility, "how meshes are seen: automatic | raster | rays | bvh");
    cmd->add_option("--size", *size, "the window's WIDTHxHEIGHT, in points");
    cmd->add_option("--frames", options->frames, "close after this many frames and print their timings");
    cmd->add_option("--light-samples", options->lightSamples, "samples per light per pixel (1 is interactive)");
    cmd->add_flag("--choose-lights", options->chooseLights, "one light a sample, chosen by power");
    cmd->add_flag("--edr", options->edr, "extended dynamic range: a float surface and ACES 2.0 up to the screen's peak");
    cmd->add_option("--ocio-config", options->ocioConfig, "an OpenColorIO config (default with --ocio-display/--ocio-view: ocio://studio-config-latest)");
    cmd->add_option("--ocio-display", options->ocioDisplay, "the OCIO display (default: the config's)");
    cmd->add_option("--ocio-view", options->ocioView, "the OCIO view (default: the display's)");
    cmd->add_option("--snapshot", options->snapshot, "with --frames: the last frame as shown, to this EXR");
    cmd->add_flag("--play", options->play, "start with the timeline playing");
    cmd->callback([options, size] {
        if (std::sscanf(size->c_str(), "%ux%u", &options->width, &options->height) != 2) {
            std::fprintf(stderr, "--size wants WIDTHxHEIGHT\n");
            throw CLI::RuntimeError(1);
        }
        auto stats = view::runViewer(*options);
        if (!stats) {
            std::fprintf(stderr, "%s\n", stats.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        std::printf("%u frames: draw %.2f ms, frame %.2f ms (medians); %u distinct times, the last %.2f\n",
                    stats->frames, stats->medianDrawMs, stats->medianFrameMs, stats->distinctTimes, stats->lastTime);
    });
}

}   // namespace lrt::cli
