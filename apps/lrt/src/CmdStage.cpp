// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt convert` and `lrt render --stage`: USD in and out.
#include <cstdio>
#include <memory>
#include <string>

#include "Commands.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/io/Exr.h"
#include "lrt/io/Readers.h"
#include "lrt/lod/Lrtc.h"
#include "lrt/scene/GpuClouds.h"
#include "lrt/usd/Export.h"
#include "lrt/usd/StageRenderer.h"

namespace lrt::cli {

void addConvert(CLI::App& app) {
    struct Options {
        std::string input, output;
        unsigned degree = 3;
        double rotateX = 0.0;
        bool noCamera = false;
        uint32_t chunkSplats = uint32_t{1} << 16;
        float maxGroupFraction = 0.5F;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("convert",
                                   "a splat file into a USD ParticleField stage, or into a .lrtc with levels of detail");
    cmd->add_option("input", o->input, ".ply / .splat / .spz / .sog / meta.json")->required();
    cmd->add_option("output", o->output, ".usda / .usdc / .usd / .lrtc")->required();
    cmd->add_option("--chunk-splats", o->chunkSplats, ".lrtc: splats per streamed chunk");
    cmd->add_option("--max-group-fraction", o->maxGroupFraction,
                    ".lrtc: the finest merged level holds at most this many groups per splat");
    cmd->add_option("--degree", o->degree, "harmonic degree cap 0..3");
    cmd->add_option("--rotate-x", o->rotateX, "turn the cloud about x (COLMAP clouds: 180)");
    cmd->add_flag("--no-camera", o->noCamera, "do not add /World/Camera");
    cmd->callback([o] {
        auto device = gpu::Device::create();
        if (!device) {
            std::fprintf(stderr, "%s\n", device.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        gpu::ShaderLibrary library(*device);
        auto loader = scene::CloudLoader::create(library);
        if (!loader) {
            std::fprintf(stderr, "%s\n", loader.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        if (lod::isLrtc(o->output)) {
            if (o->rotateX != 0.0) {
                std::fprintf(stderr, "a .lrtc keeps the cloud as it is: turn it where it is used\n");
                throw CLI::RuntimeError(1);
            }
            auto splats = scene::loadSplatFile(*loader, o->input, o->degree);
            auto builder = splats ? lod::LodBuilder::create(library) : Result<lod::LodBuilder>(splats.error());
            if (!builder) {
                std::fprintf(stderr, "%s\n", builder.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            lod::LodBuildSettings settings;
            settings.chunkSplats = o->chunkSplats;
            settings.maxGroupFraction = o->maxGroupFraction;
            auto built = builder->build(*splats, settings);
            auto written = built ? lod::writeLrtc(**device, *built, o->output) : Result<void>(built.error());
            if (!written) {
                std::fprintf(stderr, "%s\n", written.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            std::printf("wrote %s\n", o->output.c_str());
            return;
        }
        auto raw = scene::readSplatRecords(*loader, o->input, o->degree);
        if (!raw) {
            std::fprintf(stderr, "%s\n", raw.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        usd::ExportOptions options;
        options.maxDegree = o->degree;
        options.addCamera = !o->noCamera;
        options.rotateXDegrees = o->rotateX;
        if (auto written = usd::writeParticleFieldStage(library, *raw, o->output, options); !written) {
            std::fprintf(stderr, "%s\n", written.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        std::printf("wrote %s\n", o->output.c_str());
    });
}

void addStage(CLI::App& app) {
    struct Options {
        std::string stage, camera, output = "out.exr", size = "1920x1080", technique = "raster";
        double time = 0.0;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("stage", "render a USD stage through the engine's Hydra delegate");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc")->required();
    cmd->add_option("--camera", o->camera, "camera prim path (default: the first)");
    cmd->add_option("--time", o->time, "USD time code");
    cmd->add_option("--size", o->size, "WIDTHxHEIGHT");
    cmd->add_option("--technique", o->technique, "raster | rt (the delegate's lrt:technique setting)");
    cmd->add_option("-o,--output", o->output, "EXR path");
    cmd->callback([o] {
        uint32_t width = 0, height = 0;
        if (std::sscanf(o->size.c_str(), "%ux%u", &width, &height) != 2) {
            std::fprintf(stderr, "--size wants WIDTHxHEIGHT\n");
            throw CLI::RuntimeError(1);
        }
        auto renderer = usd::StageRenderer::open(o->stage);
        if (!renderer) {
            std::fprintf(stderr, "%s\n", renderer.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        auto image = (*renderer)->render(o->camera, o->time, width, height, o->technique);
        if (!image) {
            std::fprintf(stderr, "%s\n", image.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        if (auto written = io::writeExr(o->output, width, height, image->rgba, image->depth); !written) {
            std::fprintf(stderr, "%s\n", written.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        std::printf("wrote %s\n", o->output.c_str());
    });
}

}   // namespace lrt::cli
