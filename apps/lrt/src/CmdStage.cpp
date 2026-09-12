// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt convert` and `lrt render --stage`: USD in and out.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

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
        std::string stage, camera, output = "out.exr", size = "1920x1080", technique = "raster",
                    visibility = "automatic";
        double time = 0.0;
        std::vector<double> eye, target, up{0.0, 1.0, 0.0};
        double focal = 35.0, nearZ = 0.1, farZ = 100000.0;
        uint32_t frames = 1;
        uint32_t pathSamples = 1, pathBounces = 1, pathTotal = 1, motionBuckets = 4, refine = 0;
        bool denoise = false;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("stage", "render a USD stage through the engine's Hydra delegate");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc")->required();
    cmd->add_option("--camera", o->camera, "camera prim path (default: the first)");
    cmd->add_option("--time", o->time, "USD time code");
    cmd->add_option("--size", o->size, "WIDTHxHEIGHT");
    cmd->add_option("--technique", o->technique, "raster | rt (the delegate's lrt:technique setting)");
    cmd->add_option("--visibility", o->visibility, "how meshes are seen: automatic | raster | rays | bvh");
    cmd->add_option("--path-samples", o->pathSamples, "rt: paths a pixel each pass");
    cmd->add_option("--path-bounces", o->pathBounces, "rt: bounces after the first hit");
    cmd->add_option("--path-total", o->pathTotal, "rt: paths a pixel the image is drawn until it holds");
    cmd->add_flag("--denoise", o->denoise, "rt: denoise the image once it holds its total (OIDN)");
    cmd->add_option("--motion-buckets", o->motionBuckets,
                    "rt: shutter slices for motion blur, 1 to 8 (the shutter is the camera's)");
    cmd->add_option("--refine", o->refine, "subdivision surfaces refined this many levels (0: the control mesh)");
    cmd->add_option("--eye", o->eye, "a camera of its own at x y z (with --target), not one on the stage")->expected(3);
    cmd->add_option("--target", o->target, "where that camera looks")->expected(3);
    cmd->add_option("--up", o->up, "its up vector")->expected(3);
    cmd->add_option("--focal", o->focal, "its focal length, mm (24.576 mm aperture)");
    cmd->add_option("--near", o->nearZ, "its near clipping distance");
    cmd->add_option("--far", o->farZ, "its far clipping distance");
    cmd->add_option("-o,--output", o->output, "EXR path");
    cmd->add_option("--frames", o->frames,
                    "render this many times and print the time a frame takes (Hydra sync, drawing and the readback)");
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
        if (auto set = (*renderer)->setMeshVisibility(o->visibility); !set) {
            std::fprintf(stderr, "%s\n", set.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        (*renderer)->setPathSamples(o->pathSamples);
        (*renderer)->setPathBounces(o->pathBounces);
        (*renderer)->setPathTotal(o->pathTotal);
        (*renderer)->setDenoise(o->denoise);
        (*renderer)->setMotionBuckets(o->motionBuckets);
        (*renderer)->setRefineLevel(o->refine);
        Result<usd::StageImage> image = Error(ErrorCode::InvalidArgument, "no image");
        std::vector<double> ms;
        for (uint32_t frame = 0; frame < std::max(o->frames, uint32_t{1}); ++frame) {
            const auto start = std::chrono::steady_clock::now();
            if (o->eye.size() == 3 && o->target.size() == 3) {
                render::Camera camera = render::Camera::lookingAt({o->eye[0], o->eye[1], o->eye[2]},
                                                                  {o->target[0], o->target[1], o->target[2]},
                                                                  {o->up[0], o->up[1], o->up[2]});
                camera.lens.focal = o->focal;
                camera.lens.nearZ = o->nearZ;
                camera.lens.farZ = o->farZ;
                image = (*renderer)->render(camera, o->time, width, height, o->technique);
            } else {
                image = (*renderer)->render(o->camera, o->time, width, height, o->technique);
            }
            if (!image) {
                break;
            }
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        if (ms.size() > 1) {
            // The first frame loads the stage onto the device and compiles; the rest are steady.
            std::vector<double> steady(ms.begin() + 1, ms.end());
            std::sort(steady.begin(), steady.end());
            std::printf("first frame %.1f ms; then median %.2f ms, fastest %.2f ms over %zu frames\n", ms.front(),
                        steady[steady.size() / 2], steady.front(), steady.size());
        }
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
