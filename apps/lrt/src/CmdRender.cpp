// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt render` and `lrt bench` for splat files, before scenes come from USD:
// a cloud, an eye, a target, a lens.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Commands.h"
#include "lrt/core/Log.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/io/Exr.h"
#include "lrt/io/Readers.h"
#include "lrt/lod/Lod.h"
#include "lrt/render/GaussianRayTracer.h"
#include "lrt/render/PointRasterizer.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::cli {
namespace {

struct RenderOptions {
    std::vector<std::string> splats;
    std::vector<std::string> points;
    float                    lod = 0.0F;   ///< px a merged cell may span; 0 draws the splats
    std::string              technique = "raster";    ///< raster | rt | rt-hw | rt-bvh | reference | reference-rt
    std::string              pointRoute = "raster";   ///< raster | discs
    float                    pointSize = 0.01F;
    bool                     pointPixels = false;
    float                    edl = 0.0F;
    float                    surface = 0.0F;
    std::string              output = "out.exr";
    std::vector<double>      eye;
    std::vector<double>      target;
    std::vector<double>      up{0.0, 1.0, 0.0};
    std::vector<double>      rotateX;   ///< degrees applied to every cloud (COLMAP: 180)
    std::vector<double>      scale;     ///< per-axis scale applied to every cloud
    double                   focal = 35.0;
    double                   nearZ = 0.01;
    std::string              size = "1920x1080";
    unsigned                 degree = 3;
    bool                     noAntialias = false;
    int                      repeat = 1;
    bool                     stages = false;
};

bool parseSize(const std::string& text, uint32_t& w, uint32_t& h) {
    return std::sscanf(text.c_str(), "%ux%u", &w, &h) == 2 && w > 0 && h > 0;
}

render::Vec3 vec(const std::vector<double>& v, render::Vec3 fallback) {
    return v.size() == 3 ? render::Vec3{v[0], v[1], v[2]} : fallback;
}

int run(const RenderOptions& options, bool bench) {
    uint32_t width = 0;
    uint32_t height = 0;
    if (!parseSize(options.size, width, height)) {
        std::fprintf(stderr, "--size wants WIDTHxHEIGHT\n");
        return 1;
    }
    auto device = gpu::Device::create();
    if (!device) {
        std::fprintf(stderr, "%s\n", device.error().toString().c_str());
        return 1;
    }
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) {
        std::fprintf(stderr, "%s\n", loader.error().toString().c_str());
        return 1;
    }
    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    scene::Bounds all;
    bool first = true;
    for (const std::string& path : options.splats) {
        auto splats = scene::loadSplatFile(*loader, path, options.degree);
        if (!splats) {
            std::fprintf(stderr, "%s\n", splats.error().toString().c_str());
            return 1;
        }
        for (int k = 0; k < 3; ++k) {
            const auto kk = static_cast<size_t>(k);
            all.min[kk] = first ? splats->bounds.min[kk] : std::min(all.min[kk], splats->bounds.min[kk]);
            all.max[kk] = first ? splats->bounds.max[kk] : std::max(all.max[kk], splats->bounds.max[kk]);
        }
        first = false;
        clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*splats)));
    }
    std::vector<std::unique_ptr<scene::GpuPoints>> pointClouds;
    for (const std::string& path : options.points) {
        auto raw = io::readPoints(path);
        if (!raw) {
            std::fprintf(stderr, "%s\n", raw.error().toString().c_str());
            return 1;
        }
        auto uploaded = loader->upload(*raw);
        if (!uploaded) {
            std::fprintf(stderr, "%s\n", uploaded.error().toString().c_str());
            return 1;
        }
        for (int k = 0; k < 3; ++k) {
            const auto kk = static_cast<size_t>(k);
            all.min[kk] = first ? uploaded->bounds.min[kk] : std::min(all.min[kk], uploaded->bounds.min[kk]);
            all.max[kk] = first ? uploaded->bounds.max[kk] : std::max(all.max[kk], uploaded->bounds.max[kk]);
        }
        first = false;
        pointClouds.push_back(std::make_unique<scene::GpuPoints>(std::move(*uploaded)));
    }
    if (clouds.empty() && pointClouds.empty()) {
        std::fprintf(stderr, "nothing to render: give --splats or --points\n");
        return 1;
    }

    render::Mat4 model = render::Mat4::identity();
    if (options.rotateX.size() == 1) {
        model = aofx::xform::rotationX(options.rotateX[0]);
    }
    if (options.scale.size() == 3) {
        model = model * aofx::xform::scaling({options.scale[0], options.scale[1], options.scale[2]});
    }
    std::vector<render::SplatInstance> instances;
    for (const auto& cloud : clouds) {
        instances.push_back({cloud.get(), model});
    }
    std::vector<render::PointInstance> pointInstances;
    for (const auto& cloud : pointClouds) {
        render::PointInstance instance{cloud.get(), model, {}};
        instance.style.size = options.pointSize;
        instance.style.sizeMode =
            options.pointPixels ? render::PointStyle::Size::Pixels : render::PointStyle::Size::World;
        instance.style.edlStrength = options.edl;
        instance.style.surfaceDepthOffset = options.surface;
        pointInstances.push_back(instance);
    }

    const auto d = [](float v) { return static_cast<double>(v); };
    const render::Vec3 centre{(d(all.min[0]) + d(all.max[0])) * 0.5, (d(all.min[1]) + d(all.max[1])) * 0.5,
                              (d(all.min[2]) + d(all.max[2])) * 0.5};
    const double extent = std::max({d(all.max[0]) - d(all.min[0]), d(all.max[1]) - d(all.min[1]),
                                    d(all.max[2]) - d(all.min[2])});
    const render::Vec3 target = vec(options.target, model.point(centre));
    const render::Vec3 eye =
        vec(options.eye, target + render::Vec3{0.0, 0.0, std::max(extent, 1e-3) * 1.2});
    render::Camera camera = render::Camera::lookingAt(eye, target, vec(options.up, {0, 1, 0}));
    camera.lens.focal = options.focal;
    camera.lens.nearZ = options.nearZ;

    auto rasterizer = render::TileRasterizer::create(library);
    if (!rasterizer) {
        std::fprintf(stderr, "%s\n", rasterizer.error().toString().c_str());
        return 1;
    }
    render::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    settings.antialias = !options.noAntialias;
    settings.timeStages = options.stages;
    render::RenderTargets targets;
    std::optional<render::PointRasterizer> pointRaster;
    render::RenderTargets pointLayer;
    const bool rasterPoints = !pointInstances.empty() && options.pointRoute == "raster";
    if (rasterPoints) {
        auto made = render::PointRasterizer::create(library);
        if (!made) {
            std::fprintf(stderr, "%s\n", made.error().toString().c_str());
            return 1;
        }
        pointRaster.emplace(std::move(*made));
    }

    const int rounds = std::max(options.repeat, 1);
    if (options.technique != "raster") {
        if (!pointInstances.empty()) {
            std::fprintf(stderr, "--technique %s draws splats only; points need raster\n",
                         options.technique.c_str());
            return 1;
        }
        std::optional<render::GaussianRayTracer> tracer;
        std::optional<render::ReferenceRenderer> reference;
        if (options.technique == "rt" || options.technique == "rt-bvh" || options.technique == "rt-hw") {
            render::RayTracerSettings rtSettings;
            rtSettings.route = options.technique == "rt" ? render::RayTracingRoute::Auto
                               : options.technique == "rt-hw" ? render::RayTracingRoute::Hardware
                                                              : render::RayTracingRoute::ComputeBvh;
            auto made = render::GaussianRayTracer::create(library, rtSettings);
            if (!made) {
                std::fprintf(stderr, "%s\n", made.error().toString().c_str());
                return 1;
            }
            tracer.emplace(std::move(*made));
        } else if (options.technique == "reference" || options.technique == "reference-rt") {
            auto made = render::ReferenceRenderer::create(library);
            if (!made) {
                std::fprintf(stderr, "%s\n", made.error().toString().c_str());
                return 1;
            }
            reference.emplace(std::move(*made));
        } else {
            std::fprintf(stderr, "--technique wants raster, rt, rt-hw, rt-bvh, reference or reference-rt\n");
            return 1;
        }
        double bestMs = 1e30;
        for (int round = 0; round < rounds; ++round) {
            double ms = 0.0;
            if (tracer) {
                auto stats = tracer->render(camera, instances, settings, targets);
                if (!stats) {
                    std::fprintf(stderr, "%s\n", stats.error().toString().c_str());
                    return 1;
                }
                ms = stats->totalMs;
                if (bench) {
                    std::printf("frame %d: %.2f ms (structures and colours %.2f%s, trace %.2f)\n", round, ms,
                                stats->buildMs, stats->rebuilt ? " with per-cloud build" : "",
                                stats->renderMs);
                }
            } else {
                const auto start = std::chrono::steady_clock::now();
                auto over = options.technique == "reference"
                                ? reference->render(camera, instances, settings, targets)
                                : reference->renderPeaks(camera, instances, settings, targets);
                if (!over) {
                    std::fprintf(stderr, "%s\n", over.error().toString().c_str());
                    return 1;
                }
                if (*over != 0) {
                    lrt::log::warn("{} pixels had more contributors than the reference holds", *over);
                }
                ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                if (bench) {
                    std::printf("frame %d: %.2f ms\n", round, ms);
                }
            }
            bestMs = std::min(bestMs, ms);
        }
        std::printf("%s, %ux%u: best %.2f ms\n", options.technique.c_str(), width, height, bestMs);
    }
    // Levels of detail: built once, cut every frame.
    std::optional<lod::CutSelector> cutter;
    std::vector<lod::LodCloud> lodClouds;
    std::vector<lod::LodInstance> lodInstances;
    if (options.lod > 0.0F) {
        if (options.technique != "raster" || !pointInstances.empty()) {
            std::fprintf(stderr, "--lod draws splats through the rasteriser\n");
            return 1;
        }
        auto builder = lod::LodBuilder::create(library);
        auto made = lod::CutSelector::create(library);
        if (!builder || !made) {
            std::fprintf(stderr, "%s\n", (!builder ? builder.error() : made.error()).toString().c_str());
            return 1;
        }
        cutter.emplace(std::move(*made));
        const auto start = std::chrono::steady_clock::now();
        for (const auto& cloud : clouds) {
            auto built = builder->build(*cloud);
            if (!built) {
                std::fprintf(stderr, "%s\n", built.error().toString().c_str());
                return 1;
            }
            lodClouds.push_back(std::move(*built));
        }
        std::printf("levels of detail built in %.1f ms\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        for (size_t k = 0; k < lodClouds.size(); ++k) {
            lodInstances.push_back({&lodClouds[k], model});
        }
    }
    render::FrameStats best;
    best.totalMs = 1e30;
    double bestCutMs = 0.0;
    lod::CutStats bestCut;
    for (int round = 0; round < rounds && options.technique == "raster"; ++round) {
        std::vector<render::SplatInstance> drawn = instances;
        double cutMs = 0.0;
        std::vector<lod::CutStats> cutStats;
        if (cutter) {
            const auto start = std::chrono::steady_clock::now();
            auto selected = cutter->select(render::projectionFor(camera, width, height), lodInstances, options.lod,
                                           &cutStats);
            if (!selected) {
                std::fprintf(stderr, "%s\n", selected.error().toString().c_str());
                return 1;
            }
            drawn = std::move(*selected);
            cutMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        if (rasterPoints) {
            if (auto drawn = pointRaster->render(camera, pointInstances, settings, pointLayer); !drawn) {
                std::fprintf(stderr, "%s\n", drawn.error().toString().c_str());
                return 1;
            }
        }
        auto stats = rasterPoints
                         ? rasterizer->render(camera, drawn, settings, targets, {}, &pointLayer)
                         : rasterizer->render(camera, drawn, settings, targets, pointInstances);
        if (!stats) {
            std::fprintf(stderr, "%s\n", stats.error().toString().c_str());
            return 1;
        }
        if (stats->totalMs + cutMs < best.totalMs + bestCutMs) {
            best = *stats;
            bestCutMs = cutMs;
            if (!cutStats.empty()) {
                bestCut = cutStats.front();
            }
        }
        if (bench) {
            if (cutter) {
                std::printf("frame %d: %.2f ms (cut %.2f + render %.2f)\n", round, stats->totalMs + cutMs, cutMs,
                            stats->totalMs);
            } else {
                std::printf("frame %d: %.2f ms\n", round, stats->totalMs);
            }
        }
    }
    if (options.technique == "raster") {
        if (cutter) {
            std::printf("cut: %u splats + %u merged of %u, %.2f ms; ", bestCut.splats, bestCut.merged,
                        bestCut.available, bestCutMs);
        }
        std::printf("%u splats (%u visible), %u pairs, %ux%u: best %.2f ms", best.splats,
                    best.visible, best.pairs, width, height, best.totalMs);
        if (options.stages) {
            std::printf(" (project %.2f, depth sort %.2f, counts %.2f, emit %.2f, tile sort+ranges %.2f, blend %.2f)",
                        best.projectMs, best.depthSortMs, best.countsMs, best.emitMs, best.tileSortMs,
                        best.blendMs);
        }
        std::printf("\n");
    }

    if (!bench) {
        auto colour = targets.colour.readAll<float>(**device);
        auto depth = targets.depth.readAll<float>(**device);
        if (!colour || !depth) {
            std::fprintf(stderr, "readback failed\n");
            return 1;
        }
        if (auto written = io::writeExr(options.output, width, height, *colour, *depth); !written) {
            std::fprintf(stderr, "%s\n", written.error().toString().c_str());
            return 1;
        }
        std::printf("wrote %s\n", options.output.c_str());
    }
    return 0;
}

void addOptions(CLI::App* cmd, RenderOptions& o) {
    cmd->add_option("--splats", o.splats, ".ply / .splat / .spz / .sog files");
    cmd->add_option("--technique", o.technique,
                    "raster | rt (ray traced, the device's faster route) | rt-hw (hardware "
                    "BVH) | rt-bvh (compute BVH) | reference (raster's GPU ground truth) | "
                    "reference-rt (the ray tracer's)");
    cmd->add_option("--points", o.points, "point files: .ply .xyz .txt .pts .csv points3D.txt/.bin");
    cmd->add_option("--point-route", o.pointRoute, "raster | discs");
    cmd->add_option("--point-size", o.pointSize, "point diameter (world units, or pixels with --point-pixels)");
    cmd->add_flag("--point-pixels", o.pointPixels, "point size in pixels");
    cmd->add_option("--edl", o.edl, "eye-dome lighting strength (raster route)");
    cmd->add_option("--surface", o.surface, "surface splatting depth slack (raster route)");
    cmd->add_option("--eye", o.eye, "camera position x y z")->expected(3);
    cmd->add_option("--target", o.target, "look-at point x y z")->expected(3);
    cmd->add_option("--up", o.up, "up vector")->expected(3);
    cmd->add_option("--rotate-x", o.rotateX, "turn every cloud about x (COLMAP clouds: 180)")->expected(1);
    cmd->add_option("--scale", o.scale, "scale every cloud x y z")->expected(3);
    cmd->add_option("--focal", o.focal, "focal length, mm (35mm-style 24.576 aperture)");
    cmd->add_option("--size", o.size, "WIDTHxHEIGHT");
    cmd->add_option("--near", o.nearZ, "near clipping distance");
    cmd->add_option("--degree", o.degree, "harmonic degree cap 0..3");
    cmd->add_option("--lod", o.lod, "levels of detail: pixels a merged cell may span (0: off)");
    cmd->add_flag("--no-antialias", o.noAntialias, "no Mip-Splatting 2D filter compensation");
}

}   // namespace

void addRender(CLI::App& app) {
    auto options = std::make_shared<RenderOptions>();
    auto* cmd = app.add_subcommand("render", "render splat files to an EXR");
    addOptions(cmd, *options);
    cmd->add_option("-o,--output", options->output, "EXR path");
    cmd->callback([options] {
        if (int code = run(*options, false); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

void addBench(CLI::App& app) {
    auto options = std::make_shared<RenderOptions>();
    options->repeat = 20;
    auto* cmd = app.add_subcommand("bench", "time repeated frames");
    addOptions(cmd, *options);
    cmd->add_option("--repeat", options->repeat, "frames");
    cmd->add_flag("--stages", options->stages, "wait after each stage to time it");
    cmd->callback([options] {
        if (int code = run(*options, true); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

}   // namespace lrt::cli
