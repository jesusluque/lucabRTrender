// Copyright (c) 2026 lucabRTrender contributors.
//
// `lrt aofx list` and `lrt aofx run`: the AOFX host from a terminal. What a
// bundle declares, why one was refused, and one effect over EXR files.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Commands.h"
#include "aofx/Effect.h"
#include "lrt/aofx/EffectRegistry.h"
#include "lrt/aofx/EffectRender.h"
#include "lrt/gpu_host/Context.h"
#include "lrt/io/Exr.h"

namespace lrt::cli {
namespace {

const char* typeName(aofx::ParamType type) {
    switch (type) {
    case aofx::ParamType::Double: return "double";
    case aofx::ParamType::Integer: return "integer";
    case aofx::ParamType::Boolean: return "boolean";
    case aofx::ParamType::Choice: return "choice";
    case aofx::ParamType::String: return "string";
    case aofx::ParamType::Colour: return "colour";
    default: return "other";
    }
}

struct Common {
    std::vector<std::string> paths;
};

gpu_host::Context* openContext() {
    gpu_host::Context* context = gpu_host::installProcessContext();
    if (context == nullptr || context->compute() == nullptr) {
        std::fprintf(stderr, "no GPU compute device for AOFX kernels (gpe has no backend here)\n");
        return nullptr;
    }
    return context;
}

void scan(aofx_host::EffectRegistry& registry, const Common& common, gpu_host::Context* context) {
    for (const std::string& path : common.paths) {
        registry.addSearchPath(path);
    }
#ifdef LRT_AOFX_BUNDLE_DIR
    registry.addSearchPath(LRT_AOFX_BUNDLE_DIR);
#endif
    registry.scan(context);
}

/// "name=1", "name=0.2,0.4", "name=text".
bool parseParam(const std::string& text, aofx::ParamValue& out) {
    const size_t eq = text.find('=');
    if (eq == std::string::npos || eq == 0) {
        return false;
    }
    out.name = text.substr(0, eq);
    const std::string value = text.substr(eq + 1);
    std::stringstream items(value);
    std::string item;
    std::vector<double> numbers;
    while (std::getline(items, item, ',')) {
        char* end = nullptr;
        const double number = std::strtod(item.c_str(), &end);
        if (end == item.c_str() || *end != '\0') {
            out.numbers.clear();
            out.text = value;
            return true;
        }
        numbers.push_back(number);
    }
    out.numbers = std::move(numbers);
    return true;
}

}   // namespace

void addAofx(CLI::App& app) {
    auto* group = app.add_subcommand("aofx", "AOFX effects: list the bundles found, run one over EXR files");
    group->require_subcommand(1);
    auto common = std::make_shared<Common>();

    auto* list = group->add_subcommand("list", "every bundle found, loaded or refused, and what it declares");
    list->add_option("--path", common->paths, "extra bundle directories (after $AOFX_PLUGIN_PATH)");
    list->callback([common] {
        gpu_host::Context* context = openContext();
        if (context == nullptr) {
            throw CLI::RuntimeError(1);
        }
        aofx_host::EffectRegistry registry;
        scan(registry, *common, context);
        std::printf("search path:\n");
        for (const auto& path : registry.searchPaths()) {
            std::printf("  %s\n", path.string().c_str());
        }
        for (const auto& report : registry.reports()) {
            std::printf("%s %s\n", report.loaded ? "loaded " : "refused", report.path.string().c_str());
            if (!report.loaded) {
                std::printf("    %s\n", report.reason.c_str());
            }
            for (const std::string& id : report.effects) {
                const auto found = registry.descriptions().find(id);
                if (found == registry.descriptions().end()) {
                    continue;
                }
                const aofx::EffectDesc& desc = found->second;
                std::printf("    %s  \"%s\" (%s)\n", desc.identifier.c_str(), desc.label.c_str(),
                            desc.grouping.c_str());
                for (const aofx::ClipDesc& clip : desc.inputs) {
                    if (!clip.hostWired) {
                        std::printf("      input %s%s\n", clip.name.c_str(), clip.optional ? " (optional)" : "");
                    }
                }
                for (const aofx::ParamDesc& param : desc.params) {
                    std::printf("      param %s: %s", param.name.c_str(), typeName(param.type));
                    if (param.dimension > 1) {
                        std::printf("[%d]", param.dimension);
                    }
                    if (!param.defaults.empty()) {
                        std::printf(" =");
                        for (double v : param.defaults) {
                            std::printf(" %g", v);
                        }
                    }
                    std::printf("\n");
                }
            }
        }
    });

    struct RunOptions {
        std::string              effect;
        std::vector<std::string> inputs;
        std::vector<std::string> params;
        std::string              output = "out.exr";
        double                   time = 0.0;
    };
    auto run = std::make_shared<RunOptions>();
    auto* runCmd = group->add_subcommand("run", "one effect over EXR files");
    runCmd->add_option("effect", run->effect, "effect identifier, e.g. tv.mediapro.aofx.invert")->required();
    runCmd->add_option("inputs", run->inputs,
                       "input EXRs, in the effect's input order; or Clip=path to name the input")
        ->required();
    runCmd->add_option("-o,--output", run->output, "EXR path");
    runCmd->add_option("--param", run->params, "name=value or name=v1,v2,... (repeatable)");
    runCmd->add_option("--time", run->time, "frame time");
    runCmd->add_option("--path", common->paths, "extra bundle directories (after $AOFX_PLUGIN_PATH)");
    runCmd->callback([common, run] {
        gpu_host::Context* context = openContext();
        if (context == nullptr) {
            throw CLI::RuntimeError(1);
        }
        aofx_host::EffectRegistry registry;
        scan(registry, *common, context);
        aofx::Effect* effect = registry.find(run->effect);
        const auto desc = registry.descriptions().find(run->effect);
        if (effect == nullptr || desc == registry.descriptions().end()) {
            std::fprintf(stderr, "no effect '%s' (try `lrt aofx list`)\n", run->effect.c_str());
            throw CLI::RuntimeError(1);
        }
        std::vector<const aofx::ClipDesc*> ports;
        for (const aofx::ClipDesc& clip : desc->second.inputs) {
            if (!clip.hostWired) {
                ports.push_back(&clip);
            }
        }
        aofx_host::EffectJob job;
        job.time = run->time;
        uint32_t width = 0, height = 0;
        for (size_t k = 0; k < run->inputs.size(); ++k) {
            std::string clip;
            std::string path = run->inputs[k];
            if (const size_t eq = path.find('='); eq != std::string::npos) {
                clip = path.substr(0, eq);
                path = path.substr(eq + 1);
            } else if (k < ports.size()) {
                clip = ports[k]->name;
            } else {
                std::fprintf(stderr, "input %zu (%s): the effect has %zu inputs\n", k + 1, path.c_str(),
                             ports.size());
                throw CLI::RuntimeError(1);
            }
            auto pixels = io::readExr(path);
            if (!pixels) {
                std::fprintf(stderr, "%s\n", pixels.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            auto image = image::Image::create(
                {0, 0, static_cast<int32_t>(pixels->width), static_cast<int32_t>(pixels->height)});
            if (!image) {
                std::fprintf(stderr, "%s\n", image.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            {
                auto floats = (*image)->floats();
                const int stride = (*image)->stride();
                for (uint32_t y = 0; y < pixels->height; ++y) {
                    for (uint32_t x = 0; x < pixels->width; ++x) {
                        for (int c = 0; c < 4; ++c) {
                            floats[(size_t{y} * static_cast<size_t>(stride) + x) * 4 + static_cast<size_t>(c)] =
                                pixels->rgba[(size_t{y} * pixels->width + x) * 4 + static_cast<size_t>(c)];
                        }
                    }
                }
            }
            if (job.inputs.empty()) {
                width = pixels->width;
                height = pixels->height;
            }
            job.inputs.push_back({clip, *image});
        }
        for (const std::string& text : run->params) {
            aofx::ParamValue value;
            if (!parseParam(text, value)) {
                std::fprintf(stderr, "--param wants name=value, got '%s'\n", text.c_str());
                throw CLI::RuntimeError(1);
            }
            job.params.push_back(std::move(value));
        }
        auto rendered = aofx_host::renderEffect(*context, *effect, job);
        if (!rendered) {
            std::fprintf(stderr, "%s\n", rendered.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        const image::Image& out = **rendered;
        width = static_cast<uint32_t>(out.bounds().width());
        height = static_cast<uint32_t>(out.bounds().height());
        std::vector<float> rgba(size_t{width} * height * 4);
        const auto floats = out.floats();
        const int stride = out.stride();
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                for (int c = 0; c < 4; ++c) {
                    rgba[(size_t{y} * width + x) * 4 + static_cast<size_t>(c)] =
                        floats[(size_t{y} * static_cast<size_t>(stride) + x) * 4 + static_cast<size_t>(c)];
                }
            }
        }
        if (auto written = io::writeExr(run->output, width, height, rgba, {}, false); !written) {
            std::fprintf(stderr, "%s\n", written.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        std::printf("%s: wrote %s (%ux%u)\n", run->effect.c_str(), run->output.c_str(), width, height);
    });
}

}   // namespace lrt::cli
