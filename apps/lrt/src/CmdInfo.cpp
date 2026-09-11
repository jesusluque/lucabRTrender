// Copyright (c) 2026 lucabRTrender contributors.
#include <cstdio>
#include <string>

#include <slang.h>

#include "Commands.h"
#include "lrt/gpu/Device.h"

namespace lrt::cli {

void addInfo(CLI::App& app) {
    auto* cmd = app.add_subcommand("info", "the device, what it can do, and the toolchain");
    auto backend = std::make_shared<std::string>();
    cmd->add_option("--backend", *backend, "metal | cuda | vulkan");
    cmd->callback([backend] {
        gpu::DeviceDesc desc;
        if (*backend == "metal") desc.backends = {gpu::Backend::Metal};
        if (*backend == "cuda") desc.backends = {gpu::Backend::CUDA};
        if (*backend == "vulkan") desc.backends = {gpu::Backend::Vulkan};

        auto device = gpu::Device::create(desc);
        if (!device) {
            std::fprintf(stderr, "%s\n", device.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        const gpu::Caps& caps = (*device)->caps();
        const auto yes = [](bool b) { return b ? "yes" : "no"; };
        std::printf("backend        %s\n", gpu::toString((*device)->backend()));
        std::printf("api            %s\n", caps.apiName.c_str());
        std::printf("adapter        %s\n", caps.adapterName.c_str());
        std::printf("rasterization  %s\n", yes(caps.rasterization));
        std::printf("ray tracing    %s (pipeline), %s (ray query), %s (AS)\n",
                    yes(caps.rayTracing), yes(caps.rayQuery), yes(caps.accelerationStructure));
        std::printf("timestamps     %s\n", yes(caps.timestampQuery));
        std::printf("half           %s\n", yes(caps.half));
        std::printf("unified memory %s\n", yes(caps.unifiedMemory));
        if (caps.optixVersion != 0) {
            std::printf("optix          %u\n", caps.optixVersion);
        }
        std::printf("slang          %s\n", spGetBuildTagString());
        std::printf("shaders        %s\n", gpu::shaderDirectory().string().c_str());
    });
}

}   // namespace lrt::cli
