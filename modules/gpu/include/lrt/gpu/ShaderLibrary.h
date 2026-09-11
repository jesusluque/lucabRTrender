// Copyright (c) 2026 lucabRTrender contributors.
//
// Slang programs, compiled once per (module, entry points) and kept.
#pragma once

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "lrt/core/Result.h"

namespace lrt::gpu {

class Device;

/// A linked program and what reflection says about it.
struct Program {
    rhi::ComPtr<rhi::IShaderProgram>   program;
    rhi::ComPtr<slang::IComponentType> linked;
    /// [numthreads] of the first (compute) entry point. Read, not assumed:
    /// dispatch sizes are threads divided by this, rounded up.
    std::array<uint32_t, 3>            threadGroup{1, 1, 1};
};

class ShaderLibrary {
public:
    explicit ShaderLibrary(std::shared_ptr<Device> device);

    /// `module` is a Slang module name as `import` would write it
    /// ("lrt/algo/prefixsum"); `entries` its entry points.
    [[nodiscard]] Result<std::shared_ptr<const Program>> load(
        const std::string& module, const std::vector<std::string>& entries);

    [[nodiscard]] Device& device() noexcept { return *device_; }

private:
    std::shared_ptr<Device>                                   device_;
    std::mutex                                                guard_;
    std::map<std::string, std::shared_ptr<const Program>>     programs_;
};

}   // namespace lrt::gpu
