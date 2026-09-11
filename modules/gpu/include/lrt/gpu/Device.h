// Copyright (c) 2026 lucabRTrender contributors.
//
// The render device: slang-rhi, one per process.
//
// WHY SLANG-RHI AND NOT GPE FOR RENDERING
//
// gpe is a compute engine with kernels compiled at build time and bound by
// position. That is right for image effects and wrong for a renderer that
// needs a rasterisation pipeline for points, acceleration structures for ray
// tracing, and bindings a person cannot get wrong by counting. slang-rhi gives
// all three on Metal, CUDA (with OptiX) and Vulkan, from one Slang source
// compiled for whichever device the process opened.
//
// gpe is not replaced: it adopts this device (see lrt::gpu_host::RhiBridge),
// so the two share memory and nothing crosses between them.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "lrt/core/Result.h"

namespace lrt::gpu {

enum class Backend { Metal, CUDA, Vulkan, D3D12 };

[[nodiscard]] const char* toString(Backend) noexcept;

struct DeviceDesc {
    /// Empty means the platform's preference: Metal on Apple, CUDA then Vulkan
    /// on Linux, D3D12 then Vulkan on Windows. There is no CPU entry and there
    /// never will be.
    std::vector<Backend> backends;
    /// slang-rhi's own validation layer. Slow; for tests and debugging.
    bool validation = false;
    /// Extra directories searched for `import`ed Slang modules, after the
    /// engine's own shader directory.
    std::vector<std::filesystem::path> shaderPaths;
};

/// What the device can do, asked once at creation.
struct Caps {
    std::string apiName;
    std::string adapterName;
    bool rasterization = false;
    bool rayTracing = false;       ///< ray tracing pipelines (shader tables)
    bool rayQuery = false;         ///< inline RayQuery in compute
    bool accelerationStructure = false;
    bool timestampQuery = false;
    bool half = false;
    bool unifiedMemory = false;
    uint32_t optixVersion = 0;
};

/// The native objects behind the device, for adopting it elsewhere (gpe).
struct NativeHandles {
    rhi::NativeHandle device;    ///< MTLDevice | CUcontext | VkDevice
    rhi::NativeHandle queue;     ///< MTLCommandQueue | CUstream | VkQueue
};

class Device {
public:
    [[nodiscard]] static Result<std::shared_ptr<Device>> create(const DeviceDesc& desc = {});

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device();

    [[nodiscard]] Backend backend() const noexcept { return backend_; }
    [[nodiscard]] const Caps& caps() const noexcept { return caps_; }
    [[nodiscard]] NativeHandles native() const;

    [[nodiscard]] rhi::IDevice* rhi() const noexcept { return device_.get(); }
    [[nodiscard]] rhi::ICommandQueue* queue() const noexcept { return queue_.get(); }
    [[nodiscard]] slang::ISession* slangSession() const noexcept { return session_.get(); }

    /// The directories Slang searches, in order, engine shaders first.
    [[nodiscard]] const std::vector<std::string>& shaderSearchPaths() const noexcept {
        return searchPaths_;
    }

    /// Blocks until everything submitted has run.
    void waitIdle();

private:
    Device() = default;

    Backend                         backend_ = Backend::Metal;
    Caps                            caps_;
    std::vector<std::string>        searchPaths_;
    rhi::ComPtr<rhi::IDevice>       device_;
    rhi::ComPtr<rhi::ICommandQueue> queue_;
    rhi::ComPtr<slang::ISession>    session_;
};

/// Where the engine's own .slang files are: $LRT_SHADER_DIR, then
/// <exe>/../shaders, then the build tree.
[[nodiscard]] std::filesystem::path shaderDirectory();

}   // namespace lrt::gpu
