// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/Device.h"

#include <utility>

#include "lrt/core/Log.h"
#include "lrt/core/Platform.h"
#include "ShaderCache.h"

namespace lrt::gpu {
namespace {

rhi::DeviceType toRhi(Backend backend) {
    switch (backend) {
    case Backend::Metal: return rhi::DeviceType::Metal;
    case Backend::CUDA: return rhi::DeviceType::CUDA;
    case Backend::Vulkan: return rhi::DeviceType::Vulkan;
    case Backend::D3D12: return rhi::DeviceType::D3D12;
    }
    return rhi::DeviceType::Default;
}

std::vector<Backend> platformPreference() {
#if defined(__APPLE__)
    return {Backend::Metal};
#elif defined(_WIN32)
    return {Backend::D3D12, Backend::Vulkan};
#else
    // CUDA first: it is the backend gpe shares, so the render device and the
    // compute engine are one device. Vulkan is the answer for a machine
    // without an NVIDIA card, where gpe has no backend at all.
    return {Backend::CUDA, Backend::Vulkan};
#endif
}

/// Messages from slang-rhi's validation and the drivers, into the engine log.
class DebugLog final : public rhi::IDebugCallback {
public:
    SLANG_NO_THROW void SLANG_MCALL handleMessage(rhi::DebugMessageType type,
                                                  rhi::DebugMessageSource,
                                                  const char* message) override {
        switch (type) {
        case rhi::DebugMessageType::Error: log::error("rhi: {}", message); break;
        case rhi::DebugMessageType::Warning: log::warn("rhi: {}", message); break;
        default: log::debug("rhi: {}", message); break;
        }
    }
};

DebugLog gDebugLog;

}   // namespace

const char* toString(Backend backend) noexcept {
    switch (backend) {
    case Backend::Metal: return "Metal";
    case Backend::CUDA: return "CUDA";
    case Backend::Vulkan: return "Vulkan";
    case Backend::D3D12: return "D3D12";
    }
    return "?";
}

std::filesystem::path shaderDirectory() {
    if (std::string fromEnv = platform::env("LRT_SHADER_DIR"); !fromEnv.empty()) {
        return fromEnv;
    }
    std::error_code ignored;
    if (auto beside = platform::executableDir() / ".." / "shaders";
        std::filesystem::is_directory(beside, ignored)) {
        return std::filesystem::weakly_canonical(beside, ignored);
    }
    return LRT_BUILD_SHADER_DIR;
}

Result<std::shared_ptr<Device>> Device::create(const DeviceDesc& desc) {
    std::vector<Backend> order = desc.backends.empty() ? platformPreference() : desc.backends;
    // LRT_BACKEND=cuda|vulkan|metal|d3d12 (a comma-separated order) chooses
    // for a whole run what a caller left to the platform: how one suite is
    // run once a backend at a time.
    if (desc.backends.empty()) {
        const std::string fromEnv = platform::env("LRT_BACKEND");
        std::vector<Backend> named;
        size_t start = 0;
        while (start <= fromEnv.size() && !fromEnv.empty()) {
            const size_t comma = fromEnv.find(',', start);
            const std::string word = fromEnv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (word == "metal") named.push_back(Backend::Metal);
            else if (word == "cuda") named.push_back(Backend::CUDA);
            else if (word == "vulkan") named.push_back(Backend::Vulkan);
            else if (word == "d3d12") named.push_back(Backend::D3D12);
            else if (!word.empty()) log::warn("LRT_BACKEND: '{}' is not a backend (metal, cuda, vulkan, d3d12)", word);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        if (!named.empty()) {
            order = named;
        }
    }

    auto device = std::shared_ptr<Device>(new Device());
    device->searchPaths_.push_back(shaderDirectory().string());
    for (const auto& extra : desc.shaderPaths) {
        device->searchPaths_.push_back(extra.string());
    }
    std::vector<const char*> searchPaths;
    for (const std::string& path : device->searchPaths_) {
        searchPaths.push_back(path.c_str());
    }

    if (desc.validation) {
        rhi::getRHI()->enableDebugLayers();
    }
    if (desc.useShaderCache) {
        std::filesystem::path directory = desc.shaderCache;
        if (directory.empty()) {
            const std::string fromEnv = platform::env("LRT_SHADER_CACHE");
            directory = !fromEnv.empty() ? std::filesystem::path(fromEnv)
                                         : platform::cacheDirectory() / "lucabRTrender" / "shaders";
        }
        device->shaderCache_.attach(new DiskShaderCache(directory));
    }

    std::string failures;
    for (Backend backend : order) {
        if (!rhi::getRHI()->isDeviceTypeSupported(toRhi(backend))) {
            failures += std::format(" {} (not built in)", toString(backend));
            continue;
        }
        rhi::DeviceDesc rhiDesc;
        rhiDesc.deviceType = toRhi(backend);
        rhiDesc.enableValidation = desc.validation;
        rhiDesc.debugCallback = &gDebugLog;
        rhiDesc.slang.searchPaths = searchPaths.data();
        rhiDesc.slang.searchPathCount = static_cast<uint32_t>(searchPaths.size());
        rhiDesc.slang.optimizationLevel = SLANG_OPTIMIZATION_LEVEL_HIGH;
        rhiDesc.persistentShaderCache = device->shaderCache_.get();

        rhi::ComPtr<rhi::IDevice> made;
        if (SLANG_FAILED(rhi::getRHI()->createDevice(rhiDesc, made.writeRef())) ||
            made == nullptr) {
            failures += std::format(" {} (no device)", toString(backend));
            continue;
        }
        device->backend_ = backend;
        device->device_ = made;
        break;
    }
    if (device->device_ == nullptr) {
        return Error::make(ErrorCode::DeviceFailure, "no GPU device could be opened:{}",
                           failures);
    }

    rhi::IDevice* rhiDevice = device->device_.get();
    if (SLANG_FAILED(rhiDevice->getQueue(rhi::QueueType::Graphics,
                                         device->queue_.writeRef()))) {
        return Error(ErrorCode::DeviceFailure, "the device has no queue");
    }
    if (SLANG_FAILED(rhiDevice->getSlangSession(device->session_.writeRef()))) {
        return Error(ErrorCode::ShaderFailure, "the device has no Slang session");
    }

    const rhi::DeviceInfo& info = rhiDevice->getInfo();
    Caps& caps = device->caps_;
    caps.apiName = info.apiName != nullptr ? info.apiName : "";
    caps.adapterName = info.adapterName != nullptr ? info.adapterName : "";
    caps.optixVersion = info.optixVersion;
    caps.rasterization = rhiDevice->hasFeature(rhi::Feature::Rasterization);
    caps.rayTracing = rhiDevice->hasFeature(rhi::Feature::RayTracing);
    caps.rayQuery = rhiDevice->hasFeature(rhi::Feature::RayQuery);
    caps.accelerationStructure = rhiDevice->hasFeature(rhi::Feature::AccelerationStructure);
    caps.timestampQuery = rhiDevice->hasFeature(rhi::Feature::TimestampQuery);
    caps.half = rhiDevice->hasFeature(rhi::Feature::Half);
    // Apple silicon is unified; every discrete CUDA card is not. A Vulkan
    // integrated GPU would be, but nothing here depends on it being told.
    caps.unifiedMemory = device->backend_ == Backend::Metal;
    caps.drawIdsIncludeStart = device->backend_ == Backend::Metal;   // tests/gpu/test_textures.cpp measures it

    log::info("GPU: {} on {}", caps.apiName, caps.adapterName);
    return device;
}

Device::~Device() {
    if (queue_ != nullptr) {
        queue_->waitOnHost();
    }
}

NativeHandles Device::native() const {
    NativeHandles handles;
    rhi::DeviceNativeHandles deviceHandles;
    if (SLANG_SUCCEEDED(device_->getNativeDeviceHandles(&deviceHandles))) {
        // Metal: [0] is the MTLDevice. CUDA: [0] device, [1] context.
        // Vulkan: [0] instance, [1] physical device, [2] device.
        switch (backend_) {
        case Backend::CUDA:
            for (const rhi::NativeHandle& handle : deviceHandles.handles) {
                if (handle.type == rhi::NativeHandleType::CUcontext) {
                    handles.device = handle;
                }
            }
            break;
        case Backend::Vulkan: handles.device = deviceHandles.handles[2]; break;
        default: handles.device = deviceHandles.handles[0]; break;
        }
    }
    rhi::NativeHandle queueHandle;
    if (SLANG_SUCCEEDED(queue_->getNativeHandle(&queueHandle))) {
        handles.queue = queueHandle;
    }
    return handles;
}

ShaderCacheStats Device::shaderCacheStats() const {
    return shaderCache_ != nullptr ? static_cast<const DiskShaderCache*>(shaderCache_.get())->stats()
                                   : ShaderCacheStats{};
}

void Device::waitIdle() {
    if (queue_ != nullptr) {
        queue_->waitOnHost();
    }
}

}   // namespace lrt::gpu
