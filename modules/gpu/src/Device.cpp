// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/Device.h"

#include <string>
#include <utility>
#include <vector>

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

/// A Slang global session whose CUDA prelude does not read through the
/// read-only data cache.
///
/// Slang emits every load of a constant buffer or a read-only buffer on CUDA
/// as `__ldg`, the load that goes through that cache. Inside a ray tracing
/// launch the cache is served lines that no launch invalidated:
/// `tests/gpu/test_uniforms.cpp` measures a ray generation entry reading the
/// value of the launch before it 31 times in 64, while a writable buffer over
/// the same address is right every time and a compute launch is right on all
/// three routes. That is a frame drawn with the parameters of the frame
/// before, and no image says so.
///
/// The prelude is where this can be said once: the macro is appended after
/// Slang's own declaration of `__ldg`, so the declaration still parses and
/// every call site that follows is an ordinary load. It costs that cache and
/// nothing else. nvrtc cannot be told this instead -- Slang keeps one
/// `DownstreamArgs` entry per downstream compiler, and that one is already
/// the OptiX include path.
slang::IGlobalSession* cudaGlobalSession() {
    static Slang::ComPtr<slang::IGlobalSession> session = [] {
        Slang::ComPtr<slang::IGlobalSession> made;
        if (SLANG_FAILED(slang::createGlobalSession(made.writeRef())) || made == nullptr) {
            log::warn("CUDA: no Slang session of our own; a ray tracing launch may read the launch before it");
            return Slang::ComPtr<slang::IGlobalSession>();
        }
        Slang::ComPtr<ISlangBlob> prelude;
        made->getLanguagePrelude(SLANG_SOURCE_LANGUAGE_CUDA, prelude.writeRef());
        std::string text;
        if (prelude != nullptr && prelude->getBufferPointer() != nullptr) {
            text.assign(static_cast<const char*>(prelude->getBufferPointer()), prelude->getBufferSize());
        }
        // LRT_CUDA_LDG=1 leaves the cache in place, which is how the defect
        // is reproduced and how the cost of giving it up is measured.
        if (platform::env("LRT_CUDA_LDG") != "1") {
            text += "\n#undef __ldg\n#define __ldg(p) (*(p))\n";
        }
        made->setLanguagePrelude(SLANG_SOURCE_LANGUAGE_CUDA, text.c_str());
        return made;
    }();
    return session.get();
}

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
        // The cache is keyed by slang-rhi, which knows nothing of the prelude
        // this process hands Slang (cudaGlobalSession below). A cache written
        // before that prelude existed holds CUDA code that reads through the
        // read-only data cache, and serving it back would put the defect
        // quietly under a fixed build. The generation is part of the path, so
        // a changed prelude simply looks elsewhere.
        directory /= platform::env("LRT_CUDA_LDG") == "1" ? "gen1-ldg" : "gen1";
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
        // CUDA compiles through nvrtc at run time, and a kernel that traces
        // includes <optix.h>. Slang looks for an installed OptiX SDK and
        // there is none here, so nvrtc is handed the headers slang-rhi
        // fetched at build time (LRT_OPTIX_INCLUDE_DIR), or whatever
        // LRT_OPTIX_INCLUDE says. Without it every ray tracing pipeline on
        // CUDA fails with "Failed to locate OptiX headers".
        std::vector<std::string> cudaArgs;
        std::vector<slang::CompilerOptionEntry> cudaEntries;
        if (backend == Backend::CUDA) {
            std::string include = platform::env("LRT_OPTIX_INCLUDE");
#if defined(LRT_OPTIX_INCLUDE_DIR)
            if (include.empty()) {
                include = LRT_OPTIX_INCLUDE_DIR;
            }
#endif
            if (!include.empty()) {
                cudaArgs.push_back("-I" + include);
            }
            for (const std::string& arg : cudaArgs) {
                slang::CompilerOptionEntry entry{};
                entry.name = slang::CompilerOptionName::DownstreamArgs;
                entry.value.kind = slang::CompilerOptionValueKind::String;
                entry.value.stringValue0 = "nvrtc";
                entry.value.stringValue1 = arg.c_str();
                cudaEntries.push_back(entry);
            }
            rhiDesc.slang.compilerOptionEntries = cudaEntries.data();
            rhiDesc.slang.compilerOptionEntryCount = static_cast<uint32_t>(cudaEntries.size());
            rhiDesc.slang.slangGlobalSession = cudaGlobalSession();
        }
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
    caps.convertingStores = device->backend_ != Backend::CUDA;       // and this

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
