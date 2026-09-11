// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/Denoiser.h"

#include "lrt/gpu/Device.h"

#if LRT_HAVE_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

namespace lrt::technique {

struct Denoiser::Impl {
    std::string description;
#if LRT_HAVE_OIDN
    OIDNDevice device = nullptr;
    ~Impl() {
        if (device != nullptr) {
            oidnReleaseDevice(device);
        }
    }
#endif
};

bool denoiserBuilt() noexcept {
#if LRT_HAVE_OIDN
    return true;
#else
    return false;
#endif
}

const std::string& Denoiser::description() const noexcept {
    return impl_->description;
}

Result<Denoiser> Denoiser::create(gpu::Device& device) {
#if LRT_HAVE_OIDN
    auto impl = std::make_shared<Impl>();
    const gpu::NativeHandles native = device.native();
    const char* on = "";
    switch (device.backend()) {
    case gpu::Backend::Metal: {
        auto queue = static_cast<MTLCommandQueue_id>(reinterpret_cast<void*>(native.queue.value));
        impl->device = oidnNewMetalDevice(&queue, 1);
        on = "Metal";
        break;
    }
    case gpu::Backend::CUDA: {
        const int ids[1] = {0};
        auto stream = static_cast<cudaStream_t>(reinterpret_cast<void*>(native.queue.value));
        impl->device = oidnNewCUDADevice(ids, &stream, 1);
        on = "CUDA";
        break;
    }
    default:
        return Error(ErrorCode::Unsupported, "no denoiser on this backend: OIDN runs on Metal and CUDA");
    }
    if (impl->device == nullptr) {
        return Error(ErrorCode::DeviceFailure, std::string("OIDN could not open a ") + on + " device");
    }
    oidnCommitDevice(impl->device);
    const char* message = nullptr;
    if (oidnGetDeviceError(impl->device, &message) != OIDN_ERROR_NONE) {
        return Error(ErrorCode::DeviceFailure, std::string("OIDN: ") + (message != nullptr ? message : "unknown"));
    }
    impl->description = std::string("OIDN ") + OIDN_VERSION_STRING + " on " + on;
    Denoiser denoiser;
    denoiser.impl_ = std::move(impl);
    return denoiser;
#else
    (void)device;
    return Error(ErrorCode::Unsupported, "built without OIDN (scripts/build-oidn.sh, LRT_OIDN_ROOT)");
#endif
}

}   // namespace lrt::technique
