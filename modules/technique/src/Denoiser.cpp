// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/Denoiser.h"

#include "lrt/core/Platform.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

#include <optional>

#if LRT_HAVE_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

namespace lrt::technique {

struct Denoiser::Impl {
    std::string description;
    gpu::Device* gpu = nullptr;
    std::optional<gpu::ComputeKernel> copy;   ///< buffer_copy: slang-rhi's Metal blit skips wrapped buffers
#if LRT_HAVE_OIDN
    OIDNDevice device = nullptr;
    OIDNFilter filter = nullptr;   ///< "RT", kept between frames; images rebound each call
    bool       metal = false;
    /// On Metal, OIDN shares only buffers with hazard tracking, and slang-rhi
    /// makes none (it orders its own work). So each image goes through a
    /// staging buffer the platform makes tracked: the engine's buffer is copied
    /// into it on the device, OIDN reads and writes it, and the output is
    /// copied back -- device to device throughout, nothing through the host.
    struct Staging {
        void*       native = nullptr;   ///< the MTLBuffer, ours to release
        gpu::Buffer wrapped;            ///< the same buffer as slang-rhi sees it, for the copies
        OIDNBuffer  shared = nullptr;   ///< the same buffer as OIDN sees it
        uint64_t    bytes = 0;
    };
    Staging colour, albedo, normal, output;
    void drop(Staging& s) {
        if (s.shared != nullptr) {
            oidnReleaseBuffer(s.shared);
            s.shared = nullptr;
        }
        s.wrapped = gpu::Buffer();
        if (s.native != nullptr) {
            platform::releaseMetalBuffer(s.native);
            s.native = nullptr;
        }
        s.bytes = 0;
    }
    Result<void> stage(Staging& s, uint64_t bytes, const char* label) {
        if (s.bytes == bytes && s.native != nullptr) {
            return ok();
        }
        drop(s);
        s.native = platform::newTrackedMetalBuffer(reinterpret_cast<void*>(gpu->native().device.value), bytes);
        if (s.native == nullptr) {
            return Error(ErrorCode::DeviceFailure, std::string("no tracked Metal buffer for ") + label);
        }
        rhi::NativeHandle handle;
        handle.type = rhi::NativeHandleType::MTLBuffer;
        handle.value = reinterpret_cast<uint64_t>(s.native);
        gpu::BufferDesc desc;
        desc.bytes = bytes;
        desc.elementBytes = 16;
        desc.label = label;
        auto wrapped = gpu::Buffer::wrap(*gpu, handle, desc);
        if (!wrapped) {
            drop(s);
            return std::move(wrapped).error();
        }
        s.wrapped = std::move(*wrapped);
        s.shared = oidnNewSharedBufferFromMetal(device, static_cast<MTLBuffer_id>(s.native));
        if (s.shared == nullptr) {
            const char* message = nullptr;
            oidnGetDeviceError(device, &message);
            drop(s);
            return Error(ErrorCode::DeviceFailure,
                         std::string("OIDN would not share the staging buffer: ") + (message ? message : "?"));
        }
        s.bytes = bytes;
        return ok();
    }
    ~Impl() {
        drop(colour);
        drop(albedo);
        drop(normal);
        drop(output);
        if (filter != nullptr) {
            oidnReleaseFilter(filter);
        }
        if (device != nullptr) {
            oidnReleaseDevice(device);
        }
    }
    /// A CUDA buffer as OIDN sees it: the same memory, no copy.
    OIDNBuffer shareDirect(const gpu::Buffer& buffer) const {
        const rhi::NativeHandle handle = buffer.native();
        return oidnNewSharedBuffer(device, reinterpret_cast<void*>(handle.value), buffer.bytes());
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

Result<Denoiser> Denoiser::create(gpu::ShaderLibrary& library) {
    auto made = create(library.device());
    if (!made) return made;
    auto copy = gpu::ComputeKernel::create(library, "lrt/technique/buffer_copy", "copyWords");
    if (!copy) return std::move(copy).error();
    made->impl_->copy.emplace(std::move(*copy));
    return made;
}

Result<Denoiser> Denoiser::create(gpu::Device& device) {
#if LRT_HAVE_OIDN
    auto impl = std::make_shared<Impl>();
    impl->gpu = &device;
    const gpu::NativeHandles native = device.native();
    const char* on = "";
    switch (device.backend()) {
    case gpu::Backend::Metal: {
        auto queue = static_cast<MTLCommandQueue_id>(reinterpret_cast<void*>(native.queue.value));
        impl->device = oidnNewMetalDevice(&queue, 1);
        impl->metal = true;
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
    impl->filter = oidnNewFilter(impl->device, "RT");
    if (impl->filter == nullptr) {
        return Error(ErrorCode::DeviceFailure, "OIDN: no RT filter on this device");
    }
    Denoiser denoiser;
    denoiser.impl_ = std::move(impl);
    return denoiser;
#else
    (void)device;
    return Error(ErrorCode::Unsupported, "built without OIDN (scripts/build-oidn.sh, LRT_OIDN_ROOT)");
#endif
}

Result<void> Denoiser::denoise(const gpu::Buffer& colour, const gpu::Buffer* albedo, const gpu::Buffer* normal,
                               gpu::Buffer& out, uint32_t width, uint32_t height) {
#if LRT_HAVE_OIDN
    Impl& impl = *impl_;
    const size_t pixelStride = 16;   // float4 a pixel; OIDN reads the first three
    const size_t rowStride = pixelStride * width;
    const uint64_t bytes = uint64_t{width} * height * pixelStride;

    OIDNBuffer colourShared = nullptr;
    OIDNBuffer outShared = nullptr;
    OIDNBuffer albedoShared = nullptr;
    OIDNBuffer normalShared = nullptr;
    if (impl.metal) {
        if (!impl.copy.has_value()) {
            return Error(ErrorCode::InvalidArgument,
                         "denoiser made without a shader library: on Metal the staging copies are a kernel");
        }
        const auto copyWords = [&](gpu::CommandBatch& batch, const gpu::Buffer& from, const gpu::Buffer& to) {
            const uint32_t words = static_cast<uint32_t>(bytes / 4);
            impl.copy->dispatch(batch, {words, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["src"].setBinding(from.rhi());
                cursor["dst"].setBinding(to.rhi());
                cursor["copy"]["words"].setData(words);
            });
        };
        LRT_TRY(impl.stage(impl.colour, bytes, "denoise.colour"));
        LRT_TRY(impl.stage(impl.output, bytes, "denoise.output"));
        if (albedo != nullptr) LRT_TRY(impl.stage(impl.albedo, bytes, "denoise.albedo"));
        if (normal != nullptr) LRT_TRY(impl.stage(impl.normal, bytes, "denoise.normal"));
        // In: the engine's buffers into the tracked ones, on the device.
        gpu::CommandBatch batch(*impl.gpu);
        copyWords(batch, colour, impl.colour.wrapped);
        // The output staging starts as the colour too: OIDN writes three of a
        // pixel's four floats and leaves the fourth -- the alpha -- alone, and
        // a fresh private buffer holds anything there. Seeded from the input,
        // the alpha that comes back is the input's.
        copyWords(batch, colour, impl.output.wrapped);
        if (albedo != nullptr) copyWords(batch, *albedo, impl.albedo.wrapped);
        if (normal != nullptr) copyWords(batch, *normal, impl.normal.wrapped);
        LRT_TRY(batch.submit(true));
        colourShared = impl.colour.shared;
        outShared = impl.output.shared;
        albedoShared = albedo != nullptr ? impl.albedo.shared : nullptr;
        normalShared = normal != nullptr ? impl.normal.shared : nullptr;
    } else {
        colourShared = impl.shareDirect(colour);
        outShared = impl.shareDirect(out);
        albedoShared = albedo != nullptr ? impl.shareDirect(*albedo) : nullptr;
        normalShared = normal != nullptr ? impl.shareDirect(*normal) : nullptr;
        if (colourShared == nullptr || outShared == nullptr || (albedo != nullptr && albedoShared == nullptr) ||
            (normal != nullptr && normalShared == nullptr)) {
            const char* message = nullptr;
            oidnGetDeviceError(impl.device, &message);
            return Error(ErrorCode::DeviceFailure,
                         std::string("OIDN could not share the engine's buffers: ") + (message ? message : "?"));
        }
    }
    oidnSetFilterImage(impl.filter, "color", colourShared, OIDN_FORMAT_FLOAT3, width, height, 0, pixelStride,
                       rowStride);
    oidnSetFilterImage(impl.filter, "output", outShared, OIDN_FORMAT_FLOAT3, width, height, 0, pixelStride,
                       rowStride);
    if (albedoShared != nullptr) {
        oidnSetFilterImage(impl.filter, "albedo", albedoShared, OIDN_FORMAT_FLOAT3, width, height, 0, pixelStride,
                           rowStride);
    } else {
        oidnUnsetFilterImage(impl.filter, "albedo");
    }
    if (normalShared != nullptr) {
        oidnSetFilterImage(impl.filter, "normal", normalShared, OIDN_FORMAT_FLOAT3, width, height, 0, pixelStride,
                           rowStride);
    } else {
        oidnUnsetFilterImage(impl.filter, "normal");
    }
    oidnSetFilterBool(impl.filter, "hdr", true);
    oidnCommitFilter(impl.filter);
    oidnExecuteFilter(impl.filter);
    const char* message = nullptr;
    const OIDNError error = oidnGetDeviceError(impl.device, &message);
    if (!impl.metal) {
        oidnReleaseBuffer(colourShared);
        oidnReleaseBuffer(outShared);
        if (albedoShared != nullptr) oidnReleaseBuffer(albedoShared);
        if (normalShared != nullptr) oidnReleaseBuffer(normalShared);
    }
    if (error != OIDN_ERROR_NONE) {
        return Error(ErrorCode::DeviceFailure, std::string("OIDN: ") + (message != nullptr ? message : "unknown"));
    }
    if (impl.metal) {
        // Out: the denoised staging back into the engine's buffer.
        gpu::CommandBatch batch(*impl.gpu);
        const uint32_t words = static_cast<uint32_t>(bytes / 4);
        impl.copy->dispatch(batch, {words, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["src"].setBinding(impl.output.wrapped.rhi());
            cursor["dst"].setBinding(out.rhi());
            cursor["copy"]["words"].setData(words);
        });
        LRT_TRY(batch.submit(true));
    }
    return ok();
#else
    (void)colour; (void)albedo; (void)normal; (void)out; (void)width; (void)height;
    return Error(ErrorCode::Unsupported, "built without OIDN");
#endif
}

}   // namespace lrt::technique
