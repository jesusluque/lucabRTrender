// Copyright (c) 2026 lucabRTrender contributors.
//
// The GPU thread is openFXplayer's gpu_host::Context, ported; the device
// underneath is new (slang-rhi, adopted by gpe).
#include "lrt/gpu_host/Context.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "gpe/adopt.h"
#include "gpe/device.h"
#include "gpe/pool.h"
#include "lrt/core/Log.h"
#include "lrt/gpu_host/ImageStorage.h"

namespace lrt::gpu_host {
namespace {

Context* gCurrent = nullptr;

struct Job {
    const std::function<void()>* work = nullptr;
    std::exception_ptr           failure;
    bool                         done = false;
};

/// gpe on slang-rhi's device, or null where gpe has no backend for it.
std::unique_ptr<gpe::Device> adoptForCompute(gpu::Device& device) {
    const gpu::NativeHandles native = device.native();
    switch (device.backend()) {
    case gpu::Backend::Metal:
        return gpe::adoptMetalDevice(reinterpret_cast<void*>(native.device.value),
                                     reinterpret_cast<void*>(native.queue.value));
    case gpu::Backend::CUDA:
        return gpe::adoptCudaContext(reinterpret_cast<void*>(native.device.value),
                                     reinterpret_cast<void*>(native.queue.value));
    default:
        return nullptr;
    }
}

rhi::NativeHandleType bufferHandleType(gpu::Backend backend) {
    switch (backend) {
    case gpu::Backend::Metal: return rhi::NativeHandleType::MTLBuffer;
    case gpu::Backend::CUDA: return rhi::NativeHandleType::CUdeviceptr;
    default: return rhi::NativeHandleType::Undefined;
    }
}

}   // namespace

struct Context::Impl {
    std::shared_ptr<gpu::Device>       device;
    std::shared_ptr<gpe::PooledDevice> compute;
    std::shared_ptr<ImageStorage>      storage;
    std::string                        backendName;
    bool                               installed = false;

    std::thread             worker;
    std::mutex              guard;
    std::condition_variable arrived;
    std::condition_variable finished;
    std::deque<Job*>        queue;
    bool                    stopping = false;
};

Context::Context() : impl_(std::make_unique<Impl>()) {}

Context::~Context() {
    if (impl_->worker.joinable()) {
        {
            const std::lock_guard<std::mutex> held(impl_->guard);
            impl_->stopping = true;
        }
        impl_->arrived.notify_all();
        impl_->worker.join();
    }
    if (impl_->installed) {
        image::setStorage(nullptr);
        if (gCurrent == this) {
            gCurrent = nullptr;
        }
    }
    // Images first, then gpe, then the device both of them sit on.
    impl_->storage.reset();
    if (impl_->compute) {
        impl_->compute->sync();
    }
    impl_->compute.reset();
    impl_->device.reset();
}

Result<std::unique_ptr<Context>> Context::create(const ContextDesc& desc) {
    auto device = gpu::Device::create(desc.device);
    if (!device) {
        return std::move(device).error();
    }

    auto context = std::unique_ptr<Context>(new Context());
    Impl& impl = *context->impl_;
    impl.device = *device;
    impl.backendName = gpu::toString(impl.device->backend());

    if (std::unique_ptr<gpe::Device> native = adoptForCompute(*impl.device); native != nullptr) {
        const bool unified = impl.device->caps().unifiedMemory;
        size_t total = 0;
        size_t available = 0;
        native->memory(total, available);
        // openFXplayer's rule: a discrete card's memory is for this and nothing
        // else, so the pool may take three quarters of what is free; on
        // unified memory every byte is the machine's, and the caller's figure
        // stands.
        const size_t poolBudget =
            unified ? desc.budgetBytes : std::max(desc.budgetBytes, available / 4 * 3);
        impl.compute = std::make_shared<gpe::PooledDevice>(std::move(native), poolBudget);
        // gpe batches its dispatches on the shared queue; whatever slang-rhi
        // submits must follow them, so it hands the batch over first.
        impl.device->setBeforeSubmit(
            [weak = std::weak_ptr<gpe::PooledDevice>(impl.compute)] {
                if (const auto compute = weak.lock()) {
                    compute->flush();
                }
            });
        impl.storage = std::make_shared<ImageStorage>(
            impl.compute, poolBudget / 2,
            unified ? ImageStorage::Mode::Shared : ImageStorage::Mode::Paired);
        log::info("gpe shares the {} device ({} images)", impl.backendName,
                  unified ? "shared" : "paired");
    } else {
        log::warn("gpe has no backend for {}: aofx effects are unavailable", impl.backendName);
    }

    Impl* raw = &impl;
    impl.worker = std::thread([raw] {
        // A CUDA context is current per thread, and this thread is new. gpe
        // binds the context for its own calls, but the work queued here
        // reaches the same context through slang-rhi, whose allocations would
        // otherwise be the first driver calls on a thread that has no context
        // at all -- which the driver reports as a failure to allocate, on a
        // card with everything free.
        if (raw->compute != nullptr) {
            raw->compute->bindThread();
        }
        for (;;) {
            Job* job = nullptr;
            {
                std::unique_lock<std::mutex> held(raw->guard);
                raw->arrived.wait(held, [raw] { return raw->stopping || !raw->queue.empty(); });
                if (raw->stopping && raw->queue.empty()) {
                    return;
                }
                job = raw->queue.front();
                raw->queue.pop_front();
            }
            try {
                (*job->work)();
            } catch (...) {
                job->failure = std::current_exception();
            }
            {
                const std::lock_guard<std::mutex> held(raw->guard);
                job->done = true;
            }
            raw->finished.notify_all();
        }
    });
    return context;
}

void Context::install() {
    if (!impl_->installed) {
        image::setStorage(impl_->storage);
        gCurrent = this;
        impl_->installed = true;
    }
}

Context* Context::current() noexcept { return gCurrent; }

Result<void> Context::run(const std::function<void()>& work, Priority priority) {
    if (!impl_->worker.joinable()) {
        return Error(ErrorCode::InternalError, "the GPU thread is not running");
    }
    if (std::this_thread::get_id() == impl_->worker.get_id()) {
        try {
            work();
        } catch (const std::exception& thrown) {
            return Error(ErrorCode::InternalError, thrown.what());
        }
        return ok();
    }
    Job job;
    job.work = &work;
    {
        const std::lock_guard<std::mutex> held(impl_->guard);
        if (impl_->stopping) {
            return Error(ErrorCode::Cancelled, "the GPU thread is shutting down");
        }
        if (priority == Priority::Interactive) {
            impl_->queue.push_front(&job);
        } else {
            impl_->queue.push_back(&job);
        }
    }
    impl_->arrived.notify_one();
    {
        std::unique_lock<std::mutex> held(impl_->guard);
        impl_->finished.wait(held, [&job] { return job.done; });
    }
    if (job.failure) {
        try {
            std::rethrow_exception(job.failure);
        } catch (const std::exception& thrown) {
            return Error(ErrorCode::InternalError, thrown.what());
        } catch (...) {
            return Error(ErrorCode::InternalError, "the GPU thread threw a non-exception");
        }
    }
    return ok();
}

gpu::Device& Context::device() noexcept { return *impl_->device; }
std::shared_ptr<gpu::Device> Context::deviceShared() const noexcept { return impl_->device; }
gpe::PooledDevice* Context::compute() noexcept { return impl_->compute.get(); }
std::shared_ptr<gpe::PooledDevice> Context::computeShared() const noexcept {
    return impl_->compute;
}
ImageStorage* Context::sharedStorage() noexcept { return impl_->storage.get(); }
const std::string& Context::backendName() const noexcept { return impl_->backendName; }

Result<gpu::Buffer> Context::renderView(uint64_t gpeBuffer, size_t bytes, uint32_t elementBytes,
                                        std::string label) {
    if (impl_->compute == nullptr) {
        return Error(ErrorCode::Unsupported, "no gpe device to view");
    }
    const uint64_t object = impl_->compute->backendBuffer(gpeBuffer);
    if (object == 0) {
        return Error(ErrorCode::InvalidArgument, "not a live gpe buffer");
    }
    rhi::NativeHandle handle;
    handle.type = bufferHandleType(impl_->device->backend());
    handle.value = object;
    gpu::BufferDesc desc;
    desc.bytes = bytes;
    desc.elementBytes = elementBytes;
    desc.label = std::move(label);
    return gpu::Buffer::wrap(*impl_->device, handle, desc);
}

Result<uint64_t> Context::computeView(const gpu::Buffer& buffer) {
    if (impl_->compute == nullptr) {
        return Error(ErrorCode::Unsupported, "no gpe device to adopt into");
    }
    const rhi::NativeHandle handle = buffer.native();
    if (handle.value == 0) {
        return Error(ErrorCode::InvalidArgument, "the buffer has no native handle");
    }
    const uint64_t adopted = impl_->compute->adopt(handle.value, buffer.bytes());
    if (adopted == 0) {
        return Error(ErrorCode::DeviceFailure, "gpe refused the buffer");
    }
    return adopted;
}

void Context::flushCompute() {
    if (impl_->compute != nullptr) {
        impl_->compute->sync();
    }
}

Context* installProcessContext(const ContextDesc& desc) {
    static std::unique_ptr<Context> process = [&desc]() -> std::unique_ptr<Context> {
        auto made = Context::create(desc);
        if (!made) {
            log::warn("no GPU context: {}", made.error().toString());
            return nullptr;
        }
        (*made)->install();
        return std::move(*made);
    }();
    return process.get();
}

}   // namespace lrt::gpu_host
