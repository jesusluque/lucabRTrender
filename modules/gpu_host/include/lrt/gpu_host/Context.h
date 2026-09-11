// Copyright (c) 2026 lucabRTrender contributors.
//
// The one GPU, two runtimes on it, and the one thread that talks to it.
//
// WHAT IS SHARED, AND HOW
//
// slang-rhi opens the device (lrt::gpu::Device). gpe does not open another: it
// adopts that one -- the same MTLDevice and command queue on Apple, the same
// CUDA context and stream on Linux -- through gpe::adoptMetalDevice /
// adoptCudaContext. So:
//
//   - a gpe buffer is an MTLBuffer / CUDA allocation slang-rhi can wrap
//     (`renderView`), and a slang-rhi buffer is one gpe can adopt
//     (`computeView`): no copies in either direction;
//   - ordering is the queue's (Metal commits in order on one queue) or the
//     stream's (CUDA). gpe holds dispatches back in batches on Metal, so the
//     render device hands gpe's batch over before every submission of its
//     own (gpu::Device::setBeforeSubmit): the calls run in the order they
//     were made. gpe's uploads ride its own transfer queue/stream and are
//     ordered against gpe's work only, so `flushCompute()` is what a pass
//     calls before slang-rhi reads bytes gpe *uploaded*.
//
// On Vulkan there is no gpe backend: `compute()` is null and aofx is not
// available. That is Linux without an NVIDIA card.
//
// ONE THREAD
//
// openFXplayer's rule, for openFXplayer's reason: gpe's pool rewrites handles
// in place and is confined to one thread by contract, and slang-rhi command
// encoding is not re-entrant either. Every piece of GPU work runs through
// `run()`, on this context's thread.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/Device.h"

namespace gpe {
class PooledDevice;
}

namespace lrt::gpu_host {

class ImageStorage;

struct ContextDesc {
    /// Pool ceiling for unified memory; a discrete card takes 3/4 of what is
    /// free when that is more (openFXplayer's measured rule).
    size_t          budgetBytes = size_t{2} << 30;
    gpu::DeviceDesc device;
};

class Context {
public:
    [[nodiscard]] static Result<std::unique_ptr<Context>> create(const ContextDesc& desc = {});
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /// Makes this the process's context and its storage the image storage.
    void install();
    [[nodiscard]] static Context* current() noexcept;

    enum class Priority { Batch, Interactive };

    /// Runs `work` on the GPU thread and waits. Re-entrant from that thread.
    [[nodiscard]] Result<void> run(const std::function<void()>& work,
                                   Priority priority = Priority::Batch);

    [[nodiscard]] gpu::Device& device() noexcept;
    [[nodiscard]] std::shared_ptr<gpu::Device> deviceShared() const noexcept;

    /// gpe on the same device, or null (Vulkan).
    [[nodiscard]] gpe::PooledDevice* compute() noexcept;
    [[nodiscard]] std::shared_ptr<gpe::PooledDevice> computeShared() const noexcept;
    [[nodiscard]] ImageStorage* sharedStorage() noexcept;

    /// A slang-rhi view of a gpe buffer: the same memory, bound by name.
    [[nodiscard]] Result<gpu::Buffer> renderView(uint64_t gpeBuffer, size_t bytes,
                                                 uint32_t elementBytes,
                                                 std::string label = {});
    /// A gpe handle on a slang-rhi buffer. Release it before the buffer goes.
    [[nodiscard]] Result<uint64_t> computeView(const gpu::Buffer& buffer);

    /// Waits for gpe's queued work and transfers. Call before slang-rhi reads
    /// what gpe uploaded; unnecessary after a gpe dispatch, which shares the
    /// queue/stream and is handed over before the render device submits.
    void flushCompute();

    [[nodiscard]] const std::string& backendName() const noexcept;

private:
    Context();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Makes the process context once, installs it and returns it (or null).
Context* installProcessContext(const ContextDesc& desc = {});

}   // namespace lrt::gpu_host
