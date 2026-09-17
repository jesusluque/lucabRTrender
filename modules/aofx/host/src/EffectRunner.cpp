// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/aofx/EffectRunner.h"

#include <mutex>
#include <unordered_map>

#include "gpe/args.h"
#include "gpe/kernels.h"
#include "gpe/pool.h"
#include "lrt/core/Log.h"
#include "lrt/core/Platform.h"
#include "lrt/gpu_host/Context.h"

namespace lrt::aofx_host {
namespace {

std::mutex& complaintGuard() {
    static std::mutex kGuard;
    return kGuard;
}
std::string& complaintText() {
    static std::string kText;
    return kText;
}

/// `keep`: process-wide, because the point is to outlive a render (openFXplayer
/// EffectRunner.cpp, "What keep is holding").
struct Kept {
    std::mutex                                     mutex;
    std::unordered_map<std::string, gpe::BufferId> buffers;
    std::unordered_map<std::string, size_t>        sizes;
    /// The pages each borrowed key was bound over, so a second borrow can
    /// tell the same mapping from a new one.
    std::unordered_map<std::string, const void*>   borrowed;
};
Kept& kept() {
    static Kept kStore;
    return kStore;
}

std::mutex& publishedGuard() {
    static std::mutex kGuard;
    return kGuard;
}
std::map<std::string, std::map<std::string, double>>& publishedMap() {
    static std::map<std::string, std::map<std::string, double>> kPublished;
    return kPublished;
}

aofx::Buffer rowBuffer(gpe::BufferId buffer, size_t bytes) {
    aofx::Buffer out;
    out.device = static_cast<uint64_t>(buffer);
    const size_t count = bytes / gpe::kBytesPerPixel;
    out.width = static_cast<int>(count);
    out.height = 1;
    out.stride = static_cast<int>(count);
    out.rect = aofx::Rect{0, 0, out.width, 1};
    return out;
}

constexpr const char* kNoMedia =
    "clips and decoding are openFXplayer's media stack, which this host does not carry";
constexpr const char* kNoModels =
    "models are openFXplayer's inference stack, which this host does not carry";

}   // namespace

struct EffectRunner::Impl {
    gpu_host::Context*                    context = nullptr;
    gpe::PooledDevice*                    device = nullptr;
    std::map<std::string, aofx::KernelId> loaded;
    std::map<aofx::KernelId, std::string> names;
    std::vector<gpe::BufferId>            scratch;
    uint64_t                              dispatches = 0;
};

EffectRunner::EffectRunner(gpu_host::Context& context) : impl_(std::make_unique<Impl>()) {
    impl_->context = &context;
    impl_->device = context.compute();
}

EffectRunner::~EffectRunner() {
    if (impl_->device != nullptr) {
        for (const gpe::BufferId buffer : impl_->scratch) {
            impl_->device->release(buffer);
        }
    }
}

std::string EffectRunner::lastComplaint() {
    const std::lock_guard<std::mutex> held(complaintGuard());
    return complaintText();
}

void EffectRunner::complain(std::string what) {
    const std::lock_guard<std::mutex> held(complaintGuard());
    complaintText() = std::move(what);
}

aofx::KernelId EffectRunner::load(const std::string& name) {
    if (const auto found = impl_->loaded.find(name); found != impl_->loaded.end()) {
        return found->second;
    }
    if (impl_->device == nullptr) {
        complain("no gpe device");
        return aofx::kInvalidKernel;
    }
    const gpe::KernelId kernel = impl_->device->load(name);
    if (kernel == gpe::kInvalidKernel) {
        log::error("no AOFX kernel '{}'; the plugin's build and its source disagree", name);
        complain("no kernel '" + name + "'; the plugin declares one its build does not have");
    }
    const auto id = static_cast<aofx::KernelId>(kernel);
    impl_->loaded.emplace(name, id);
    impl_->names.emplace(id, name);
    return id;
}

bool EffectRunner::run(aofx::KernelId kernel, aofx::Grid grid,
                       const std::vector<aofx::Buffer>& buffers, const void* uniforms,
                       size_t uniformBytes) {
    if (kernel == aofx::kInvalidKernel || buffers.empty() || impl_->device == nullptr) {
        return false;
    }
    // Checked against the kernel's own reflection when its blob carries it:
    // the run that would have rendered the wrong buffers (openFXplayer D30) is
    // a refusal with a reason instead.
    const auto name = impl_->names.find(kernel);
    std::optional<gpe::KernelInfo> info;
    if (name != impl_->names.end()) {
        info = gpe::kernelInfo(name->second);
    }
    if (info.has_value()) {
        if (info->elementBytes.size() != buffers.size()) {
            complain("kernel '" + name->second + "' declares " +
                     std::to_string(info->elementBytes.size()) + " buffers and was run with " +
                     std::to_string(buffers.size()));
            return false;
        }
        if (info->uniformBytes != uniformBytes) {
            complain("kernel '" + name->second + "' declares " +
                     std::to_string(info->uniformBytes) + " bytes of uniforms and was run with " +
                     std::to_string(uniformBytes));
            return false;
        }
    }
    gpe::Args args;
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i].device == gpe::kInvalidBuffer) {
            return false;
        }
        // The kernel's element size when known; one byte otherwise, which is
        // openFXplayer's workaround for CUDA's bounds check on a T the host
        // cannot see (and which gpe now overrides from the trailer anyway).
        const uint32_t element = info.has_value() && info->elementBytes[i] > 0
                                     ? info->elementBytes[i]
                                     : 1u;
        args.buffer(static_cast<gpe::BufferId>(buffers[i].device), element);
    }
    if (uniforms != nullptr && uniformBytes > 0) {
        args.uniformBytes(uniforms, uniformBytes);
    }
    impl_->device->dispatch(static_cast<gpe::KernelId>(kernel),
                            gpe::Grid{grid.x, grid.y, grid.z}, args.data(), args.size());
    ++impl_->dispatches;
    return true;
}

aofx::Buffer EffectRunner::scratch(int width, int height) {
    if (width <= 0 || height <= 0 || impl_->device == nullptr) {
        return {};
    }
    const size_t bytes =
        static_cast<size_t>(width) * static_cast<size_t>(height) * gpe::kBytesPerPixel;
    const gpe::BufferId buffer = impl_->device->alloc(bytes);
    if (buffer == gpe::kInvalidBuffer) {
        return {};
    }
    impl_->scratch.push_back(buffer);
    aofx::Buffer out;
    out.device = static_cast<uint64_t>(buffer);
    out.width = width;
    out.height = height;
    out.stride = width;
    out.rect = aofx::Rect{0, 0, width, height};
    return out;
}

aofx::Buffer EffectRunner::keep(const std::string& key, const void* data, size_t bytes) {
    if (key.empty() || bytes % gpe::kBytesPerPixel != 0 || impl_->device == nullptr) {
        return {};
    }
    Kept& store = kept();
    const std::lock_guard<std::mutex> holding(store.mutex);
    if (const auto found = store.buffers.find(key); found != store.buffers.end()) {
        return rowBuffer(found->second, store.sizes[key]);
    }
    if (data == nullptr || bytes == 0) {
        return {};
    }
    const gpe::BufferId buffer = impl_->device->alloc(bytes);
    if (buffer == gpe::kInvalidBuffer) {
        return {};
    }
    impl_->device->upload(buffer, data, bytes);
    store.buffers.emplace(key, buffer);
    store.sizes.emplace(key, bytes);
    return rowBuffer(buffer, bytes);
}

aofx::Buffer EffectRunner::borrow(const std::string& key, const void* pages, size_t bytes) {
    if (key.empty() || impl_->device == nullptr) {
        return {};
    }
    Kept& store = kept();
    const std::lock_guard<std::mutex> holding(store.mutex);
    if (const auto found = store.buffers.find(key); found != store.buffers.end()) {
        const auto bound = store.borrowed.find(key);
        const bool same = pages == nullptr ||
                          (bound != store.borrowed.end() && bound->second == pages && store.sizes[key] == bytes);
        if (same) {
            return rowBuffer(found->second, store.sizes[key]);
        }
        // The key names other pages now: the old binding goes rather than
        // being handed back over pages that may already be unmapped.
        impl_->device->release(found->second);
        store.borrowed.erase(key);
        store.sizes.erase(key);
        store.buffers.erase(found);
    }
    if (pages == nullptr || bytes == 0 || bytes % gpe::kBytesPerPixel != 0) {
        return {};
    }
    void* wrapped = platform::newMetalBufferOverPages(reinterpret_cast<void*>(impl_->device->backendDevice()),
                                                      pages, bytes);
    if (wrapped == nullptr) {
        return {};
    }
    // The pool adopts it like any foreign buffer, taking its own reference;
    // `drop` gives that back, and the pages stay the effect's.
    const gpe::BufferId buffer = impl_->device->adopt(reinterpret_cast<uint64_t>(wrapped), bytes);
    platform::releaseMetalBuffer(wrapped);
    if (buffer == gpe::kInvalidBuffer) {
        return {};
    }
    store.buffers.emplace(key, buffer);
    store.sizes.emplace(key, bytes);
    store.borrowed.insert_or_assign(key, pages);
    return rowBuffer(buffer, bytes);
}

aofx::Buffer EffectRunner::importFd(const std::string& /*key*/, int /*fd*/, size_t /*bytes*/) {
    return {};
}

void EffectRunner::drop(const std::string& key) {
    Kept& store = kept();
    const std::lock_guard<std::mutex> holding(store.mutex);
    if (const auto found = store.buffers.find(key); found != store.buffers.end()) {
        if (impl_->device != nullptr) {
            impl_->device->release(found->second);
        }
        store.sizes.erase(key);
        store.borrowed.erase(key);
        store.buffers.erase(found);
    }
}

bool EffectRunner::read(const aofx::Buffer& buffer, void* into, size_t bytes) {
    if (!buffer.isValid() || into == nullptr || bytes == 0 || impl_->device == nullptr) {
        return false;
    }
    impl_->device->download(into, static_cast<gpe::BufferId>(buffer.device), bytes);
    return true;
}

void EffectRunner::publish(const std::string& instance, const char* key, double value) {
    if (key == nullptr) {
        return;
    }
    const std::lock_guard<std::mutex> held(publishedGuard());
    publishedMap()[instance][key] = value;
}

std::map<std::string, std::map<std::string, double>> EffectRunner::publishedState() {
    const std::lock_guard<std::mutex> held(publishedGuard());
    return publishedMap();
}

uint64_t EffectRunner::dispatches() const noexcept { return impl_->dispatches; }

aofx::ClipId EffectRunner::clip(const std::string& path) {
    complain(std::string(kNoMedia) + " ('" + path + "')");
    return aofx::kInvalidClip;
}
aofx::ClipInfo EffectRunner::clipInfo(aofx::ClipId) const { return {}; }
bool EffectRunner::decode(aofx::ClipId, int, const aofx::Buffer&) {
    complain(kNoMedia);
    return false;
}
bool EffectRunner::decodeNext(aofx::ClipId, int, const aofx::Buffer&, bool& atStart,
                              bool& atEnd) {
    atStart = true;
    atEnd = true;
    complain(kNoMedia);
    return false;
}
aofx::ModelId EffectRunner::model(const std::string& name) {
    complain(std::string(kNoModels) + " ('" + name + "')");
    return aofx::kInvalidModel;
}
aofx::ModelIo EffectRunner::modelInput(aofx::ModelId, int) const { return {}; }
aofx::ModelIo EffectRunner::modelOutput(aofx::ModelId, int) const { return {}; }
bool EffectRunner::infer(aofx::ModelId, const std::vector<aofx::Buffer>&,
                         const std::vector<aofx::Buffer>&) {
    complain(kNoModels);
    return false;
}
bool EffectRunner::inferred(aofx::ModelId) { return false; }
bool EffectRunner::inferShaped(aofx::ModelId, const std::vector<aofx::Buffer>&,
                               const std::vector<aofx::ModelIo>&,
                               const std::vector<aofx::Buffer>&,
                               const std::vector<aofx::ModelIo>&) {
    complain(kNoModels);
    return false;
}

}   // namespace lrt::aofx_host
