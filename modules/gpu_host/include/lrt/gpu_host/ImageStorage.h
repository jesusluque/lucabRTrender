// Copyright (c) 2026 lucabRTrender contributors.
//
// Images that live where a kernel can already read them.
//
// Ported from openFXplayer's gpu_host::ImageStorage, whose comments carry the
// measurements behind every choice here (4K upload 6.81 ms -> 1.39 ms shared;
// pinning a plate costs ~20 ms, hence the spares). Two modes, one question --
// where can a kernel find this picture:
//
//   Shared  unified memory: one allocation both sides address.
//   Paired  a discrete card: pinned host memory and a device buffer, kept in
//           step only when one side actually reads what the other wrote.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "lrt/image/Image.h"

namespace gpe {
class PooledDevice;
}

namespace lrt::gpu_host {

class ImageStorage final : public image::Storage {
public:
    enum class Mode { Shared, Paired };

    ImageStorage(std::shared_ptr<gpe::PooledDevice> device, size_t budgetBytes, Mode mode);
    ~ImageStorage() override;

    [[nodiscard]] std::span<std::byte> allocate(size_t bytes, uint64_t& token) override;
    void release(uint64_t token) noexcept override;
    [[nodiscard]] bool mirrors() const noexcept override { return mode_ == Mode::Paired; }
    void toHost(uint64_t token) const noexcept override;
    void toDevice(uint64_t token) const noexcept override;
    [[nodiscard]] bool clearOnDevice(uint64_t token) const noexcept override;

    /// The gpe buffer an image's pixels live in, or kInvalidBuffer for heap
    /// pixels (the caller then uploads).
    [[nodiscard]] uint64_t bufferFor(const void* pixels) const;

    [[nodiscard]] uint64_t declined() const noexcept { return declined_; }
    [[nodiscard]] size_t bytesHeld() const noexcept { return held_; }

private:
    struct Block {
        void*    memory = nullptr;
        uint64_t buffer = 0;
        size_t   bytes = 0;
        bool     pinned = false;
    };

    [[nodiscard]] Block blockFor(uint64_t token) const noexcept;
    [[nodiscard]] void* takeHost(size_t bytes, bool& pinned);
    void giveHost(void* memory, size_t bytes, bool pinned);

    std::shared_ptr<gpe::PooledDevice>   device_;
    Mode                                 mode_ = Mode::Shared;
    size_t                               budget_ = 0;
    size_t                               held_ = 0;
    uint64_t                             nextToken_ = 1;
    uint64_t                             declined_ = 0;
    std::map<size_t, std::vector<void*>> spare_;
    size_t                               spareBytes_ = 0;
    mutable std::mutex                   mutex_;
    std::map<uint64_t, Block>            blocks_;
    std::map<const void*, uint64_t>      byAddress_;
};

}   // namespace lrt::gpu_host
