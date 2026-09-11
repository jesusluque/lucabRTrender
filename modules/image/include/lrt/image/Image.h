// Copyright (c) 2026 lucabRTrender contributors.
//
// A picture in memory: float32 RGBA, linear, premultiplied -- the one format
// gpe and aofx know -- in bounds with a bottom-left origin.
//
// Ported from openFXplayer's image::Image, narrowed to what a renderer and an
// aofx host need: one depth, planes, attached numbers, and pluggable storage
// so the pixels can live where a kernel reads them. What is left out (bit
// depths, render scale, audio) stays openFXplayer's until the engine is
// wired into it.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lrt/core/Result.h"

namespace lrt::image {

/// Half-open pixel bounds, y up.
struct PixelRect {
    int32_t x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    [[nodiscard]] constexpr int32_t width() const noexcept { return x2 - x1; }
    [[nodiscard]] constexpr int32_t height() const noexcept { return y2 - y1; }
    [[nodiscard]] constexpr bool isEmpty() const noexcept { return x2 <= x1 || y2 <= y1; }
    [[nodiscard]] constexpr bool contains(int32_t x, int32_t y) const noexcept {
        return x >= x1 && x < x2 && y >= y1 && y < y2;
    }
    friend constexpr bool operator==(const PixelRect&, const PixelRect&) = default;
};

/// Where pixels come from: the heap by default, device-visible memory when a
/// host installs one. See openFXplayer's image::Storage, which this mirrors.
class Storage {
public:
    virtual ~Storage() = default;
    [[nodiscard]] virtual std::span<std::byte> allocate(size_t bytes, uint64_t& token) = 0;
    virtual void release(uint64_t token) noexcept = 0;
    /// Does an allocation have a device copy distinct from the host one?
    [[nodiscard]] virtual bool mirrors() const noexcept { return false; }
    virtual void toHost(uint64_t /*token*/) const noexcept {}
    virtual void toDevice(uint64_t /*token*/) const noexcept {}
    [[nodiscard]] virtual bool clearOnDevice(uint64_t /*token*/) const noexcept { return false; }
};

void setStorage(std::shared_ptr<Storage>);
[[nodiscard]] const std::shared_ptr<Storage>& storage() noexcept;

inline constexpr size_t kBytesPerPixel = 16;

class Image {
public:
    [[nodiscard]] static Result<std::shared_ptr<Image>> create(const PixelRect& bounds);

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    ~Image();

    [[nodiscard]] const PixelRect& bounds() const noexcept { return bounds_; }
    /// Pixels per row in memory. Rows are padded to a 64-byte boundary, so this
    /// can exceed the width -- kernels index `y * stride + x`.
    [[nodiscard]] int32_t stride() const noexcept { return stride_; }
    [[nodiscard]] size_t sizeBytes() const noexcept { return pixels_.size(); }

    /// Host pixels, made current. The non-const overload also records that the
    /// device copy is behind (see openFXplayer's Image for the whole rule).
    [[nodiscard]] std::span<std::byte> data() noexcept {
        if (mirrored_) { pullToHost(); willWriteHost(); }
        return pixels_;
    }
    [[nodiscard]] std::span<const std::byte> data() const noexcept {
        if (mirrored_) { pullToHost(); }
        return pixels_;
    }
    [[nodiscard]] std::span<float> floats() noexcept {
        auto bytes = data();
        return {reinterpret_cast<float*>(bytes.data()), bytes.size() / sizeof(float)};
    }
    [[nodiscard]] std::span<const float> floats() const noexcept {
        auto bytes = data();
        return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
    }

    /// Which allocation this is, without making either copy current.
    [[nodiscard]] const void* address() const noexcept { return pixels_.data(); }

    [[nodiscard]] bool mirrored() const noexcept { return mirrored_; }
    /// Before a kernel reads this image's buffer.
    void syncDevice() const noexcept { if (mirrored_) { pushToDevice(); } }
    /// After a kernel wrote it.
    void deviceWrote() noexcept;

    void clear() noexcept;

    void attachPlane(std::string id, std::shared_ptr<Image> plane);
    [[nodiscard]] std::shared_ptr<Image> plane(std::string_view id) const;
    void attach(std::string id, std::vector<float> values);
    [[nodiscard]] const std::vector<float>* attached(std::string_view id) const;
    [[nodiscard]] const std::vector<std::pair<std::string, std::vector<float>>>&
    attachments() const noexcept { return attached_; }

private:
    Image() = default;
    void pullToHost() const noexcept;
    void pushToDevice() const noexcept;
    void willWriteHost() noexcept;

    PixelRect                    bounds_;
    int32_t                      stride_ = 0;
    std::unique_ptr<std::byte[]> owned_;
    std::shared_ptr<Storage>     from_;
    uint64_t                     token_ = 0;
    std::span<std::byte>         pixels_;
    bool                         mirrored_ = false;
    mutable std::atomic<bool>    hostStale_{false};
    mutable std::atomic<bool>    deviceStale_{false};
    std::vector<std::pair<std::string, std::shared_ptr<Image>>> planes_;
    std::vector<std::pair<std::string, std::vector<float>>>      attached_;
};

using ImagePtr = std::shared_ptr<Image>;

}   // namespace lrt::image
