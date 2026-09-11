// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/image/Image.h"

#include <cstring>
#include <new>

namespace lrt::image {
namespace {

constexpr size_t kRowAlignment = 64;
constexpr size_t kMaxImageBytes = size_t{16} << 30;

std::shared_ptr<Storage>& slot() {
    static std::shared_ptr<Storage> kStorage;
    return kStorage;
}

}   // namespace

void setStorage(std::shared_ptr<Storage> storage) { slot() = std::move(storage); }
const std::shared_ptr<Storage>& storage() noexcept { return slot(); }

Result<std::shared_ptr<Image>> Image::create(const PixelRect& bounds) {
    if (bounds.isEmpty()) {
        return Error(ErrorCode::InvalidArgument, "an image needs non-empty bounds");
    }
    const size_t width = static_cast<size_t>(bounds.width());
    const size_t rowBytes =
        (width * kBytesPerPixel + kRowAlignment - 1) / kRowAlignment * kRowAlignment;
    const size_t height = static_cast<size_t>(bounds.height());
    if (height > kMaxImageBytes / rowBytes) {
        return Error::make(ErrorCode::OutOfMemory, "a {}x{} image is too large",
                           bounds.width(), bounds.height());
    }
    const size_t bytes = rowBytes * height;

    auto image = std::shared_ptr<Image>(new Image());
    image->bounds_ = bounds;
    image->stride_ = static_cast<int32_t>(rowBytes / kBytesPerPixel);
    if (const std::shared_ptr<Storage>& from = slot(); from != nullptr) {
        uint64_t token = 0;
        if (std::span<std::byte> external = from->allocate(bytes, token);
            external.size() >= bytes) {
            image->from_ = from;
            image->token_ = token;
            image->pixels_ = external.first(bytes);
            image->mirrored_ = from->mirrors();
            image->deviceStale_.store(image->mirrored_);
            return image;
        }
    }
    image->owned_.reset(new (std::nothrow) std::byte[bytes]);
    if (image->owned_ == nullptr) {
        return Error::make(ErrorCode::OutOfMemory, "cannot allocate {} bytes", bytes);
    }
    image->pixels_ = {image->owned_.get(), bytes};
    return image;
}

Image::~Image() {
    if (from_ != nullptr) {
        from_->release(token_);
    }
}

void Image::pullToHost() const noexcept {
    if (hostStale_.exchange(false)) {
        from_->toHost(token_);
    }
}

void Image::pushToDevice() const noexcept {
    if (deviceStale_.exchange(false)) {
        from_->toDevice(token_);
    }
}

void Image::willWriteHost() noexcept { deviceStale_.store(true); }

void Image::deviceWrote() noexcept {
    if (mirrored_) {
        hostStale_.store(true);
        deviceStale_.store(false);
    }
}

void Image::clear() noexcept {
    if (mirrored_ && from_->clearOnDevice(token_)) {
        deviceWrote();
        return;
    }
    std::span<std::byte> bytes = data();
    std::memset(bytes.data(), 0, bytes.size());
}

void Image::attachPlane(std::string id, std::shared_ptr<Image> plane) {
    if (plane == nullptr || id.empty()) {
        return;
    }
    for (auto& [existing, held] : planes_) {
        if (existing == id) {
            held = std::move(plane);
            return;
        }
    }
    planes_.emplace_back(std::move(id), std::move(plane));
}

std::shared_ptr<Image> Image::plane(std::string_view id) const {
    for (const auto& [name, held] : planes_) {
        if (name == id) {
            return held;
        }
    }
    return nullptr;
}

void Image::attach(std::string id, std::vector<float> values) {
    if (id.empty()) {
        return;
    }
    for (auto& [existing, held] : attached_) {
        if (existing == id) {
            held = std::move(values);
            return;
        }
    }
    attached_.emplace_back(std::move(id), std::move(values));
}

const std::vector<float>* Image::attached(std::string_view id) const {
    for (const auto& [name, held] : attached_) {
        if (name == id) {
            return &held;
        }
    }
    return nullptr;
}

}   // namespace lrt::image
