// Copyright (c) 2026 lucabRTrender contributors.
//
// Ported from openFXplayer src/gpu_host/src/ImageStorage.cpp; see that file
// for the measurements behind the spares and the Paired mode.
#include "lrt/gpu_host/ImageStorage.h"

#include <cstdlib>

#include "gpe/pool.h"
#include "gpe/types.h"

namespace lrt::gpu_host {

ImageStorage::ImageStorage(std::shared_ptr<gpe::PooledDevice> device, size_t budgetBytes,
                           Mode mode)
    : device_(std::move(device)), mode_(mode), budget_(budgetBytes) {}

void* ImageStorage::takeHost(size_t bytes, bool& pinned) {
    if (auto found = spare_.find(bytes); found != spare_.end() && !found->second.empty()) {
        void* memory = found->second.back();
        found->second.pop_back();
        spareBytes_ -= bytes;
        pinned = true;
        return memory;
    }
    void* memory = device_->allocHost(bytes);
    pinned = memory != nullptr;
    return memory != nullptr ? memory : std::malloc(bytes);
}

void ImageStorage::giveHost(void* memory, size_t bytes, bool pinned) {
    if (memory == nullptr) {
        return;
    }
    if (!pinned) {
        std::free(memory);
        return;
    }
    if (spareBytes_ + bytes <= budget_) {
        spare_[bytes].push_back(memory);
        spareBytes_ += bytes;
        return;
    }
    device_->freeHost(memory);
}

ImageStorage::~ImageStorage() {
    const std::lock_guard<std::mutex> held(mutex_);
    for (auto& [bytes, blocks] : spare_) {
        for (void* memory : blocks) {
            device_->freeHost(memory);
        }
    }
    for (const auto& [token, block] : blocks_) {
        device_->release(static_cast<gpe::BufferId>(block.buffer));
        if (mode_ == Mode::Paired) {
            if (block.pinned) {
                device_->freeHost(block.memory);
            } else {
                std::free(block.memory);
            }
        }
    }
}

std::span<std::byte> ImageStorage::allocate(size_t bytes, uint64_t& token) {
    if (bytes == 0) {
        return {};
    }
    const std::lock_guard<std::mutex> held(mutex_);
    if (held_ + bytes > budget_) {
        ++declined_;
        return {};
    }
    gpe::BufferId buffer = gpe::kInvalidBuffer;
    void*         memory = nullptr;
    bool          pinned = false;
    if (mode_ == Mode::Shared) {
        memory = device_->allocShared(bytes, buffer);
    } else {
        buffer = device_->alloc(bytes);
        if (buffer != gpe::kInvalidBuffer) {
            memory = takeHost(bytes, pinned);
            if (memory == nullptr) {
                device_->release(buffer);
                buffer = gpe::kInvalidBuffer;
            }
        }
    }
    if (memory == nullptr || buffer == gpe::kInvalidBuffer) {
        ++declined_;
        return {};
    }
    token = nextToken_++;
    blocks_[token] = Block{memory, static_cast<uint64_t>(buffer), bytes, pinned};
    byAddress_[memory] = static_cast<uint64_t>(buffer);
    held_ += bytes;
    return {static_cast<std::byte*>(memory), bytes};
}

void ImageStorage::release(uint64_t token) noexcept {
    const std::lock_guard<std::mutex> held(mutex_);
    const auto found = blocks_.find(token);
    if (found == blocks_.end()) {
        return;
    }
    const Block block = found->second;
    byAddress_.erase(block.memory);
    held_ -= block.bytes;
    device_->release(static_cast<gpe::BufferId>(block.buffer));
    if (mode_ == Mode::Paired) {
        giveHost(block.memory, block.bytes, block.pinned);
    }
    blocks_.erase(found);
}

uint64_t ImageStorage::bufferFor(const void* pixels) const {
    const std::lock_guard<std::mutex> held(mutex_);
    const auto found = byAddress_.find(pixels);
    return found == byAddress_.end() ? gpe::kInvalidBuffer : found->second;
}

ImageStorage::Block ImageStorage::blockFor(uint64_t token) const noexcept {
    const std::lock_guard<std::mutex> held(mutex_);
    const auto found = blocks_.find(token);
    return found == blocks_.end() ? Block{} : found->second;
}

void ImageStorage::toHost(uint64_t token) const noexcept {
    if (const Block block = blockFor(token); block.memory != nullptr) {
        device_->download(block.memory, static_cast<gpe::BufferId>(block.buffer), block.bytes);
    }
}

void ImageStorage::toDevice(uint64_t token) const noexcept {
    if (const Block block = blockFor(token); block.memory != nullptr) {
        device_->upload(static_cast<gpe::BufferId>(block.buffer), block.memory, block.bytes);
    }
}

bool ImageStorage::clearOnDevice(uint64_t token) const noexcept {
    const Block block = blockFor(token);
    return block.memory != nullptr &&
           device_->fill(static_cast<gpe::BufferId>(block.buffer), 0, block.bytes);
}

}   // namespace lrt::gpu_host
