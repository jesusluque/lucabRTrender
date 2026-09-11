// Copyright (c) 2026 lucabRTrender contributors.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "lrt/core/Result.h"

namespace lrt::gpu {

class Device;

enum class Memory {
    Device,     ///< GPU-only. The default for everything a kernel writes.
    Upload,     ///< CPU writes, GPU reads.
    Readback,   ///< GPU writes (by copy), CPU reads.
};

struct BufferDesc {
    uint64_t    bytes = 0;
    /// Bytes per element for a structured buffer. Zero for raw bytes.
    uint32_t    elementBytes = 0;
    Memory      memory = Memory::Device;
    /// Add vertex/index/indirect/AS-input usage on top of storage + copy.
    rhi::BufferUsage extraUsage = rhi::BufferUsage::None;
    std::string label;
};

/// A device buffer. Owns its slang-rhi object; cheap to copy (ref-counted).
class Buffer {
public:
    Buffer() = default;

    [[nodiscard]] static Result<Buffer> create(Device& device, const BufferDesc& desc,
                                               const void* initial = nullptr);

    /// Wraps a buffer somebody else allocated on the same device -- a gpe
    /// buffer, found by its MTLBuffer* or CUDA pointer. The owner keeps it.
    [[nodiscard]] static Result<Buffer> wrap(Device& device, rhi::NativeHandle handle,
                                             const BufferDesc& desc);

    template <typename T>
    [[nodiscard]] static Result<Buffer> fromSpan(Device& device, std::span<const T> data,
                                                 std::string label,
                                                 Memory memory = Memory::Device) {
        BufferDesc desc;
        desc.bytes = data.size_bytes();
        desc.elementBytes = sizeof(T);
        desc.memory = memory;
        desc.label = std::move(label);
        return create(device, desc, data.data());
    }

    [[nodiscard]] bool valid() const noexcept { return buffer_ != nullptr; }
    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] uint32_t elementBytes() const noexcept { return elementBytes_; }
    [[nodiscard]] uint64_t count() const noexcept {
        return elementBytes_ != 0 ? bytes_ / elementBytes_ : bytes_;
    }
    [[nodiscard]] rhi::IBuffer* rhi() const noexcept { return buffer_.get(); }
    [[nodiscard]] rhi::NativeHandle native() const;

    /// Synchronous readback, for the CLI's final image and for tests' metric
    /// counters. Never on a frame path: see gpu::AsyncReadback.
    [[nodiscard]] Result<void> read(Device& device, uint64_t offset, uint64_t bytes,
                                    void* into) const;

    template <typename T>
    [[nodiscard]] Result<std::vector<T>> readAll(Device& device) const {
        std::vector<T> out(bytes_ / sizeof(T));
        LRT_TRY(read(device, 0, out.size() * sizeof(T), out.data()));
        return out;
    }

    /// Synchronous upload into a device buffer.
    [[nodiscard]] Result<void> write(Device& device, uint64_t offset, uint64_t bytes,
                                     const void* from);

private:
    rhi::ComPtr<rhi::IBuffer> buffer_;
    uint64_t                  bytes_ = 0;
    uint32_t                  elementBytes_ = 0;
};

}   // namespace lrt::gpu
