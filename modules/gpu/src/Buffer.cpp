// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/Buffer.h"

#include <cstring>

#include "lrt/gpu/Device.h"

namespace lrt::gpu {
namespace {

rhi::BufferDesc toRhi(const BufferDesc& desc) {
    rhi::BufferDesc out;
    out.size = desc.bytes;
    out.elementSize = desc.elementBytes;
    out.label = desc.label.empty() ? nullptr : desc.label.c_str();
    switch (desc.memory) {
    case Memory::Device:
        out.memoryType = rhi::MemoryType::DeviceLocal;
        out.usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::UnorderedAccess |
                    rhi::BufferUsage::CopySource | rhi::BufferUsage::CopyDestination;
        out.defaultState = rhi::ResourceState::UnorderedAccess;
        break;
    case Memory::Upload:
        out.memoryType = rhi::MemoryType::Upload;
        out.usage = rhi::BufferUsage::CopySource | rhi::BufferUsage::ShaderResource;
        out.defaultState = rhi::ResourceState::General;
        break;
    case Memory::Readback:
        out.memoryType = rhi::MemoryType::ReadBack;
        out.usage = rhi::BufferUsage::CopyDestination;
        out.defaultState = rhi::ResourceState::CopyDestination;
        break;
    }
    out.usage = out.usage | desc.extraUsage;
    return out;
}

}   // namespace

Result<Buffer> Buffer::create(Device& device, const BufferDesc& desc, const void* initial) {
    if (desc.bytes == 0) {
        return Error::make(ErrorCode::InvalidArgument, "buffer '{}' of zero bytes", desc.label);
    }
    Buffer buffer;
    const rhi::BufferDesc rhiDesc = toRhi(desc);
    if (SLANG_FAILED(device.rhi()->createBuffer(rhiDesc, initial, buffer.buffer_.writeRef()))) {
        return Error::make(ErrorCode::OutOfMemory, "cannot allocate buffer '{}' ({} bytes)",
                           desc.label, desc.bytes);
    }
    buffer.bytes_ = desc.bytes;
    buffer.elementBytes_ = desc.elementBytes;
    return buffer;
}

Result<Buffer> Buffer::wrap(Device& device, rhi::NativeHandle handle, const BufferDesc& desc) {
    Buffer buffer;
    const rhi::BufferDesc rhiDesc = toRhi(desc);
    if (SLANG_FAILED(device.rhi()->createBufferFromNativeHandle(handle, rhiDesc,
                                                                buffer.buffer_.writeRef()))) {
        return Error::make(ErrorCode::DeviceFailure,
                           "cannot wrap native buffer for '{}' ({} bytes)", desc.label,
                           desc.bytes);
    }
    buffer.bytes_ = desc.bytes;
    buffer.elementBytes_ = desc.elementBytes;
    return buffer;
}

rhi::NativeHandle Buffer::native() const {
    rhi::NativeHandle handle;
    if (buffer_ != nullptr) {
        (void)buffer_->getNativeHandle(&handle);
    }
    return handle;
}

Result<void> Buffer::read(Device& device, uint64_t offset, uint64_t bytes, void* into) const {
    if (buffer_ == nullptr || offset + bytes > bytes_) {
        return Error(ErrorCode::InvalidArgument, "read outside the buffer");
    }
    device.beforeSubmit();
    if (SLANG_FAILED(device.rhi()->readBuffer(buffer_.get(), offset, bytes, into))) {
        return Error(ErrorCode::DeviceFailure, "readBuffer failed");
    }
    return ok();
}

Result<void> Buffer::write(Device& device, uint64_t offset, uint64_t bytes, const void* from) {
    if (buffer_ == nullptr || offset + bytes > bytes_) {
        return Error(ErrorCode::InvalidArgument, "write outside the buffer");
    }
    rhi::ComPtr<rhi::ICommandEncoder> encoder = device.queue()->createCommandEncoder();
    encoder->uploadBufferData(buffer_.get(), offset, bytes, const_cast<void*>(from));
    device.beforeSubmit();
    if (SLANG_FAILED(device.queue()->submit(encoder->finish()))) {
        return Error(ErrorCode::DeviceFailure, "upload submit failed");
    }
    return ok();
}

}   // namespace lrt::gpu
