// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/RayTracingScene.h"

#include <algorithm>

#include <slang-rhi/acceleration-structure-utils.h>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::world {
namespace {

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t bytes, uint32_t element, const char* label,
                                 rhi::BufferUsage extra = rhi::BufferUsage::None) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(bytes, element);
    desc.elementBytes = element;
    desc.label = label;
    desc.extraUsage = extra;
    return gpu::Buffer::create(device, desc);
}

Result<rhi::ComPtr<rhi::IAccelerationStructure>> buildStructure(gpu::Device& device,
                                                                const rhi::AccelerationStructureBuildDesc& build,
                                                                rhi::AccelerationStructureKind kind,
                                                                const char* label) {
    rhi::AccelerationStructureSizes sizes;
    if (SLANG_FAILED(device.rhi()->getAccelerationStructureSizes(build, &sizes))) {
        return Error::make(ErrorCode::DeviceFailure, "no sizes for acceleration structure '{}'", label);
    }
    auto scratch = deviceBuffer(device, sizes.scratchSize, 1, "scene.as.scratch");
    if (!scratch) return std::move(scratch).error();
    rhi::AccelerationStructureDesc desc;
    desc.kind = kind;
    desc.size = sizes.accelerationStructureSize;
    desc.label = label;
    rhi::ComPtr<rhi::IAccelerationStructure> structure;
    if (SLANG_FAILED(device.rhi()->createAccelerationStructure(desc, structure.writeRef()))) {
        return Error::make(ErrorCode::OutOfMemory, "cannot allocate acceleration structure '{}'", label);
    }
    gpu::CommandBatch batch(device);
    batch.encoder()->buildAccelerationStructure(build, structure, nullptr, rhi::BufferOffsetPair(scratch->rhi(), 0),
                                                0, nullptr);
    batch.markDirty();
    LRT_TRY(batch.submit(true));   // the scratch buffer goes when this returns
    return structure;
}

}   // namespace

Result<RayTracingScene> RayTracingScene::create(gpu::ShaderLibrary& library) {
    if (!library.device().caps().accelerationStructure || !library.device().caps().rayQuery) {
        return Error(ErrorCode::Unsupported, "no hardware acceleration structures on this device");
    }
    auto descs = gpu::ComputeKernel::create(library, "lrt/world/instance_descs", "instanceDescs");
    if (!descs) return std::move(descs).error();
    RayTracingScene rt;
    rt.device_ = &library.device();
    rt.descs_ = std::move(*descs);
    return rt;
}

Result<void> RayTracingScene::build(const GpuScene& scene) {
    gpu::Device& device = *device_;
    if (scene.generation() != generation_) {
        bottom_.clear();
        std::vector<uint32_t> handles;
        for (uint32_t m = 0; m < scene.meshCount(); ++m) {
            const geom::GpuMesh& mesh = scene.mesh(m);
            rhi::AccelerationStructureBuildInput input = {};
            input.type = rhi::AccelerationStructureBuildInputType::Triangles;
            input.triangles.vertexBuffers[0] =
                rhi::BufferOffsetPair(scene.positions().rhi(), uint64_t{scene.firstPoint(m)} * 16);
            input.triangles.vertexBufferCount = 1;
            input.triangles.vertexFormat = rhi::Format::RGB32Float;
            input.triangles.vertexCount = mesh.points;
            input.triangles.vertexStride = 16;
            input.triangles.indexBuffer =
                rhi::BufferOffsetPair(scene.indices().rhi(), uint64_t{scene.firstTriangle(m)} * 12);
            input.triangles.indexFormat = rhi::IndexFormat::Uint32;
            input.triangles.indexCount = mesh.triangles * 3;
            input.triangles.flags = rhi::AccelerationStructureGeometryFlags::Opaque;
            rhi::AccelerationStructureBuildDesc build;
            build.inputs = &input;
            build.inputCount = 1;
            build.flags = rhi::AccelerationStructureBuildFlags::PreferFastTrace;
            auto made = buildStructure(device, build, rhi::AccelerationStructureKind::BottomLevel, "scene.blas");
            if (!made) return std::move(made).error();
            const uint64_t handle = (*made)->getHandle().value;
            handles.push_back(static_cast<uint32_t>(handle));
            handles.push_back(static_cast<uint32_t>(handle >> 32));
            bottom_.push_back(std::move(*made));
        }
        auto madeHandles = gpu::Buffer::fromSpan<uint32_t>(device, handles.empty() ? std::vector<uint32_t>{0, 0} : handles,
                                                           "scene.as.handles");
        if (!madeHandles) return std::move(madeHandles).error();
        handles_ = std::move(*madeHandles);
        generation_ = scene.generation();
    }

    const uint32_t count = scene.instanceCount();
    const auto type = rhi::getAccelerationStructureInstanceDescType(device.rhi());
    const size_t stride = rhi::getAccelerationStructureInstanceDescSize(type);
    const uint32_t layout = type == rhi::AccelerationStructureInstanceDescType::Metal   ? 2u
                            : type == rhi::AccelerationStructureInstanceDescType::Optix ? 1u
                                                                                        : 0u;
    if (instanceDescs_.bytes() < std::max<uint64_t>(count, 1) * stride) {
        auto made = deviceBuffer(device, std::max<uint64_t>(count, 1) * stride * 3 / 2, 4, "scene.as.instances",
                                 rhi::BufferUsage::AccelerationStructureBuildInput);
        if (!made) return std::move(made).error();
        instanceDescs_ = std::move(*made);
    }
    if (count > 0) {
        gpu::CommandBatch batch(device);
        descs_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["instances"].setBinding(scene.instanceRecords().rhi());
            cursor["handles"].setBinding(handles_.rhi());
            cursor["descs"].setBinding(instanceDescs_.rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["layout"].setData(layout);
            cursor["params"]["words"].setData(static_cast<uint32_t>(stride / 4));
            cursor["params"]["flags"].setData(static_cast<uint32_t>(
                rhi::AccelerationStructureInstanceFlags::ForceOpaque |
                rhi::AccelerationStructureInstanceFlags::TriangleFacingCullDisable));
        });
        LRT_TRY(batch.submit(true));
    }
    rhi::AccelerationStructureBuildInput input = {};
    input.type = rhi::AccelerationStructureBuildInputType::Instances;
    input.instances.instanceBuffer = rhi::BufferOffsetPair(instanceDescs_.rhi(), 0);
    input.instances.instanceStride = static_cast<uint32_t>(stride);
    input.instances.instanceCount = count;
    rhi::AccelerationStructureBuildDesc build;
    build.inputs = &input;
    build.inputCount = 1;
    build.flags = rhi::AccelerationStructureBuildFlags::PreferFastBuild;
    auto top = buildStructure(device, build, rhi::AccelerationStructureKind::TopLevel, "scene.tlas");
    if (!top) return std::move(top).error();
    topLevel_ = std::move(*top);
    return ok();
}

}   // namespace lrt::world
