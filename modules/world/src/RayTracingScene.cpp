// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/RayTracingScene.h"

#include <algorithm>
#include <cstring>   // slang-rhi's acceleration-structure-utils.h calls memcpy without it

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
                                                                const char* label, uint64_t* updateScratch = nullptr) {
    rhi::AccelerationStructureSizes sizes;
    if (SLANG_FAILED(device.rhi()->getAccelerationStructureSizes(build, &sizes))) {
        return Error::make(ErrorCode::DeviceFailure, "no sizes for acceleration structure '{}'", label);
    }
    if (updateScratch != nullptr) {
        *updateScratch = std::max(*updateScratch, sizes.updateScratchSize);
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

Result<void> RayTracingScene::build(const GpuScene& scene, bool refit) {
    gpu::Device& device = *device_;
    if (scene.generation() != generation_) {
        bottom_.clear();
        bottomInputs_.clear();
        meshRevisions_.clear();
        uint64_t updateScratch = 0;
        std::vector<uint32_t> handles;
        // One structure a mesh, or one a bucket for a mesh that deforms; the
        // handle table has an entry a bucket either way, a still mesh's all
        // the same.
        const uint32_t buckets = scene.buckets();
        for (uint32_t m = 0; m < scene.meshCount(); ++m) {
            const geom::GpuMesh& mesh = scene.mesh(m);
            const uint32_t copies = scene.meshDeforms(m) ? buckets : 1;
            for (uint32_t b = 0; b < copies; ++b) {
            rhi::AccelerationStructureBuildInput input = {};
            input.type = rhi::AccelerationStructureBuildInputType::Triangles;
            input.triangles.vertexBuffers[0] = rhi::BufferOffsetPair(
                scene.positions().rhi(), (uint64_t{b} * scene.pointsStride() + scene.firstPoint(m)) * 16);
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
            // Updatable: a deformed mesh refits this structure in place.
            build.flags = rhi::AccelerationStructureBuildFlags::PreferFastTrace |
                          rhi::AccelerationStructureBuildFlags::AllowUpdate;
            auto made = buildStructure(device, build, rhi::AccelerationStructureKind::BottomLevel, "scene.blas",
                                       &updateScratch);
            if (!made) return std::move(made).error();
            const uint64_t handle = (*made)->getHandle().value;
            for (uint32_t h = 0; h < (copies == 1 ? buckets : 1u); ++h) {
                handles.push_back(static_cast<uint32_t>(handle));
                handles.push_back(static_cast<uint32_t>(handle >> 32));
            }
            bottom_.push_back(std::move(*made));
            bottomInputs_.push_back(input);
            bottomMesh_.push_back(m);
            meshRevisions_.push_back(scene.meshRevision(m));
            }
        }
        auto scratch = deviceBuffer(device, updateScratch, 1, "scene.as.updateScratch");
        if (!scratch) return std::move(scratch).error();
        updateScratch_ = std::move(*scratch);
        auto madeHandles = gpu::Buffer::fromSpan<uint32_t>(device, handles.empty() ? std::vector<uint32_t>{0, 0} : handles,
                                                           "scene.as.handles");
        if (!madeHandles) return std::move(madeHandles).error();
        handles_ = std::move(*madeHandles);
        generation_ = scene.generation();
    } else if (refit) {
        // The pools stand; a mesh deformed in place refits its structure over
        // the same triangles -- one submit each, since they share the scratch.
        for (uint32_t s = 0; s < bottom_.size(); ++s) {
            const uint32_t m = bottomMesh_[s];
            if (scene.meshRevision(m) == meshRevisions_[s]) {
                continue;
            }
            rhi::AccelerationStructureBuildDesc build;
            build.inputs = &bottomInputs_[s];
            build.inputCount = 1;
            build.mode = rhi::AccelerationStructureBuildMode::Update;
            build.flags = rhi::AccelerationStructureBuildFlags::PreferFastTrace |
                          rhi::AccelerationStructureBuildFlags::AllowUpdate;
            gpu::CommandBatch batch(device);
            batch.encoder()->buildAccelerationStructure(build, bottom_[s], bottom_[s],
                                                        rhi::BufferOffsetPair(updateScratch_.rhi(), 0), 0, nullptr);
            batch.markDirty();
            LRT_TRY(batch.submit(true));
            meshRevisions_[s] = scene.meshRevision(m);
        }
    }

    const uint32_t count = scene.tlasCount();
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
            cursor["params"]["base"].setData(scene.tlasFirst());
            cursor["params"]["buckets"].setData(scene.buckets());
            cursor["params"]["layout"].setData(layout);
            cursor["params"]["words"].setData(static_cast<uint32_t>(stride / 4));
            // Counter-clockwise is the front, as USD's right-handed meshes
            // (after the triangulation's left-handed reversal) have it -- the
            // APIs' default winding once a ray looks at a triangle -- judged in
            // object space, so mirrored instances keep their front.
            cursor["params"]["flags"].setData(
                static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::ForceOpaque));
            cursor["params"]["doubleSided"].setData(
                static_cast<uint32_t>(rhi::AccelerationStructureInstanceFlags::TriangleFacingCullDisable));
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
