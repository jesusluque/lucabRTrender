// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/GpuScene.h"

#include <algorithm>
#include <map>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::world {
namespace {

struct MeshRecord {
    uint32_t firstPoint, points, firstTriangle, triangles, hasNormals, pad0, pad1, pad2;
};

struct InstanceRecord {
    float    view[12];
    float    normal[12];
    float    world[12];
    float    colour[4];
    uint32_t mesh, primId, instanceId, flags;
};
static_assert(sizeof(MeshRecord) == 32);
static_assert(sizeof(InstanceRecord) == 176);

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                                 const void* data = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, count > 0 ? data : nullptr);
}

void rows(const render::Mat4& m, float* into) {
    const std::array<float, 12> r = m.rows3x4();
    std::copy(r.begin(), r.end(), into);
}

}   // namespace

Result<GpuScene> GpuScene::create(gpu::ShaderLibrary& library) {
    GpuScene scene;
    scene.device_ = &library.device();
    return scene;
}

Result<void> GpuScene::repack() {
    gpu::Device& device = *device_;
    ranges_.assign(meshes_.size(), {});
    uint64_t points = 0;
    uint64_t triangles = 0;
    for (size_t k = 0; k < meshes_.size(); ++k) {
        ranges_[k].firstPoint = static_cast<uint32_t>(points);
        ranges_[k].firstTriangle = static_cast<uint32_t>(triangles);
        points += meshes_[k]->points;
        triangles += meshes_[k]->triangles;
    }
    if (points > UINT32_MAX || triangles * 3 > UINT32_MAX) {
        return Error(ErrorCode::OutOfMemory, "the scene's meshes exceed 32-bit pool offsets");
    }
    const auto make = [&](gpu::Buffer& into, uint64_t count, uint32_t element, const char* label) -> Result<void> {
        auto made = deviceBuffer(device, count, element, label);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    LRT_TRY(make(positions_, points, 16, "scene.positions"));
    LRT_TRY(make(normals_, points, 16, "scene.normals"));
    LRT_TRY(make(indices_, triangles * 3, 4, "scene.indices"));
    LRT_TRY(make(triangleCorners_, triangles * 3, 4, "scene.triangleCorners"));
    LRT_TRY(make(triangleFaces_, triangles, 4, "scene.triangleFaces"));
    std::vector<MeshRecord> records(meshes_.size());
    gpu::CommandBatch batch(device);
    for (size_t k = 0; k < meshes_.size(); ++k) {
        const geom::GpuMesh& m = *meshes_[k];
        const Range& r = ranges_[k];
        rhi::ICommandEncoder* e = batch.encoder();
        e->copyBuffer(positions_.rhi(), uint64_t{r.firstPoint} * 16, m.positions.rhi(), 0, uint64_t{m.points} * 16);
        if (m.normals.valid()) {
            e->copyBuffer(normals_.rhi(), uint64_t{r.firstPoint} * 16, m.normals.rhi(), 0, uint64_t{m.points} * 16);
        }
        if (m.triangles > 0) {
            e->copyBuffer(indices_.rhi(), uint64_t{r.firstTriangle} * 12, m.indices.rhi(), 0,
                          uint64_t{m.triangles} * 12);
            e->copyBuffer(triangleCorners_.rhi(), uint64_t{r.firstTriangle} * 12, m.triangleCorners.rhi(), 0,
                          uint64_t{m.triangles} * 12);
            e->copyBuffer(triangleFaces_.rhi(), uint64_t{r.firstTriangle} * 4, m.triangleFaces.rhi(), 0,
                          uint64_t{m.triangles} * 4);
        }
        records[k] = {r.firstPoint, m.points, r.firstTriangle, m.triangles, m.normals.valid() ? 1u : 0u, 0, 0, 0};
    }
    batch.markDirty();
    LRT_TRY(batch.submit(true));
    auto made = deviceBuffer(device, records.size(), sizeof(MeshRecord), "scene.meshes", records.data());
    if (!made) return std::move(made).error();
    meshRecords_ = std::move(*made);
    return ok();
}

Result<void> GpuScene::update(std::span<const MeshInstance> instances, const render::Projection& projection) {
    // The meshes, in first-appearance order; instances grouped by mesh.
    std::vector<std::shared_ptr<const geom::GpuMesh>> meshes;
    std::map<const geom::GpuMesh*, uint32_t> indexOf;
    std::vector<std::vector<uint32_t>> byMesh;
    for (uint32_t i = 0; i < instances.size(); ++i) {
        const MeshInstance& instance = instances[i];
        if (instance.mesh == nullptr || instance.mesh->triangles == 0) {
            continue;
        }
        auto [it, inserted] = indexOf.try_emplace(instance.mesh.get(), static_cast<uint32_t>(meshes.size()));
        if (inserted) {
            meshes.push_back(instance.mesh);
            byMesh.emplace_back();
        }
        byMesh[it->second].push_back(i);
    }
    if (meshes != meshes_) {
        meshes_ = std::move(meshes);
        LRT_TRY(repack());
    }

    std::vector<InstanceRecord> records;
    records.reserve(instances.size());
    draws_.clear();
    for (uint32_t m = 0; m < meshes_.size(); ++m) {
        DrawRange draw{m, static_cast<uint32_t>(records.size()), 0};
        for (const uint32_t i : byMesh[m]) {
            const MeshInstance& instance = instances[i];
            // Per instance, in double: object to view, and its inverse
            // transpose for normals. A prim's matrix, not the prim's data.
            const render::Mat4 view = projection.worldToView * instance.objectToWorld;
            const render::Mat4 inverse = aofx::xform::inverseAffine(view);
            render::Mat4 normal = render::Mat4::identity();
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    normal.at(r, c) = inverse.at(c, r);
                }
            }
            InstanceRecord record{};
            rows(view, record.view);
            rows(normal, record.normal);
            rows(instance.objectToWorld, record.world);
            record.colour[0] = instance.displayColor[0];
            record.colour[1] = instance.displayColor[1];
            record.colour[2] = instance.displayColor[2];
            record.colour[3] = instance.displayOpacity;
            record.mesh = m;
            record.primId = instance.primId;
            record.instanceId = instance.instanceId;
            record.flags = instance.doubleSided ? 1u : 0u;
            records.push_back(record);
            ++draw.instances;
        }
        draws_.push_back(draw);
    }
    instanceCount_ = static_cast<uint32_t>(records.size());
    if (instanceRecords_.count() < std::max<size_t>(records.size(), 1)) {
        auto made = deviceBuffer(*device_, std::max<size_t>(records.size(), 1) * 3 / 2 + 1, sizeof(InstanceRecord),
                                 "scene.instances");
        if (!made) return std::move(made).error();
        instanceRecords_ = std::move(*made);
    }
    if (!records.empty()) {
        LRT_TRY(instanceRecords_.write(*device_, 0, records.size() * sizeof(InstanceRecord), records.data()));
    }
    return ok();
}

}   // namespace lrt::world
