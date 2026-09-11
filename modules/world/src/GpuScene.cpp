// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/GpuScene.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::world {
namespace {

struct MeshRecord {
    uint32_t firstPoint, points, firstTriangle, triangles, hasNormals, nodeBase, slotBase, pad2;
    float    lo[4];
    float    hi[4];
};

struct InstanceRecord {
    float    view[12];
    float    normal[12];
    float    world[12];
    float    colour[4];
    uint32_t mesh, primId, instanceId, flags;
};
static_assert(sizeof(MeshRecord) == 64);
static_assert(sizeof(InstanceRecord) == 176);

/// shaders/lrt/world/instancing.slang's SetRecord.
struct SetRecord {
    float    prototype[12];
    float    colour[4];
    uint32_t first;
    uint32_t count;
    uint32_t mesh;
    uint32_t primId;
    uint32_t flags;
    uint32_t pad[3];
};
static_assert(sizeof(SetRecord) == 96);

struct PrimvarRecord {
    uint32_t interpolation, components, first, count;
};

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                                 const void* data = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    // Positions and indices also feed acceleration structures.
    if (device.caps().accelerationStructure) {
        desc.extraUsage = rhi::BufferUsage::AccelerationStructureBuildInput;
    }
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
    auto records = gpu::ComputeKernel::create(library, "lrt/world/instancing", "instanceRecords");
    if (!records) return std::move(records).error();
    scene.records_ = std::move(*records);
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
    uint64_t primvarValues = 0;
    uint64_t primvarCount = 0;
    for (const auto& mesh : meshes_) {
        for (const geom::GpuPrimvar& p : mesh->primvars) {
            primvarValues += p.count;
            ++primvarCount;
        }
    }
    LRT_TRY(make(primvarValues_, primvarValues, 16, "scene.primvars"));
    LRT_TRY(make(indices_, triangles * 3, 4, "scene.indices"));
    LRT_TRY(make(triangleCorners_, triangles * 3, 4, "scene.triangleCorners"));
    LRT_TRY(make(triangleFaces_, triangles, 4, "scene.triangleFaces"));
    std::vector<MeshRecord> records(meshes_.size());
    std::vector<PrimvarRecord> primvars;
    firstPrimvar_.assign(meshes_.size(), 0);
    uint32_t nodes = 0;
    uint64_t valueAt = 0;
    gpu::CommandBatch batch(device);
    for (size_t k = 0; k < meshes_.size(); ++k) {
        const geom::GpuMesh& m = *meshes_[k];
        const Range& r = ranges_[k];
        rhi::ICommandEncoder* e = batch.encoder();
        e->copyBuffer(positions_.rhi(), uint64_t{r.firstPoint} * 16, m.positions.rhi(), 0, uint64_t{m.points} * 16);

        if (m.triangles > 0) {
            e->copyBuffer(indices_.rhi(), uint64_t{r.firstTriangle} * 12, m.indices.rhi(), 0,
                          uint64_t{m.triangles} * 12);
            e->copyBuffer(triangleCorners_.rhi(), uint64_t{r.firstTriangle} * 12, m.triangleCorners.rhi(), 0,
                          uint64_t{m.triangles} * 12);
            e->copyBuffer(triangleFaces_.rhi(), uint64_t{r.firstTriangle} * 4, m.triangleFaces.rhi(), 0,
                          uint64_t{m.triangles} * 4);
        }
        firstPrimvar_[k] = static_cast<uint32_t>(primvars.size());
        for (const geom::GpuPrimvar& p : m.primvars) {
            e->copyBuffer(primvarValues_.rhi(), valueAt * 16, p.values.rhi(), 0, uint64_t{p.count} * 16);
            primvars.push_back({static_cast<uint32_t>(p.interpolation), p.components, static_cast<uint32_t>(valueAt),
                                p.count});
            valueAt += p.count;
        }
        const uint32_t slotBase = static_cast<uint32_t>(k * slotNames_.size());
        records[k] = {r.firstPoint, m.points, r.firstTriangle, m.triangles, m.primvar("normals") ? 1u : 0u, nodes,
                      slotBase, 0,
                      {m.bounds.min[0], m.bounds.min[1], m.bounds.min[2], 0.0F},
                      {m.bounds.max[0], m.bounds.max[1], m.bounds.max[2], 0.0F}};
        nodes += std::max<uint32_t>(m.triangles, 1) - 1;   // a mesh's LBVH has triangles - 1 internal nodes
    }
    batch.markDirty();
    LRT_TRY(batch.submit(true));
    meshRecordWords_.assign(records.size() * sizeof(MeshRecord) / 4, 0);
    if (!records.empty()) {
        std::memcpy(meshRecordWords_.data(), records.data(), records.size() * sizeof(MeshRecord));
    }
    auto made = deviceBuffer(device, records.size(), sizeof(MeshRecord), "scene.meshes", records.data());
    if (!made) return std::move(made).error();
    meshRecords_ = std::move(*made);
    auto madePrimvars = deviceBuffer(device, primvars.size(), sizeof(PrimvarRecord), "scene.primvarRecords",
                                     primvars.data());
    if (!madePrimvars) return std::move(madePrimvars).error();
    primvarRecords_ = std::move(*madePrimvars);
    (void)primvarCount;
    slotsDirty_ = true;
    ++generation_;
    return ok();
}

void GpuScene::setExtraPrimvarSlots(std::vector<std::string> names) {
    std::vector<std::string> all{"displayColor", "displayOpacity", "normals", "st"};
    all.insert(all.end(), names.begin(), names.end());
    if (all != slotNames_) {
        slotNames_ = std::move(all);
        slotsDirty_ = true;
    }
}

uint32_t GpuScene::slotOf(std::string_view name) const noexcept {
    for (uint32_t k = 0; k < slotNames_.size(); ++k) {
        if (slotNames_[k] == name) {
            return k;
        }
    }
    return ~uint32_t{0};
}

Result<void> GpuScene::writeSlots() {
    // Per mesh, per slot: the record of the primvar with that name, or none.
    std::vector<uint32_t> slots(std::max<size_t>(meshes_.size() * slotNames_.size(), 1), ~uint32_t{0});
    std::vector<MeshRecord> records(meshes_.size());
    for (size_t k = 0; k < meshes_.size(); ++k) {
        const auto& primvars = meshes_[k]->primvars;
        for (size_t p = 0; p < primvars.size(); ++p) {
            for (size_t slot = 0; slot < slotNames_.size(); ++slot) {
                if (primvars[p].name == slotNames_[slot]) {
                    slots[k * slotNames_.size() + slot] = firstPrimvar_[k] + static_cast<uint32_t>(p);
                }
            }
        }
    }
    auto made = gpu::Buffer::fromSpan<uint32_t>(*device_, slots, "scene.primvarSlots");
    if (!made) return std::move(made).error();
    primvarSlots_ = std::move(*made);
    // Mesh records carry each mesh's row: rewritten for the slot count.
    constexpr size_t kWords = sizeof(MeshRecord) / 4;
    constexpr size_t kSlotBaseWord = offsetof(MeshRecord, slotBase) / 4;
    for (size_t k = 0; k < meshes_.size(); ++k) {
        meshRecordWords_[k * kWords + kSlotBaseWord] = static_cast<uint32_t>(k * slotNames_.size());
    }
    if (!meshes_.empty()) {
        LRT_TRY(meshRecords_.write(*device_, 0, meshRecordWords_.size() * 4, meshRecordWords_.data()));
    }
    slotsDirty_ = false;
    return ok();
}

Result<void> GpuScene::update(std::span<const MeshInstance> instances, const render::Projection& projection,
                              std::span<const InstanceSet> sets) {
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
    for (const InstanceSet& set : sets) {
        if (set.mesh == nullptr || set.mesh->triangles == 0 || set.count == 0) {
            continue;
        }
        if (indexOf.try_emplace(set.mesh.get(), static_cast<uint32_t>(meshes.size())).second) {
            meshes.push_back(set.mesh);
            byMesh.emplace_back();
        }
    }
    if (meshes != meshes_) {
        meshes_ = std::move(meshes);
        LRT_TRY(repack());
    }
    if (slotsDirty_) {
        LRT_TRY(writeSlots());
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
    // Sets after the single instances, their records written on the device.
    uint64_t total = records.size();
    for (const InstanceSet& set : sets) {
        if (set.mesh != nullptr && set.mesh->triangles > 0) {
            total += set.count;
        }
    }
    if (total > UINT32_MAX) {
        return Error(ErrorCode::OutOfMemory, "more than 2^32 instances");
    }
    instanceCount_ = static_cast<uint32_t>(total);
    if (instanceRecords_.count() < std::max<uint64_t>(total, 1)) {
        auto made = deviceBuffer(*device_, std::max<uint64_t>(total, 1) * 3 / 2 + 1, sizeof(InstanceRecord),
                                 "scene.instances");
        if (!made) return std::move(made).error();
        instanceRecords_ = std::move(*made);
    }
    if (!records.empty()) {
        LRT_TRY(instanceRecords_.write(*device_, 0, records.size() * sizeof(InstanceRecord), records.data()));
    }
    // Every set's records in one dispatch. Their chains are pooled, copied
    // on the device when the sets' chains change; per frame only the sets'
    // own records (a prim's matrix and look each) go up.
    const uint32_t singles = static_cast<uint32_t>(records.size());
    std::vector<SetRecord> setRecords;
    std::vector<std::pair<gpu::Buffer, uint32_t>> layout;
    uint32_t setInstances = 0;
    for (const InstanceSet& set : sets) {
        if (set.mesh == nullptr || set.mesh->triangles == 0 || set.count == 0) {
            continue;
        }
        const uint32_t mesh = indexOf.at(set.mesh.get());
        SetRecord record{};
        rows(set.prototype, record.prototype);
        record.colour[0] = set.displayColor[0];
        record.colour[1] = set.displayColor[1];
        record.colour[2] = set.displayColor[2];
        record.colour[3] = set.displayOpacity;
        record.first = setInstances;
        record.count = set.count;
        record.mesh = mesh;
        record.primId = set.primId;
        record.flags = set.doubleSided ? 1u : 0u;
        setRecords.push_back(record);
        layout.emplace_back(set.chainRows, set.count);
        draws_.push_back({mesh, singles + setInstances, set.count});
        setInstances += set.count;
    }
    if (setRecords.empty()) {
        return ok();
    }
    gpu::CommandBatch batch(*device_);
    const bool samePool = std::equal(layout.begin(), layout.end(), setLayout_.begin(), setLayout_.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.first.rhi() == b.first.rhi() && a.second == b.second;
                                     });
    if (!samePool) {
        if (setRows_.count() < uint64_t{setInstances} * 3) {
            auto made = deviceBuffer(*device_, uint64_t{setInstances} * 3 * 3 / 2, 16, "scene.setRows");
            if (!made) return std::move(made).error();
            setRows_ = std::move(*made);
        }
        for (size_t k = 0; k < setRecords.size(); ++k) {
            batch.encoder()->copyBuffer(setRows_.rhi(), uint64_t{setRecords[k].first} * 48, layout[k].first.rhi(), 0,
                                        uint64_t{layout[k].second} * 48);
        }
        batch.markDirty();
        setLayout_ = std::move(layout);
    }
    if (setRecords_.count() < setRecords.size()) {
        auto made = deviceBuffer(*device_, setRecords.size() * 3 / 2 + 1, sizeof(SetRecord), "scene.sets");
        if (!made) return std::move(made).error();
        setRecords_ = std::move(*made);
    }
    LRT_TRY(setRecords_.write(*device_, 0, setRecords.size() * sizeof(SetRecord), setRecords.data()));
    const std::array<float, 12> view = projection.worldToView.rows3x4();
    records_.dispatch(batch, {setInstances, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["sets"].setBinding(setRecords_.rhi());
        cursor["setRows"].setBinding(setRows_.rhi());
        cursor["records"].setBinding(instanceRecords_.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["count"].setData(setInstances);
        p["sets"].setData(static_cast<uint32_t>(setRecords.size()));
        p["first"].setData(singles);
        static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                  "v12", "v13", "v20", "v21", "v22", "v23"};
        for (size_t k = 0; k < 12; ++k) {
            p[kNames[k]].setData(view[k]);
        }
    });
    LRT_TRY(batch.submit(true));
    return ok();
}

}   // namespace lrt::world
