// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/GpuScene.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>
#include <tuple>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::world {
namespace {

struct MeshRecord {
    uint32_t firstPoint, points, firstTriangle, triangles, hasNormals, nodeBase, slotBase, subsetBase;
    float    lo[4];
    float    hi[4];
};

struct InstanceRecord {
    float    view[12];
    float    normal[12];
    float    world[12];
    float    colour[4];
    uint32_t mesh, primId, instanceId, flags;
    uint32_t categoriesLo, categoriesHi, mask, pointsOffset;
};
static_assert(sizeof(MeshRecord) == 64);
static_assert(sizeof(InstanceRecord) == 192);

/// shaders/lrt/world/instancing.slang's SetRecord.
struct SetRecord {
    float    prototype[12];
    float    colour[4];
    uint32_t first;
    uint32_t count;
    uint32_t mesh;
    uint32_t primId;
    uint32_t flags;
    uint32_t categoriesLo, categoriesHi, pad0;
};
static_assert(sizeof(SetRecord) == 96);

struct PrimvarRecord {
    uint32_t interpolation, components, first, count;
};

struct MotionInput {
    float    worldLo[12];
    float    worldHi[12];
    float    colour[4];
    uint32_t mesh, primId, instanceId, flags;
    uint32_t categoriesLo, categoriesHi, deforms, pad0;
    float    time0, time1, pad1, pad2;
};
static_assert(sizeof(MotionInput) == 160);

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
    auto clear = gpu::ComputeKernel::create(library, "lrt/geom/mesh_subsets", "subsetClear");
    auto motionRecords = gpu::ComputeKernel::create(library, "lrt/world/motion", "motionRecords");
    if (!motionRecords) return std::move(motionRecords).error();
    auto positionsLerp = gpu::ComputeKernel::create(library, "lrt/world/motion", "positionsLerp");
    if (!positionsLerp) return std::move(positionsLerp).error();
    scene.motionRecords_ = std::move(*motionRecords);
    scene.positionsLerp_ = std::move(*positionsLerp);
    if (!clear) return std::move(clear).error();
    scene.subsetClear_ = std::move(*clear);
    for (auto [into, module, entry] : {std::tuple{&scene.worldBoxes_, "lrt/world/scene_bounds", "instanceWorldBoxes"},
                                       std::tuple{&scene.boundsChunks_, "lrt/scene/bounds_chunks", "boundsChunks"},
                                       std::tuple{&scene.boundsReduce_, "lrt/scene/bounds_reduce", "boundsReduce"}}) {
        auto made = gpu::ComputeKernel::create(library, module, entry);
        if (!made) return std::move(made).error();
        *into = std::move(*made);
    }
    return scene;
}

bool GpuScene::sameLayout(const geom::GpuMesh& a, const geom::GpuMesh& b) noexcept {
    if (a.topology == 0 || a.topology != b.topology || a.points != b.points || a.faces != b.faces ||
        a.corners != b.corners || a.triangles != b.triangles || a.subsets != b.subsets ||
        a.primvars.size() != b.primvars.size()) {
        return false;
    }
    for (size_t p = 0; p < a.primvars.size(); ++p) {
        const geom::GpuPrimvar& x = a.primvars[p];
        const geom::GpuPrimvar& y = b.primvars[p];
        if (x.name != y.name || x.interpolation != y.interpolation || x.components != y.components ||
            x.count != y.count) {
            return false;
        }
    }
    return true;
}

Result<void> GpuScene::refresh(uint32_t k, const geom::GpuMesh& m) {
    gpu::Device& device = *device_;
    const Range& r = ranges_[k];
    gpu::CommandBatch batch(device);
    rhi::ICommandEncoder* e = batch.encoder();
    e->copyBuffer(positions_.rhi(), uint64_t{r.firstPoint} * 16, m.positions.rhi(), 0, uint64_t{m.points} * 16);
    uint64_t valueAt = primvarValueBase_[k];
    for (const geom::GpuPrimvar& p : m.primvars) {
        e->copyBuffer(primvarValues_.rhi(), valueAt * 16, p.values.rhi(), 0, uint64_t{p.count} * 16);
        valueAt += p.count;
    }
    batch.markDirty();
    LRT_TRY(batch.submit(true));
    // The record's box is the mesh's new one; the rest of the record stands.
    constexpr size_t kWords = sizeof(MeshRecord) / 4;
    constexpr size_t kLoWord = offsetof(MeshRecord, lo) / 4;
    constexpr size_t kHiWord = offsetof(MeshRecord, hi) / 4;
    for (size_t c = 0; c < 3; ++c) {
        std::memcpy(&meshRecordWords_[k * kWords + kLoWord + c], &m.bounds.min[c], 4);
        std::memcpy(&meshRecordWords_[k * kWords + kHiWord + c], &m.bounds.max[c], 4);
    }
    return meshRecords_.write(device, k * sizeof(MeshRecord), sizeof(MeshRecord), &meshRecordWords_[k * kWords]);
}

Result<void> GpuScene::repack() {
    gpu::Device& device = *device_;
    ranges_.assign(meshes_.size(), {});
    meshRevisions_.assign(meshes_.size(), 0);
    primvarValueBase_.assign(meshes_.size(), 0);
    deforms_.resize(meshes_.size(), false);
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
    bool anyDeforms = false;
    for (const bool d : deforms_) {
        anyDeforms = anyDeforms || d;
    }
    pointsStride_ = static_cast<uint32_t>(points);
    LRT_TRY(make(positions_, points * (anyDeforms ? buckets_ : 1), 16, "scene.positions"));
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
    LRT_TRY(make(triangleSubsets_, triangles, 4, "scene.triangleSubsets"));
    anyHidden_ = false;
    subsetBases_.assign(meshes_.size(), 0);
    uint32_t subsetsSoFar = 0;
    for (size_t k = 0; k < meshes_.size(); ++k) {
        subsetBases_[k] = subsetsSoFar;
        subsetsSoFar += meshes_[k]->subsets;
    }
    subsetRowWords_.assign(std::max<uint32_t>(subsetsSoFar, 1), 0);
    {
        auto rows = gpu::Buffer::fromSpan<uint32_t>(device, subsetRowWords_, "scene.subsetRows");
        if (!rows) return std::move(rows).error();
        subsetRows_ = std::move(*rows);
    }
    std::vector<MeshRecord> records(meshes_.size());
    std::vector<PrimvarRecord> primvars;
    firstPrimvar_.assign(meshes_.size(), 0);
    uint32_t nodes = 0;
    uint64_t valueAt = 0;
    gpu::CommandBatch batch(device);
    subsetClear_.dispatch(batch, {static_cast<uint32_t>(std::max<uint64_t>(triangles, 1)), 1, 1},
                          [&](rhi::ShaderCursor cursor) {
                              cursor["cleared"].setBinding(triangleSubsets_.rhi());
                              cursor["params"]["count"].setData(static_cast<uint32_t>(triangles));
                              cursor["params"]["value"].setData(uint32_t{0});
                          });
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
            if ((m.subsets > 0 || m.hidden) && m.triangleSubsets.valid()) {
                e->copyBuffer(triangleSubsets_.rhi(), uint64_t{r.firstTriangle} * 4, m.triangleSubsets.rhi(), 0,
                              uint64_t{m.triangles} * 4);
            }
            anyHidden_ = anyHidden_ || m.hidden;
        }
        firstPrimvar_[k] = static_cast<uint32_t>(primvars.size());
        primvarValueBase_[k] = valueAt;
        for (const geom::GpuPrimvar& p : m.primvars) {
            e->copyBuffer(primvarValues_.rhi(), valueAt * 16, p.values.rhi(), 0, uint64_t{p.count} * 16);
            primvars.push_back({static_cast<uint32_t>(p.interpolation), p.components, static_cast<uint32_t>(valueAt),
                                p.count});
            valueAt += p.count;
        }
        const uint32_t slotBase = static_cast<uint32_t>(k * slotNames_.size());
        records[k] = {r.firstPoint, m.points, r.firstTriangle, m.triangles, m.primvar("normals") ? 1u : 0u, nodes,
                      slotBase, subsetBases_[k],
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
                              std::span<const InstanceSet> sets, uint32_t bucketsWanted, double shutterOpen,
                              double shutterClose) {
    // The meshes, in first-appearance order; instances grouped by mesh.
    std::vector<std::shared_ptr<const geom::GpuMesh>> meshes;
    std::map<const geom::GpuMesh*, uint32_t> indexOf;
    std::vector<std::vector<uint32_t>> byMesh;
    std::vector<bool> deforms;
    for (uint32_t i = 0; i < instances.size(); ++i) {
        const MeshInstance& instance = instances[i];
        if (instance.mesh == nullptr || instance.mesh->triangles == 0) {
            continue;
        }
        auto [it, inserted] = indexOf.try_emplace(instance.mesh.get(), static_cast<uint32_t>(meshes.size()));
        if (inserted) {
            meshes.push_back(instance.mesh);
            byMesh.emplace_back();
            deforms.push_back(false);
        }
        byMesh[it->second].push_back(i);
        if (bucketsWanted > 1 && instance.motion.has_value()) {
            const auto fits = [&](const std::shared_ptr<const geom::GpuMesh>& m) {
                return m != nullptr && m != instance.mesh && sameLayout(*instance.mesh, *m);
            };
            if (fits(instance.motion->meshStart) || fits(instance.motion->meshEnd)) {
                deforms[it->second] = true;
            }
        }
    }
    // Buckets only where something moves; the pools are laid out for them.
    bool moves = false;
    for (const MeshInstance& instance : instances) {
        moves = moves || instance.motion.has_value();
    }
    const uint32_t buckets = moves ? std::min<uint32_t>(std::max<uint32_t>(bucketsWanted, 1), 8) : 1;
    bool anyDeforms = false;
    for (const bool d : deforms) {
        anyDeforms = anyDeforms || d;
    }
    if (!anyDeforms) {
        deforms.assign(deforms.size(), false);
    }
    const bool relayout = buckets != buckets_ || deforms != deforms_;
    buckets_ = buckets;
    deforms_ = deforms;
    for (const InstanceSet& set : sets) {
        if (set.mesh == nullptr || set.mesh->triangles == 0 || set.count == 0) {
            continue;
        }
        if (indexOf.try_emplace(set.mesh.get(), static_cast<uint32_t>(meshes.size())).second) {
            meshes.push_back(set.mesh);
            byMesh.emplace_back();
            deforms_.push_back(false);
        }
    }
    if (relayout) {
        meshes_ = std::move(meshes);
        LRT_TRY(repack());
    } else if (meshes != meshes_) {
        // The same meshes deformed -- each slot the same mesh or one of its
        // topology key and layout -- keep the pools and take new positions
        // in place; anything else repacks.
        bool deformation = meshes.size() == meshes_.size();
        for (size_t k = 0; deformation && k < meshes.size(); ++k) {
            deformation = meshes[k] == meshes_[k] || sameLayout(*meshes[k], *meshes_[k]);
        }
        if (deformation) {
            for (size_t k = 0; k < meshes.size(); ++k) {
                if (meshes[k] != meshes_[k]) {
                    LRT_TRY(refresh(static_cast<uint32_t>(k), *meshes[k]));
                    meshes_[k] = meshes[k];
                    ++meshRevisions_[k];
                }
            }
            ++positionsRevision_;
        } else {
            meshes_ = std::move(meshes);
            LRT_TRY(repack());
        }
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
            // Bit 0: double sided; above bit 8, the material row.
            record.flags = (instance.doubleSided ? 1u : 0u) | (instance.material << 8);
            record.categoriesLo = static_cast<uint32_t>(instance.categories & 0xFFFFFFFFu);
            record.categoriesHi = static_cast<uint32_t>(instance.categories >> 32);
            record.mask = 0xFFu;
            record.pointsOffset = 0;
            records.push_back(record);
            ++draw.instances;
        }
        draws_.push_back(draw);
    }
    // Under motion, the same instances' inputs for the bucket copies, in the
    // records' order: both transforms, and whether the mesh has positions
    // per bucket.
    std::vector<MotionInput> motionInputs;
    if (buckets_ > 1) {
        motionInputs.reserve(records.size());
        for (uint32_t m = 0; m < meshes_.size() && m < byMesh.size(); ++m) {
            for (const uint32_t i : byMesh[m]) {
                const MeshInstance& instance = instances[i];
                const InstanceRecord& record = records[motionInputs.size()];
                MotionInput input{};
                rows(instance.motion.has_value() ? instance.motion->objectToWorldStart : instance.objectToWorld,
                     input.worldLo);
                rows(instance.motion.has_value() ? instance.motion->objectToWorldEnd : instance.objectToWorld,
                     input.worldHi);
                std::copy(std::begin(record.colour), std::end(record.colour), input.colour);
                input.mesh = record.mesh;
                input.primId = record.primId;
                input.instanceId = record.instanceId;
                input.flags = record.flags;
                input.categoriesLo = record.categoriesLo;
                input.categoriesHi = record.categoriesHi;
                input.deforms = deforms_[m] ? 1u : 0u;
                input.time0 = static_cast<float>(instance.motion.has_value() ? instance.motion->timeStart : 0.0);
                input.time1 = static_cast<float>(instance.motion.has_value() ? instance.motion->timeEnd : 1.0);
                motionInputs.push_back(input);
            }
        }
        // A deforming mesh's positions at each bucket, between its two meshes.
        for (uint32_t m = 0; m < meshes_.size() && m < byMesh.size(); ++m) {
            if (!deforms_[m] || byMesh[m].empty()) {
                continue;
            }
            const MeshInstance& instance = instances[byMesh[m].front()];
            const geom::GpuMesh& own = *meshes_[m];
            const auto& startMesh = instance.motion->meshStart;
            const auto& endMesh = instance.motion->meshEnd;
            const geom::GpuMesh& lo = startMesh != nullptr && sameLayout(own, *startMesh) ? *startMesh : own;
            const geom::GpuMesh& hi = endMesh != nullptr && sameLayout(own, *endMesh) ? *endMesh : own;
            gpu::CommandBatch batch(*device_);
            positionsLerp_.dispatch(batch, {lo.points * buckets_, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["pointsLo"].setBinding(lo.positions.rhi());
                cursor["pointsHi"].setBinding(hi.positions.rhi());
                cursor["positions"].setBinding(positions_.rhi());
                rhi::ShaderCursor p = cursor["lerpParams"];
                p["count"].setData(lo.points);
                p["firstPoint"].setData(ranges_[m].firstPoint);
                p["buckets"].setData(buckets_);
                p["pointsStride"].setData(pointsStride_);
                p["time0"].setData(static_cast<float>(instance.motion->timeStart));
                p["time1"].setData(static_cast<float>(instance.motion->timeEnd));
                p["open"].setData(static_cast<float>(shutterOpen));
                p["close"].setData(static_cast<float>(shutterClose));
            });
            LRT_TRY(batch.submit(true));
            ++meshRevisions_[m];   // its structures follow the new positions each frame
        }
        if (anyDeforms) {
            ++positionsRevision_;
        }
    }
    // Each mesh's subset rows, from the first instance or set of it that has them.
    {
        std::vector<uint32_t> rows(subsetRowWords_.size(), 0);
        const auto place = [&](const geom::GpuMesh* mesh, const std::vector<uint32_t>& subsetMaterials) {
            const auto found = indexOf.find(mesh);
            if (found == indexOf.end() || subsetMaterials.empty()) {
                return;
            }
            const uint32_t base = subsetBases_[found->second];
            const uint32_t count = std::min<uint32_t>(meshes_[found->second]->subsets,
                                                      static_cast<uint32_t>(subsetMaterials.size()));
            for (uint32_t k = 0; k < count; ++k) {
                rows[base + k] = subsetMaterials[k];
            }
        };
        for (const MeshInstance& instance : instances) {
            place(instance.mesh.get(), instance.subsetMaterials);
        }
        for (const InstanceSet& set : sets) {
            place(set.mesh.get(), set.subsetMaterials);
        }
        if (rows != subsetRowWords_) {
            LRT_TRY(subsetRows_.write(*device_, 0, rows.size() * 4, rows.data()));
            subsetRowWords_ = std::move(rows);
        }
    }
    // Sets after the single instances, their records written on the device;
    // and after those, under motion, every single instance's bucket copies.
    uint64_t total = records.size();
    for (const InstanceSet& set : sets) {
        if (set.mesh != nullptr && set.mesh->triangles > 0) {
            total += set.count;
        }
    }
    const uint64_t withMotion = total + (buckets_ > 1 ? uint64_t{records.size()} * buckets_ : 0);
    if (withMotion > UINT32_MAX) {
        return Error(ErrorCode::OutOfMemory, "more than 2^32 instances");
    }
    instanceCount_ = static_cast<uint32_t>(total);
    if (buckets_ > 1) {
        // The structure holds the sets (still, so answering to every bit) and
        // the bucket copies; not the frame's single records.
        tlasFirst_ = static_cast<uint32_t>(records.size());
        tlasCount_ = static_cast<uint32_t>(withMotion - records.size());
    } else {
        tlasFirst_ = 0;
        tlasCount_ = instanceCount_;
    }
    if (instanceRecords_.count() < std::max<uint64_t>(withMotion, 1)) {
        auto made = deviceBuffer(*device_, std::max<uint64_t>(withMotion, 1) * 3 / 2 + 1, sizeof(InstanceRecord),
                                 "scene.instances");
        if (!made) return std::move(made).error();
        instanceRecords_ = std::move(*made);
    }
    if (!records.empty()) {
        LRT_TRY(instanceRecords_.write(*device_, 0, records.size() * sizeof(InstanceRecord), records.data()));
    }
    if (buckets_ > 1 && !motionInputs.empty()) {
        if (motionInputs_.count() < motionInputs.size()) {
            auto made = deviceBuffer(*device_, motionInputs.size() * 3 / 2 + 1, sizeof(MotionInput), "scene.motion");
            if (!made) return std::move(made).error();
            motionInputs_ = std::move(*made);
        }
        LRT_TRY(motionInputs_.write(*device_, 0, motionInputs.size() * sizeof(MotionInput), motionInputs.data()));
        const std::array<float, 12> view = projection.worldToView.rows3x4();
        const uint32_t count = static_cast<uint32_t>(motionInputs.size());
        gpu::CommandBatch batch(*device_);
        motionRecords_.dispatch(batch, {count * buckets_, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["motionInputs"].setBinding(motionInputs_.rhi());
            cursor["records"].setBinding(instanceRecords_.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["buckets"].setData(buckets_);
            p["first"].setData(static_cast<uint32_t>(total));
            p["pointsStride"].setData(pointsStride_);
            p["open"].setData(static_cast<float>(shutterOpen));
            p["close"].setData(static_cast<float>(shutterClose));
            static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                      "v12", "v13", "v20", "v21", "v22", "v23"};
            for (size_t k = 0; k < 12; ++k) {
                p[kNames[k]].setData(view[k]);
            }
        });
        LRT_TRY(batch.submit(true));
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
        record.flags = (set.doubleSided ? 1u : 0u) | (set.material << 8);
        record.categoriesLo = static_cast<uint32_t>(set.categories & 0xFFFFFFFFu);
        record.categoriesHi = static_cast<uint32_t>(set.categories >> 32);
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

Result<std::optional<scene::Bounds>> GpuScene::worldBounds() const {
    const uint32_t n = instanceCount_;
    if (n == 0) {
        return std::optional<scene::Bounds>{};
    }
    constexpr uint32_t kChunk = 4096;
    const uint32_t points = n * 2;
    const uint32_t chunks = (points + kChunk - 1) / kChunk;
    auto corners = deviceBuffer(*device_, points, 16, "scene.bounds.corners");
    if (!corners) return std::move(corners).error();
    auto extents = deviceBuffer(*device_, uint64_t{chunks} * 2, 16, "scene.bounds.extents");
    if (!extents) return std::move(extents).error();
    auto result = deviceBuffer(*device_, 2, 16, "scene.bounds");
    if (!result) return std::move(result).error();
    gpu::CommandBatch batch(*device_);
    worldBoxes_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["meshes"].setBinding(meshRecords_.rhi());
        cursor["instances"].setBinding(instanceRecords_.rhi());
        cursor["corners"].setBinding(corners->rhi());
        cursor["params"]["count"].setData(n);
    });
    const auto params = [&](rhi::ShaderCursor cursor) {
        cursor["params"]["count"].setData(points);
        cursor["params"]["chunkSize"].setData(kChunk);
        cursor["params"]["chunkCount"].setData(chunks);
    };
    boundsChunks_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(corners->rhi());
        cursor["extents"].setBinding(extents->rhi());
        params(cursor);
    });
    boundsReduce_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["extents"].setBinding(extents->rhi());
        cursor["result"].setBinding(result->rhi());
        params(cursor);
    });
    LRT_TRY(batch.submit(true));
    float box[8];
    LRT_TRY(result->read(*device_, 0, sizeof(box), box));
    scene::Bounds bounds;
    std::copy(box, box + 3, bounds.min.begin());
    std::copy(box + 4, box + 7, bounds.max.begin());
    return std::optional<scene::Bounds>(bounds);
}

}   // namespace lrt::world
