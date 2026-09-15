// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/light/LightTable.h"

#include <algorithm>

#include <functional>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ShaderLibrary.h"

#include <cmath>
#include <vector>

#include "lrt/gpu/Device.h"

namespace lrt::light {

Result<LightTable> LightTable::create(gpu::ShaderLibrary& library) {
    gpu::Device& device = library.device();
    LightTable table;
    table.device_ = &device;
    auto prefix = gpu::ComputeKernel::create(library, "lrt/light/light_prefix", "lightPrefix");
    if (!prefix) return std::move(prefix).error();
    auto iesPrepare = gpu::ComputeKernel::create(library, "lrt/light/ies_prepare", "iesPrepare");
    if (!iesPrepare) return std::move(iesPrepare).error();
    auto instances = gpu::ComputeKernel::create(library, "lrt/light/light_instances", "lightInstances");
    if (!instances) return std::move(instances).error();
    const auto make = [&](std::optional<gpu::ComputeKernel>& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into.emplace(std::move(*kernel));
        return ok();
    };
    LightTable bvhKernels;
    LRT_TRY(make(bvhKernels.bvhLeaves_, "lrt/light/light_bvh_build", "lightLeaves"));
    LRT_TRY(make(bvhKernels.bvhPack_, "lrt/light/light_bvh_build", "lightPack"));
    LRT_TRY(make(bvhKernels.bvhParents_, "lrt/light/light_bvh_build", "lightParents"));
    LRT_TRY(make(bvhKernels.bvhSettle_, "lrt/light/light_bvh_build", "lightSettle"));
    LRT_TRY(make(bvhKernels.bvhUnbounded_, "lrt/light/light_bvh_build", "lightUnbounded"));
    LRT_TRY(make(bvhKernels.bvhLeafIndices_, "lrt/light/light_bvh_build", "lightLeafIndices"));
    LRT_TRY(make(bvhKernels.hierarchy_, "lrt/rt/bvh_hierarchy", "bvhHierarchy"));
    LRT_TRY(make(bvhKernels.refit_, "lrt/rt/bvh_refit", "bvhRefit"));
    LRT_TRY(make(bvhKernels.countNonzero_, "lrt/reference/count_nonzero", "countNonzero"));
    LRT_TRY(make(bvhKernels.boundsChunks_, "lrt/scene/bounds_chunks", "boundsChunks"));
    LRT_TRY(make(bvhKernels.boundsReduce_, "lrt/scene/bounds_reduce", "boundsReduce"));
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    bvhKernels.sort_.emplace(std::move(*sort));
    table.prefix_.emplace(std::move(*prefix));
    table.iesPrepare_.emplace(std::move(*iesPrepare));
    table.instances_.emplace(std::move(*instances));    table.bvhLeaves_ = std::move(bvhKernels.bvhLeaves_);
    table.bvhPack_ = std::move(bvhKernels.bvhPack_);
    table.bvhParents_ = std::move(bvhKernels.bvhParents_);
    table.bvhSettle_ = std::move(bvhKernels.bvhSettle_);
    table.bvhUnbounded_ = std::move(bvhKernels.bvhUnbounded_);
    table.bvhLeafIndices_ = std::move(bvhKernels.bvhLeafIndices_);
    table.hierarchy_ = std::move(bvhKernels.hierarchy_);
    table.refit_ = std::move(bvhKernels.refit_);
    table.countNonzero_ = std::move(bvhKernels.countNonzero_);
    table.boundsChunks_ = std::move(bvhKernels.boundsChunks_);
    table.boundsReduce_ = std::move(bvhKernels.boundsReduce_);
    table.sort_ = std::move(bvhKernels.sort_);
    return table;
}

LightRecord LightTable::recordOf(const Light& light) {
    LightRecord record;
    record.kind = static_cast<uint32_t>(light.kind);
    record.flags = (light.shadow ? kLightShadow : 0u) | (light.normalize ? kLightNormalize : 0u) |
                   (light.enableTemperature ? kLightTemperature : 0u) | (light.iesNormalize ? kLightIesNormalize : 0u);
    record.iesAngleScale = light.iesAngleScale;
    switch (light.kind) {
        case LightKind::Distant:
            record.sizeX = light.angle;
            break;
        case LightKind::Sphere:
        case LightKind::Disk:
            record.sizeX = light.radius;
            break;
        case LightKind::Rect:
            record.sizeX = light.width;
            record.sizeY = light.height;
            break;
        case LightKind::Cylinder:
            record.sizeX = light.radius;
            record.sizeY = light.length;
            break;
        case LightKind::Dome:
            break;
    }
    for (int c = 0; c < 3; ++c) {
        record.colour[c] = light.colour[c] * light.intensity;
    }
    record.exposure = light.exposure;
    record.temperature = light.temperature;
    // A cone of a half angle at or above a right angle shapes nothing.
    const bool shaped = light.coneAngle > 0.0F && light.coneAngle < 3.14159265358979F / 2.0F;
    record.coneCos = shaped ? std::cos(light.coneAngle) : -1.0F;
    record.coneSoftness = light.coneSoftness;
    record.texture = light.textureId;
    record.sampler = light.sampler;
    record.group = light.groupIndex;
    record.lightCategory = light.lightCategory;
    record.shadowCategory = light.shadowCategory;
    const std::array<float, 12> rows = light.lightToWorld.rows3x4();
    for (size_t k = 0; k < rows.size(); ++k) {
        record.rows[k] = rows[k];
    }
    return record;
}

/// What a light is worth to a frame: its emission times what it emits over.
Result<void> LightTable::set(std::span<const Light> lights, float sceneRadius) {
    std::vector<LightRecord> records;
    records.reserve(lights.size() + 1);
    shadows_ = false;
    domes_ = false;
    // The frame's IES profiles, each once, concatenated into one values
    // buffer: angle lists and candela tables as authored.
    struct Expansion {
        uint32_t           base;
        uint32_t           count;
        const gpu::Buffer* rows;
    };
    std::vector<Expansion> expansions;
    std::vector<const io::IesProfile*> profiles;
    std::vector<IesRecord> iesRecords;
    std::vector<float> iesValues;
    for (const Light& light : lights) {
        LightRecord record = recordOf(light);
        if (light.ies && !light.ies->vertical.empty() && !light.ies->candela.empty()) {
            size_t row = 0;
            for (; row < profiles.size(); ++row) {
                if (profiles[row] == light.ies.get()) break;
            }
            if (row == profiles.size()) {
                const io::IesProfile& p = *light.ies;
                IesRecord ies;
                ies.verticalOffset = static_cast<uint32_t>(iesValues.size());
                ies.verticalCount = static_cast<uint32_t>(p.vertical.size());
                iesValues.insert(iesValues.end(), p.vertical.begin(), p.vertical.end());
                ies.horizontalOffset = static_cast<uint32_t>(iesValues.size());
                ies.horizontalCount = static_cast<uint32_t>(p.horizontal.size());
                iesValues.insert(iesValues.end(), p.horizontal.begin(), p.horizontal.end());
                ies.candelaOffset = static_cast<uint32_t>(iesValues.size());
                iesValues.insert(iesValues.end(), p.candela.begin(), p.candela.end());
                ies.photometricType = p.photometricType;
                ies.multiplier = p.multiplier;
                profiles.push_back(light.ies.get());
                iesRecords.push_back(ies);
            }
            record.ies = static_cast<uint32_t>(row);
        }
        if (light.instanceRows != nullptr && light.instanceCount > 0) {
            // Once per instance; the placements are the device's to write.
            expansions.push_back({static_cast<uint32_t>(records.size()), light.instanceCount, light.instanceRows});
            records.insert(records.end(), light.instanceCount, record);
        } else {
            records.push_back(record);
        }
        shadows_ = shadows_ || light.shadow;
        domes_ = domes_ || light.kind == LightKind::Dome;
    }
    iesCount_ = static_cast<uint32_t>(iesRecords.size());
    if (iesRecords.empty()) {
        iesRecords.emplace_back();   // a buffer to bind, which nothing reads
    }
    // The tree over the bounded lights, the list of the unbounded: which is
    // which is the record's kind, bookkeeping. The nodes go after the IES
    // values in the same buffer, sixteen floats each.
    std::vector<uint32_t> bounded;
    std::vector<uint32_t> unbounded;
    for (uint32_t k = 0; k < records.size(); ++k) {
        if (lights.empty()) break;
        const uint32_t kind = records[k].kind;
        const bool isUnbounded = kind == static_cast<uint32_t>(LightKind::Distant) ||
                                 kind == static_cast<uint32_t>(LightKind::Dome);
        (isUnbounded ? unbounded : bounded).push_back(k);
    }
    nodeBase_ = static_cast<uint32_t>(iesValues.size());
    const uint32_t nodeCount = (bounded.empty() ? 0u : 2u * static_cast<uint32_t>(bounded.size()) - 1u) +
                               static_cast<uint32_t>(unbounded.size());
    iesValues.resize(iesValues.size() + size_t{nodeCount} * 16, 0.0F);
    if (iesValues.empty()) {
        iesValues.push_back(0.0F);
    }
    {
        auto madeRecords = gpu::Buffer::fromSpan<IesRecord>(*device_, iesRecords, "lights.ies.records");
        if (!madeRecords) return std::move(madeRecords).error();
        iesRecords_ = std::move(*madeRecords);
        auto madeValues = gpu::Buffer::fromSpan<float>(*device_, iesValues, "lights.ies.values");
        if (!madeValues) return std::move(madeValues).error();
        iesValues_ = std::move(*madeValues);
    }
    if (iesCount_ > 0) {
        // Each profile's power -- its intensity integrated over the sphere,
        // over the patches it defines -- is a statistic, so a kernel's.
        gpu::CommandBatch batch(*device_);
        iesPrepare_->dispatch(batch, {iesCount_, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["iesRecords"].setBinding(iesRecords_.rhi());
            cursor["iesValues"].setBinding(iesValues_.rhi());
            cursor["iesCount"].setData(iesCount_);
        });
        LRT_TRY(batch.submit(true));
    }
    if (records.empty()) {
        records.emplace_back();   // a buffer to bind, which nothing reads
    }
    count_ = static_cast<uint32_t>(records.size()) - (lights.empty() ? 1u : 0u);
    if (records.size() > capacity_ || !records_.valid()) {
        auto made = gpu::Buffer::fromSpan<LightRecord>(*device_, records, "lights.records");
        if (!made) return std::move(made).error();
        records_ = std::move(*made);
        capacity_ = static_cast<uint32_t>(records.size());
    } else if (!records_.write(*device_, 0, records.size() * sizeof(LightRecord), records.data())) {
        return Error(ErrorCode::DeviceFailure, "lights: cannot upload the frame's records");
    }
    // Each light's share of the frame's power, accumulated on the device:
    // the host uploads what was authored and computes nothing on it. Before
    // that, an instanced light's copies are placed.
    if (count_ > 0) {
        gpu::CommandBatch batch(*device_);
        for (const Expansion& e : expansions) {
            instances_->dispatch(batch, {e.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["records"].setBinding(records_.rhi());
                cursor["chainRows"].setBinding(e.rows->rhi());
                cursor["base"].setData(e.base);
                cursor["count"].setData(e.count);
            });
        }
        prefix_->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["records"].setBinding(records_.rhi());
            cursor["count"].setData(count_);
            cursor["sceneRadius"].setData(std::max(sceneRadius, 1.0e-6F));
        });
        LRT_TRY(batch.submit(true));
    }
    return buildBvh(bounded, unbounded);
}

Result<void> LightTable::buildBvh(const std::vector<uint32_t>& bounded, const std::vector<uint32_t>& unbounded) {
    gpu::Device& device = *device_;
    const uint32_t n = static_cast<uint32_t>(bounded.size());
    treeNodes_ = n > 0 ? 2 * n - 1 : 0;
    unbounded_ = static_cast<uint32_t>(unbounded.size());
    const auto buffer = [&](uint64_t count, uint32_t element, const char* label) -> Result<gpu::Buffer> {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(count, 1) * element;
        desc.elementBytes = element;
        desc.label = label;
        return gpu::Buffer::create(device, desc);
    };
    if (treeNodes_ + unbounded_ == 0) {
        return ok();
    }
    const auto bvhParams = [&](rhi::ShaderCursor p, uint32_t count, const float* lo, const float* hi) {
        p["count"].setData(count);
        p["nodeBase"].setData(uint32_t{0});
        p["leafBase"].setData(uint32_t{0});
        p["boundsLoX"].setData(lo[0]);
        p["boundsLoY"].setData(lo[1]);
        p["boundsLoZ"].setData(lo[2]);
        p["boundsHiX"].setData(hi[0]);
        p["boundsHiY"].setData(hi[1]);
        p["boundsHiZ"].setData(hi[2]);
    };
    const float zero[3] = {0.0F, 0.0F, 0.0F};
    if (n > 0) {
        auto leafLights = gpu::Buffer::fromSpan<uint32_t>(device, bounded, "lights.bvh.leaves");
        if (!leafLights) return std::move(leafLights).error();
        auto leafBoxes = buffer(uint64_t{n} * 2, 16, "lights.bvh.leafBoxes");
        if (!leafBoxes) return std::move(leafBoxes).error();
        gpu::SortBuffers sorting;
        for (auto [into, label] : {std::pair{&sorting.keysLo, "lights.bvh.keys"}, std::pair{&sorting.values, "lights.bvh.order"},
                                   std::pair{&sorting.scratchKeysLo, "lights.bvh.keys2"},
                                   std::pair{&sorting.scratchValues, "lights.bvh.order2"}}) {
            auto made = buffer(n, 4, label);
            if (!made) return std::move(made).error();
            *into = std::move(*made);
        }
        // The lights' boxes, then their bounds (a kernel's), for the Morton codes.
        {
            gpu::CommandBatch batch(device);
            bvhLeaves_->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["lights"].setBinding(records_.rhi());
                cursor["leafLights"].setBinding(leafLights->rhi());
                cursor["leafBoxes"].setBinding(leafBoxes->rhi());
                cursor["mortonKeys"].setBinding(sorting.keysLo.rhi());
                cursor["mortonValues"].setBinding(sorting.values.rhi());
                bvhParams(cursor["params"], n, zero, zero);
            });
            LRT_TRY(batch.submit(true));
        }
        float box[8] = {};
        {
            constexpr uint32_t kChunk = 4096;
            const uint32_t points = n * 2;
            const uint32_t chunks = (points + kChunk - 1) / kChunk;
            auto extents = buffer(uint64_t{chunks} * 2, 16, "lights.bvh.extents");
            if (!extents) return std::move(extents).error();
            auto result = buffer(2, 16, "lights.bvh.bounds");
            if (!result) return std::move(result).error();
            gpu::CommandBatch batch(device);
            const auto params = [&](rhi::ShaderCursor cursor) {
                cursor["params"]["count"].setData(points);
                cursor["params"]["chunkSize"].setData(kChunk);
                cursor["params"]["chunkCount"].setData(chunks);
            };
            boundsChunks_->dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(leafBoxes->rhi());
                cursor["extents"].setBinding(extents->rhi());
                params(cursor);
            });
            boundsReduce_->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["extents"].setBinding(extents->rhi());
                cursor["result"].setBinding(result->rhi());
                params(cursor);
            });
            LRT_TRY(batch.submit(true));
            LRT_TRY(result->read(device, 0, sizeof(box), box));
        }
        // The codes again within the bounds, sorted, and the hierarchy over them.
        auto boxes = buffer(uint64_t{std::max(n, 2u) - 1} * 2, 16, "lights.bvh.boxes");
        if (!boxes) return std::move(boxes).error();
        auto children = buffer(uint64_t{std::max(n, 2u) - 1} * 2, 4, "lights.bvh.children");
        if (!children) return std::move(children).error();
        auto leaves = buffer(n, 4, "lights.bvh.leafOrder");
        if (!leaves) return std::move(leaves).error();
        auto changed = buffer(std::max(n, 2u) - 1, 4, "lights.bvh.changed");
        if (!changed) return std::move(changed).error();
        auto changedCount = buffer(1, 4, "lights.bvh.changedCount");
        if (!changedCount) return std::move(changedCount).error();
        {
            gpu::CommandBatch batch(device);
            bvhLeaves_->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["lights"].setBinding(records_.rhi());
                cursor["leafLights"].setBinding(leafLights->rhi());
                cursor["leafBoxes"].setBinding(leafBoxes->rhi());
                cursor["mortonKeys"].setBinding(sorting.keysLo.rhi());
                cursor["mortonValues"].setBinding(sorting.values.rhi());
                bvhParams(cursor["params"], n, box, box + 4);
            });
            LRT_TRY(sort_->sort(batch, sorting, n, 30));
            hierarchy_->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["sortedKeys"].setBinding(sorting.keysLo.rhi());
                cursor["sortedValues"].setBinding(sorting.values.rhi());
                cursor["children"].setBinding(children->rhi());
                cursor["leaves"].setBinding(leaves->rhi());
                bvhParams(cursor["params"], n, box, box + 4);
            });
            LRT_TRY(batch.submit(true));
        }
        // Boxes settled up the tree, then the nodes, then power and cones.
        const auto settle = [&](const std::optional<gpu::ComputeKernel>& kernel,
                                const std::function<void(rhi::ShaderCursor)>& bind) -> Result<void> {
            if (n < 2) return ok();
            for (uint32_t passes = 0; passes < 256; passes += 8) {
                gpu::CommandBatch batch(device);
                for (uint32_t k = 0; k < 8; ++k) {
                    kernel->dispatch(batch, {n - 1, 1, 1}, bind);
                }
                countNonzero_->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["values"].setBinding(changed->rhi());
                    cursor["total"].setBinding(changedCount->rhi());
                    cursor["params"]["count"].setData(n - 1);
                });
                LRT_TRY(batch.submit(true));
                uint32_t still = 0;
                LRT_TRY(changedCount->read(device, 0, sizeof(still), &still));
                if (still == 0) return ok();
            }
            return Error(ErrorCode::InternalError, "light BVH did not settle");
        };
        LRT_TRY(settle(refit_, [&](rhi::ShaderCursor cursor) {
            cursor["leafBoxes"].setBinding(leafBoxes->rhi());
            cursor["children"].setBinding(children->rhi());
            cursor["leaves"].setBinding(leaves->rhi());
            cursor["boxes"].setBinding(boxes->rhi());
            cursor["changed"].setBinding(changed->rhi());
            bvhParams(cursor["params"], n, box, box + 4);
        }));
        {
            gpu::CommandBatch batch(device);
            const auto bindPack = [&](rhi::ShaderCursor cursor) {
                cursor["lights"].setBinding(records_.rhi());
                cursor["records"].setBinding(records_.rhi());
                cursor["leafLights"].setBinding(leafLights->rhi());
                cursor["boxes"].setBinding(boxes->rhi());
                cursor["children"].setBinding(children->rhi());
                cursor["leaves"].setBinding(leaves->rhi());
                cursor["nodeValues"].setBinding(iesValues_.rhi());
                cursor["nodeBase"].setData(nodeBase_);
                cursor["changed"].setBinding(changed->rhi());
                bvhParams(cursor["params"], n, box, box + 4);
            };
            bvhPack_->dispatch(batch, {treeNodes_, 1, 1}, bindPack);
            bvhParents_->dispatch(batch, {std::max(n, 2u) - 1, 1, 1}, bindPack);
            bvhLeafIndices_->dispatch(batch, {n, 1, 1}, bindPack);
            LRT_TRY(batch.submit(true));
        }
        LRT_TRY(settle(bvhSettle_, [&](rhi::ShaderCursor cursor) {
            cursor["nodeValues"].setBinding(iesValues_.rhi());
            cursor["nodeBase"].setData(nodeBase_);
            cursor["changed"].setBinding(changed->rhi());
            bvhParams(cursor["params"], n, box, box + 4);
        }));
    }
    if (unbounded_ > 0) {
        auto list = gpu::Buffer::fromSpan<uint32_t>(device, unbounded, "lights.bvh.unbounded");
        if (!list) return std::move(list).error();
        gpu::CommandBatch batch(device);
        bvhUnbounded_->dispatch(batch, {unbounded_, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["records"].setBinding(records_.rhi());
            cursor["unboundedLights"].setBinding(list->rhi());
            cursor["nodeValues"].setBinding(iesValues_.rhi());
            cursor["nodeBase"].setData(nodeBase_);
            cursor["treeNodeCount"].setData(treeNodes_);
            bvhParams(cursor["params"], unbounded_, zero, zero);
        });
        LRT_TRY(batch.submit(true));
    }
    return ok();
}

void LightTable::bind(rhi::ShaderCursor cursor) const {
    cursor["lights"].setBinding(records_.rhi());
    cursor["lightCount"].setData(count_);
    // The tree's uniforms only where a kernel declares them: a kernel that
    // samples lights by power alone has none.
    if (const rhi::ShaderCursor base = cursor["lightNodeBase"]; base.isValid()) {
        base.setData(nodeBase_);
        cursor["lightTreeNodes"].setData(treeNodes_);
        cursor["lightUnboundedCount"].setData(unbounded_);
    }
    cursor["iesRecords"].setBinding(iesRecords_.rhi());
    cursor["iesValues"].setBinding(iesValues_.rhi());
}

}   // namespace lrt::light
