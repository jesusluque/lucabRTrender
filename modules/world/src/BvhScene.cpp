// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/BvhScene.h"

#include <algorithm>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::world {
namespace {

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

}   // namespace

Result<BvhScene> BvhScene::create(gpu::ShaderLibrary& library) {
    BvhScene bvh;
    bvh.device_ = &library.device();
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    bvh.sort_ = std::move(*sort);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    LRT_TRY(make(bvh.triangleLeaves_, "lrt/world/bvh_scene", "triangleLeaves"));
    LRT_TRY(make(bvh.instanceLeaves_, "lrt/world/bvh_scene", "instanceLeaves"));
    LRT_TRY(make(bvh.instanceCodes_, "lrt/world/bvh_scene", "instanceCodes"));
    LRT_TRY(make(bvh.hierarchy_, "lrt/rt/bvh_hierarchy", "bvhHierarchy"));
    LRT_TRY(make(bvh.refit_, "lrt/rt/bvh_refit", "bvhRefit"));
    LRT_TRY(make(bvh.count_, "lrt/reference/count_nonzero", "countNonzero"));
    LRT_TRY(make(bvh.boundsChunks_, "lrt/scene/bounds_chunks", "boundsChunks"));
    LRT_TRY(make(bvh.boundsReduce_, "lrt/scene/bounds_reduce", "boundsReduce"));
    return bvh;
}

Result<void> BvhScene::hierarchy(const Build& build, gpu::SortBuffers& sorting, const gpu::Buffer& leafBoxes,
                                 gpu::Buffer& boxes, gpu::Buffer& children, gpu::Buffer& leaves) {
    gpu::Device& device = *device_;
    const uint32_t n = build.count;
    const auto setBvh = [&](rhi::ShaderCursor p) {
        p["count"].setData(n);
        p["nodeBase"].setData(build.nodeBase);
        p["leafBase"].setData(build.leafBase);
        p["boundsLoX"].setData(build.lo[0]);
        p["boundsLoY"].setData(build.lo[1]);
        p["boundsLoZ"].setData(build.lo[2]);
        p["boundsHiX"].setData(build.hi[0]);
        p["boundsHiY"].setData(build.hi[1]);
        p["boundsHiZ"].setData(build.hi[2]);
    };
    {
        gpu::CommandBatch batch(device);
        LRT_TRY(sort_.sort(batch, sorting, n, 30));
        hierarchy_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["sortedKeys"].setBinding(sorting.keysLo.rhi());
            cursor["sortedValues"].setBinding(sorting.values.rhi());
            cursor["children"].setBinding(children.rhi());
            cursor["leaves"].setBinding(leaves.rhi());
            setBvh(cursor["params"]);
        });
        LRT_TRY(batch.submit(true));
    }
    return settle(build, leafBoxes, boxes, children, leaves);
}

Result<void> BvhScene::settle(const Build& build, const gpu::Buffer& leafBoxes, gpu::Buffer& boxes,
                              const gpu::Buffer& children, const gpu::Buffer& leaves) {
    gpu::Device& device = *device_;
    const uint32_t n = build.count;
    const auto setBvh = [&](rhi::ShaderCursor p) {
        p["count"].setData(n);
        p["nodeBase"].setData(build.nodeBase);
        p["leafBase"].setData(build.leafBase);
        p["boundsLoX"].setData(build.lo[0]);
        p["boundsLoY"].setData(build.lo[1]);
        p["boundsLoZ"].setData(build.lo[2]);
        p["boundsHiX"].setData(build.hi[0]);
        p["boundsHiY"].setData(build.hi[1]);
        p["boundsHiZ"].setData(build.hi[2]);
    };
    if (n < 2) {
        return ok();
    }
    auto changed = deviceBuffer(device, n - 1, 4, "bvh.changed");
    if (!changed) return std::move(changed).error();
    auto changedCount = deviceBuffer(device, 1, 4, "bvh.changedCount");
    if (!changedCount) return std::move(changedCount).error();
    static constexpr uint32_t kPassesPerCheck = 8;
    static constexpr uint32_t kMaxPasses = 256;
    for (uint32_t passes = 0; passes < kMaxPasses; passes += kPassesPerCheck) {
        gpu::CommandBatch batch(device);
        for (uint32_t k = 0; k < kPassesPerCheck; ++k) {
            refit_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["leafBoxes"].setBinding(leafBoxes.rhi());
                cursor["children"].setBinding(children.rhi());
                cursor["leaves"].setBinding(leaves.rhi());
                cursor["boxes"].setBinding(boxes.rhi());
                cursor["changed"].setBinding(changed->rhi());
                setBvh(cursor["params"]);
            });
        }
        count_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["values"].setBinding(changed->rhi());
            cursor["total"].setBinding(changedCount->rhi());
            cursor["params"]["count"].setData(n - 1);
        });
        LRT_TRY(batch.submit(true));
        uint32_t still = 0;
        LRT_TRY(changedCount->read(device, 0, sizeof(still), &still));
        if (still == 0) {
            return ok();
        }
    }
    return Error::make(ErrorCode::InternalError, "BVH refit of {} leaves did not settle", n);
}

Result<void> BvhScene::build(const GpuScene& scene, bool refit) {
    gpu::Device& device = *device_;
    const auto sortBuffers = [&](uint32_t n, gpu::SortBuffers& sorting) -> Result<void> {
        for (auto [into, label] : {std::pair{&sorting.keysLo, "bvh.keys"}, std::pair{&sorting.values, "bvh.order"},
                                   std::pair{&sorting.scratchKeysLo, "bvh.keys2"},
                                   std::pair{&sorting.scratchValues, "bvh.order2"}}) {
            auto made = deviceBuffer(device, n, 4, label);
            if (!made) return std::move(made).error();
            *into = std::move(*made);
        }
        return ok();
    };

    // Per mesh, when the pools changed.
    if (scene.generation() != generation_) {
        meshBuilds_.clear();
        meshRevisions_.clear();
        uint64_t nodes = 0;
        uint64_t triangles = 0;
        for (uint32_t m = 0; m < scene.meshCount(); ++m) {
            nodes += std::max<uint32_t>(scene.mesh(m).triangles, 1) - 1;
            triangles = std::max<uint64_t>(triangles, uint64_t{scene.firstTriangle(m)} + scene.mesh(m).triangles);
        }
        auto boxes = deviceBuffer(device, nodes * 2, 16, "bvh.mesh.boxes");
        if (!boxes) return std::move(boxes).error();
        auto children = deviceBuffer(device, nodes * 2, 4, "bvh.mesh.children");
        if (!children) return std::move(children).error();
        auto leaves = deviceBuffer(device, triangles, 4, "bvh.mesh.leaves");
        if (!leaves) return std::move(leaves).error();
        meshBoxes_ = std::move(*boxes);
        meshChildren_ = std::move(*children);
        meshLeaves_ = std::move(*leaves);
        uint32_t nodeBase = 0;
        for (uint32_t m = 0; m < scene.meshCount(); ++m) {
            const geom::GpuMesh& mesh = scene.mesh(m);
            const uint32_t n = mesh.triangles;
            Build build;
            build.count = n;
            build.nodeBase = nodeBase;
            build.leafBase = scene.firstTriangle(m);
            for (int k = 0; k < 3; ++k) {
                build.lo[k] = mesh.bounds.min[static_cast<size_t>(k)];
                build.hi[k] = mesh.bounds.max[static_cast<size_t>(k)];
            }
            gpu::SortBuffers sorting;
            LRT_TRY(sortBuffers(n, sorting));
            auto leafBoxes = deviceBuffer(device, uint64_t{n} * 2, 16, "bvh.mesh.leafBoxes");
            if (!leafBoxes) return std::move(leafBoxes).error();
            {
                gpu::CommandBatch batch(device);
                triangleLeaves_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["positions"].setBinding(scene.positions().rhi());
                    cursor["indices"].setBinding(scene.indices().rhi());
                    cursor["leafBoxes"].setBinding(leafBoxes->rhi());
                    cursor["mortonKeys"].setBinding(sorting.keysLo.rhi());
                    cursor["mortonValues"].setBinding(sorting.values.rhi());
                    rhi::ShaderCursor p = cursor["params"];
                    p["count"].setData(n);
                    p["boundsLoX"].setData(build.lo[0]);
                    p["boundsLoY"].setData(build.lo[1]);
                    p["boundsLoZ"].setData(build.lo[2]);
                    p["boundsHiX"].setData(build.hi[0]);
                    p["boundsHiY"].setData(build.hi[1]);
                    p["boundsHiZ"].setData(build.hi[2]);
                    cursor["mesh"]["firstPoint"].setData(scene.firstPoint(m));
                    cursor["mesh"]["firstTriangle"].setData(scene.firstTriangle(m));
                });
                LRT_TRY(batch.submit(true));
            }
            LRT_TRY(hierarchy(build, sorting, *leafBoxes, meshBoxes_, meshChildren_, meshLeaves_));
            nodeBase += std::max<uint32_t>(n, 1) - 1;
            meshBuilds_.push_back(build);
            meshRevisions_.push_back(scene.meshRevision(m));
        }
        generation_ = scene.generation();
    } else if (refit) {
        // A mesh deformed in place: its leaves' boxes again from the pool's
        // positions, and the tree it has settled over them. The tree was
        // shaped by the old positions, so its boxes only get looser -- never
        // wrong -- until the next repack reshapes it.
        for (uint32_t m = 0; m < scene.meshCount(); ++m) {
            if (scene.meshRevision(m) == meshRevisions_[m]) {
                continue;
            }
            const Build& build = meshBuilds_[m];
            const uint32_t n = build.count;
            gpu::SortBuffers sorting;
            LRT_TRY(sortBuffers(n, sorting));   // the codes are written and not sorted: a refit keeps its order
            auto leafBoxes = deviceBuffer(device, uint64_t{n} * 2, 16, "bvh.mesh.leafBoxes");
            if (!leafBoxes) return std::move(leafBoxes).error();
            {
                gpu::CommandBatch batch(device);
                triangleLeaves_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["positions"].setBinding(scene.positions().rhi());
                    cursor["indices"].setBinding(scene.indices().rhi());
                    cursor["leafBoxes"].setBinding(leafBoxes->rhi());
                    cursor["mortonKeys"].setBinding(sorting.keysLo.rhi());
                    cursor["mortonValues"].setBinding(sorting.values.rhi());
                    rhi::ShaderCursor p = cursor["params"];
                    p["count"].setData(n);
                    p["boundsLoX"].setData(build.lo[0]);
                    p["boundsLoY"].setData(build.lo[1]);
                    p["boundsLoZ"].setData(build.lo[2]);
                    p["boundsHiX"].setData(build.hi[0]);
                    p["boundsHiY"].setData(build.hi[1]);
                    p["boundsHiZ"].setData(build.hi[2]);
                    cursor["mesh"]["firstPoint"].setData(scene.firstPoint(m));
                    cursor["mesh"]["firstTriangle"].setData(scene.firstTriangle(m));
                });
                LRT_TRY(batch.submit(true));
            }
            LRT_TRY(settle(build, *leafBoxes, meshBoxes_, meshChildren_, meshLeaves_));
            meshRevisions_[m] = scene.meshRevision(m);
        }
    }

    // The instances, every frame.
    const uint32_t n = scene.instanceCount();
    auto boxes = deviceBuffer(device, uint64_t{std::max<uint32_t>(n, 2) - 1} * 2, 16, "bvh.top.boxes");
    if (!boxes) return std::move(boxes).error();
    auto children = deviceBuffer(device, uint64_t{std::max<uint32_t>(n, 2) - 1} * 2, 4, "bvh.top.children");
    if (!children) return std::move(children).error();
    auto leaves = deviceBuffer(device, n, 4, "bvh.top.leaves");
    if (!leaves) return std::move(leaves).error();
    topBoxes_ = std::move(*boxes);
    topChildren_ = std::move(*children);
    topLeaves_ = std::move(*leaves);
    if (n == 0) {
        return ok();
    }
    auto leafBoxes = deviceBuffer(device, uint64_t{n} * 2, 16, "bvh.top.leafBoxes");
    if (!leafBoxes) return std::move(leafBoxes).error();
    auto centres = deviceBuffer(device, n, 16, "bvh.top.centres");
    if (!centres) return std::move(centres).error();
    gpu::SortBuffers sorting;
    LRT_TRY(sortBuffers(n, sorting));
    Build build;
    build.count = n;
    {
        constexpr uint32_t kChunk = 4096;
        const uint32_t chunks = (n + kChunk - 1) / kChunk;
        auto extents = deviceBuffer(device, uint64_t{chunks} * 2, 16, "bvh.top.extents");
        if (!extents) return std::move(extents).error();
        auto result = deviceBuffer(device, 2, 16, "bvh.top.bounds");
        if (!result) return std::move(result).error();
        gpu::CommandBatch batch(device);
        instanceLeaves_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["meshes"].setBinding(scene.meshRecords().rhi());
            cursor["instances"].setBinding(scene.instanceRecords().rhi());
            cursor["leafBoxes"].setBinding(leafBoxes->rhi());
            cursor["centres"].setBinding(centres->rhi());
            cursor["params"]["count"].setData(n);
        });
        const auto params = [&](rhi::ShaderCursor cursor) {
            cursor["params"]["count"].setData(n);
            cursor["params"]["chunkSize"].setData(kChunk);
            cursor["params"]["chunkCount"].setData(chunks);
        };
        boundsChunks_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(centres->rhi());
            cursor["extents"].setBinding(extents->rhi());
            params(cursor);
        });
        boundsReduce_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["extents"].setBinding(extents->rhi());
            cursor["result"].setBinding(result->rhi());
            params(cursor);
        });
        LRT_TRY(batch.submit(true));
        float extent[8];
        LRT_TRY(result->read(device, 0, sizeof(extent), extent));
        std::copy(extent, extent + 3, build.lo);
        std::copy(extent + 4, extent + 7, build.hi);
    }
    {
        gpu::CommandBatch batch(device);
        instanceCodes_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["centres"].setBinding(centres->rhi());
            cursor["mortonKeys"].setBinding(sorting.keysLo.rhi());
            cursor["mortonValues"].setBinding(sorting.values.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(n);
            p["boundsLoX"].setData(build.lo[0]);
            p["boundsLoY"].setData(build.lo[1]);
            p["boundsLoZ"].setData(build.lo[2]);
            p["boundsHiX"].setData(build.hi[0]);
            p["boundsHiY"].setData(build.hi[1]);
            p["boundsHiZ"].setData(build.hi[2]);
        });
        LRT_TRY(batch.submit(true));
    }
    return hierarchy(build, sorting, *leafBoxes, topBoxes_, topChildren_, topLeaves_);
}

}   // namespace lrt::world
