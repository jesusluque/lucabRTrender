// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/geom/Mesh.h"

#include <algorithm>
#include <bit>

#include "lrt/core/Log.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::geom {
namespace {

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

/// An int array's bytes into a device buffer (one word per int).
Result<gpu::Buffer> intBuffer(gpu::Device& device, std::span<const int32_t> values, const char* label) {
    auto made = deviceBuffer(device, values.size(), 4, label);
    if (!made) return std::move(made).error();
    if (!values.empty()) {
        LRT_TRY(made->write(device, 0, values.size_bytes(), values.data()));
    }
    return made;
}

}   // namespace

Result<MeshBuilder> MeshBuilder::create(gpu::ShaderLibrary& library) {
    MeshBuilder b;
    b.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    b.prefix_ = std::move(*prefix);
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    b.sort_ = std::move(*sort);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    LRT_TRY(make(b.points_, "lrt/geom/mesh_topology", "meshPoints"));
    LRT_TRY(make(b.holes_, "lrt/geom/mesh_topology", "meshHoles"));
    LRT_TRY(make(b.faceCounts_, "lrt/geom/mesh_topology", "meshFaceCounts"));
    LRT_TRY(make(b.triangulate_, "lrt/geom/mesh_topology", "meshTriangulate"));
    LRT_TRY(make(b.cornerKeys_, "lrt/geom/mesh_normals", "cornerKeys"));
    LRT_TRY(make(b.clearRuns_, "lrt/geom/mesh_normals", "clearRuns"));
    LRT_TRY(make(b.pointRuns_, "lrt/geom/mesh_normals", "pointRuns"));
    LRT_TRY(make(b.pointNormals_, "lrt/geom/mesh_normals", "pointNormals"));
    LRT_TRY(make(b.boundsChunks_, "lrt/scene/bounds_chunks", "boundsChunks"));
    LRT_TRY(make(b.boundsReduce_, "lrt/scene/bounds_reduce", "boundsReduce"));
    LRT_TRY(make(b.expand_, "lrt/geom/primvar_expand", "primvarExpand"));
    LRT_TRY(make(b.subsetClear_, "lrt/geom/mesh_subsets", "subsetClear"));
    LRT_TRY(make(b.subsetScatter_, "lrt/geom/mesh_subsets", "subsetScatter"));
    LRT_TRY(make(b.subsetTriangles_, "lrt/geom/mesh_subsets", "subsetTriangles"));
    return b;
}

Result<GpuMesh> MeshBuilder::build(const MeshInput& in) {
    gpu::Device& device = *device_;
    GpuMesh mesh;
    mesh.topology = in.topology;
    mesh.source = in.source;
    mesh.points = static_cast<uint32_t>(in.points.values() / 3);
    mesh.faces = static_cast<uint32_t>(in.faceVertexCounts.size());
    const uint32_t indexCount = static_cast<uint32_t>(in.faceVertexIndices.size());
    const uint32_t holeCount = static_cast<uint32_t>(in.holeIndices.size());
    if (mesh.points == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': a mesh without points", in.source);
    }

    // Points.
    {
        const uint64_t words = std::max<uint64_t>((in.points.bytes.size() + 3) / 4, 1);
        auto source = deviceBuffer(device, words, 4, "mesh.points.words");
        if (!source) return std::move(source).error();
        LRT_TRY(source->write(device, 0, in.points.bytes.size(), in.points.bytes.data()));
        auto positions = deviceBuffer(device, mesh.points, 16, "mesh.positions");
        if (!positions) return std::move(positions).error();
        mesh.positions = std::move(*positions);
        gpu::CommandBatch batch(device);
        points_.dispatch(batch, {mesh.points, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(source->rhi());
            cursor["positions"].setBinding(mesh.positions.rhi());
            cursor["params"]["count"].setData(mesh.points);
            cursor["params"]["half"].setData(uint32_t{in.points.half ? 1u : 0u});
        });
        LRT_TRY(batch.submit(true));
    }

    // Topology: holes, counts, starts, triangles.
    auto counts = intBuffer(device, in.faceVertexCounts, "mesh.faceVertexCounts");
    if (!counts) return std::move(counts).error();
    auto indices = intBuffer(device, in.faceVertexIndices, "mesh.faceVertexIndices");
    if (!indices) return std::move(indices).error();
    auto holes = intBuffer(device, in.holeIndices, "mesh.holeIndices");
    if (!holes) return std::move(holes).error();
    auto holeFlags = deviceBuffer(device, mesh.faces, 4, "mesh.holeFlags");
    if (!holeFlags) return std::move(holeFlags).error();
    auto cornerCounts = deviceBuffer(device, mesh.faces, 4, "mesh.cornerCounts");
    if (!cornerCounts) return std::move(cornerCounts).error();
    auto triangleCounts = deviceBuffer(device, mesh.faces, 4, "mesh.triangleCounts");
    if (!triangleCounts) return std::move(triangleCounts).error();
    auto faceStarts = deviceBuffer(device, mesh.faces, 4, "mesh.faceStarts");
    if (!faceStarts) return std::move(faceStarts).error();
    auto triangleStarts = deviceBuffer(device, mesh.faces, 4, "mesh.triangleStarts");
    if (!triangleStarts) return std::move(triangleStarts).error();
    auto totals = deviceBuffer(device, 2, 4, "mesh.totals");
    if (!totals) return std::move(totals).error();
    auto cornerTotal = deviceBuffer(device, 1, 4, "mesh.cornerTotal");
    if (!cornerTotal) return std::move(cornerTotal).error();
    auto triangleTotal = deviceBuffer(device, 1, 4, "mesh.triangleTotal");
    if (!triangleTotal) return std::move(triangleTotal).error();
    const auto topologyParams = [&](rhi::ShaderCursor p) {
        p["faces"].setData(mesh.faces);
        p["indices"].setData(indexCount);
        p["points"].setData(mesh.points);
        p["holes"].setData(holeCount);
        p["flip"].setData(uint32_t{in.leftHanded ? 1u : 0u});
    };
    if (mesh.faces > 0) {
        gpu::CommandBatch batch(device);
        // Nothing writes a zero here: meshHoles marks hole faces with a 1 and
        // leaves every other face alone, so without this the flags start as
        // whatever the device last left in that memory. Metal handed back
        // zeros and CUDA did not, and a stale word reads as "this face is a
        // hole", which silently drops the face's triangles.
        batch.encoder()->clearBuffer(holeFlags->rhi(), 0, uint64_t{mesh.faces} * 4);
        batch.markDirty();
        if (holeCount > 0) {
            holes_.dispatch(batch, {holeCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["holeIndices"].setBinding(holes->rhi());
                cursor["holeFlags"].setBinding(holeFlags->rhi());
                topologyParams(cursor["params"]);
            });
        }
        faceCounts_.dispatch(batch, {mesh.faces, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["faceVertexCounts"].setBinding(counts->rhi());
            cursor["holeFlags"].setBinding(holeFlags->rhi());
            cursor["cornerCounts"].setBinding(cornerCounts->rhi());
            cursor["triangleCounts"].setBinding(triangleCounts->rhi());
            topologyParams(cursor["params"]);
        });
        LRT_TRY(prefix_.apply(batch, *cornerCounts, *faceStarts, *cornerTotal, mesh.faces));
        LRT_TRY(prefix_.apply(batch, *triangleCounts, *triangleStarts, *triangleTotal, mesh.faces));
        batch.encoder()->copyBuffer(totals->rhi(), 0, cornerTotal->rhi(), 0, 4);
        batch.encoder()->copyBuffer(totals->rhi(), 4, triangleTotal->rhi(), 0, 4);
        batch.markDirty();
        LRT_TRY(batch.submit(true));
        uint32_t read[2] = {0, 0};
        LRT_TRY(totals->read(device, 0, sizeof(read), read));
        mesh.corners = std::min(read[0], indexCount);
        mesh.triangles = read[1];
    }
    auto triangles = deviceBuffer(device, uint64_t{mesh.triangles} * 3, 4, "mesh.indices");
    if (!triangles) return std::move(triangles).error();
    auto triangleCorners = deviceBuffer(device, uint64_t{mesh.triangles} * 3, 4, "mesh.triangleCorners");
    if (!triangleCorners) return std::move(triangleCorners).error();
    auto triangleFaces = deviceBuffer(device, mesh.triangles, 4, "mesh.triangleFaces");
    if (!triangleFaces) return std::move(triangleFaces).error();
    mesh.indices = std::move(*triangles);
    mesh.triangleCorners = std::move(*triangleCorners);
    mesh.triangleFaces = std::move(*triangleFaces);
    if (mesh.triangles > 0) {
        gpu::CommandBatch batch(device);
        triangulate_.dispatch(batch, {mesh.faces, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["faceVertexIndices"].setBinding(indices->rhi());
            cursor["faceStarts"].setBinding(faceStarts->rhi());
            cursor["triangleCounts"].setBinding(triangleCounts->rhi());
            cursor["triangleStarts"].setBinding(triangleStarts->rhi());
            cursor["triangles"].setBinding(mesh.indices.rhi());
            cursor["triangleCorners"].setBinding(mesh.triangleCorners.rhi());
            cursor["triangleFaces"].setBinding(mesh.triangleFaces.rhi());
            topologyParams(cursor["params"]);
        });
        LRT_TRY(batch.submit(true));
    }

    // GeomSubsets: faces to subsets, then triangles to their faces' subsets.
    if (!in.subsets.empty() && mesh.triangles > 0) {
        std::vector<uint32_t> pairs;
        for (size_t k = 0; k < in.subsets.size(); ++k) {
            for (const int32_t face : in.subsets[k]) {
                pairs.push_back(static_cast<uint32_t>(face));
                pairs.push_back(static_cast<uint32_t>(k + 1));
            }
        }
        const uint32_t entries = static_cast<uint32_t>(pairs.size() / 2);
        if (pairs.empty()) {
            pairs = {0xFFFFFFFFu, 0u};
        }
        auto subsetFaces = gpu::Buffer::fromSpan<uint32_t>(device, pairs, "mesh.subsetFaces");
        if (!subsetFaces) return std::move(subsetFaces).error();
        auto faceSubsets = deviceBuffer(device, mesh.faces, 4, "mesh.faceSubsets");
        if (!faceSubsets) return std::move(faceSubsets).error();
        auto triangleSubsets = deviceBuffer(device, mesh.triangles, 4, "mesh.triangleSubsets");
        if (!triangleSubsets) return std::move(triangleSubsets).error();
        gpu::CommandBatch batch(device);
        subsetClear_.dispatch(batch, {mesh.faces, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["cleared"].setBinding(faceSubsets->rhi());
            cursor["params"]["count"].setData(mesh.faces);
            cursor["params"]["value"].setData(uint32_t{0});
        });
        if (entries > 0) {
            subsetScatter_.dispatch(batch, {entries, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["subsetFaces"].setBinding(subsetFaces->rhi());
                cursor["faceSubsets"].setBinding(faceSubsets->rhi());
                cursor["params"]["count"].setData(entries);
                cursor["params"]["faces"].setData(mesh.faces);
            });
        }
        subsetTriangles_.dispatch(batch, {mesh.triangles, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["faceSubsets"].setBinding(faceSubsets->rhi());
            cursor["triangleFaces"].setBinding(mesh.triangleFaces.rhi());
            cursor["triangleSubsets"].setBinding(triangleSubsets->rhi());
            cursor["params"]["count"].setData(mesh.triangles);
        });
        LRT_TRY(batch.submit(true));
        mesh.subsets = static_cast<uint32_t>(in.subsets.size());
        mesh.triangleSubsets = std::move(*triangleSubsets);
    }

    // Primvars, their indices resolved on the device.
    bool authoredNormals = false;
    for (const PrimvarInput& p : in.primvars) {
        const uint32_t components = std::clamp<uint32_t>(p.components, 1, 4);
        const uint64_t elements = p.values.values() / components;
        const uint32_t count = static_cast<uint32_t>(p.indices.empty() ? elements : p.indices.size());
        if (count == 0 || p.values.empty()) {
            continue;
        }
        GpuPrimvar primvar;
        primvar.name = p.name;
        primvar.interpolation = p.interpolation;
        primvar.components = components;
        primvar.count = count;
        const uint64_t words = std::max<uint64_t>((p.values.bytes.size() + 3) / 4, 1);
        auto source = deviceBuffer(device, words, 4, "mesh.primvar.words");
        if (!source) return std::move(source).error();
        LRT_TRY(source->write(device, 0, p.values.bytes.size(), p.values.bytes.data()));
        auto primvarIndices = intBuffer(device, p.indices, "mesh.primvar.indices");
        if (!primvarIndices) return std::move(primvarIndices).error();
        auto values = deviceBuffer(device, count, 16, "mesh.primvar");
        if (!values) return std::move(values).error();
        primvar.values = std::move(*values);
        gpu::CommandBatch batch(device);
        expand_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(source->rhi());
            cursor["indices"].setBinding(primvarIndices->rhi());
            cursor["expanded"].setBinding(primvar.values.rhi());
            rhi::ShaderCursor q = cursor["params"];
            q["count"].setData(count);
            q["components"].setData(components);
            q["kind"].setData(p.values.kind());
            q["indexed"].setData(uint32_t{p.indices.empty() ? 0u : 1u});
            q["values"].setData(static_cast<uint32_t>(elements));
        });
        LRT_TRY(batch.submit(true));
        authoredNormals = authoredNormals || p.name == "normals";
        mesh.primvars.push_back(std::move(primvar));
    }

    // Smooth normals: corners keyed by point, sorted, summed per point.
    if (in.smoothNormals && !authoredNormals && mesh.corners > 0) {
        gpu::SortBuffers sorting;
        const auto assign = [&](gpu::Buffer& into, const char* label) -> Result<void> {
            auto made = deviceBuffer(device, mesh.corners, 4, label);
            if (!made) return std::move(made).error();
            into = std::move(*made);
            return ok();
        };
        LRT_TRY(assign(sorting.keysLo, "mesh.normal.keys"));
        LRT_TRY(assign(sorting.values, "mesh.normal.corners"));
        LRT_TRY(assign(sorting.scratchKeysLo, "mesh.normal.keys2"));
        LRT_TRY(assign(sorting.scratchValues, "mesh.normal.corners2"));
        auto cornerFaces = deviceBuffer(device, mesh.corners, 4, "mesh.normal.cornerFaces");
        if (!cornerFaces) return std::move(cornerFaces).error();
        auto runStarts = deviceBuffer(device, mesh.points, 4, "mesh.normal.runStarts");
        if (!runStarts) return std::move(runStarts).error();
        auto normals = deviceBuffer(device, mesh.points, 16, "mesh.normals");
        if (!normals) return std::move(normals).error();
        const uint32_t keyBits = std::max<uint32_t>(std::bit_width(mesh.points), 1);
        const auto normalParams = [&](rhi::ShaderCursor p) {
            p["faces"].setData(mesh.faces);
            p["indices"].setData(indexCount);
            p["points"].setData(mesh.points);
            p["corners"].setData(mesh.corners);
            p["flip"].setData(uint32_t{in.leftHanded ? 1u : 0u});
        };
        gpu::CommandBatch batch(device);
        cornerKeys_.dispatch(batch, {mesh.corners, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["faceStarts"].setBinding(faceStarts->rhi());
            cursor["faceVertexIndices"].setBinding(indices->rhi());
            cursor["keys"].setBinding(sorting.keysLo.rhi());
            cursor["values"].setBinding(sorting.values.rhi());
            cursor["cornerFaces"].setBinding(cornerFaces->rhi());
            normalParams(cursor["params"]);
        });
        LRT_TRY(sort_.sort(batch, sorting, mesh.corners, keyBits));
        clearRuns_.dispatch(batch, {mesh.points, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["runStarts"].setBinding(runStarts->rhi());
            normalParams(cursor["params"]);
        });
        pointRuns_.dispatch(batch, {mesh.corners, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["keys"].setBinding(sorting.keysLo.rhi());
            cursor["runStarts"].setBinding(runStarts->rhi());
            normalParams(cursor["params"]);
        });
        pointNormals_.dispatch(batch, {mesh.points, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["faceVertexCounts"].setBinding(counts->rhi());
            cursor["faceVertexIndices"].setBinding(indices->rhi());
            cursor["faceStarts"].setBinding(faceStarts->rhi());
            cursor["positions"].setBinding(mesh.positions.rhi());
            cursor["keys"].setBinding(sorting.keysLo.rhi());
            cursor["values"].setBinding(sorting.values.rhi());
            cursor["cornerFaces"].setBinding(cornerFaces->rhi());
            cursor["runStarts"].setBinding(runStarts->rhi());
            cursor["normals"].setBinding(normals->rhi());
            normalParams(cursor["params"]);
        });
        LRT_TRY(batch.submit(true));
        GpuPrimvar primvar;
        primvar.name = "normals";
        primvar.interpolation = Interpolation::Vertex;
        primvar.components = 3;
        primvar.count = mesh.points;
        primvar.values = std::move(*normals);
        mesh.primvars.push_back(std::move(primvar));
    }

    // Bounds, on the device.
    {
        constexpr uint32_t kChunk = 4096;
        const uint32_t chunks = (mesh.points + kChunk - 1) / kChunk;
        auto extents = deviceBuffer(device, uint64_t{chunks} * 2, 16, "mesh.bounds.extents");
        if (!extents) return std::move(extents).error();
        auto result = deviceBuffer(device, 2, 16, "mesh.bounds");
        if (!result) return std::move(result).error();
        const auto params = [&](rhi::ShaderCursor cursor) {
            cursor["params"]["count"].setData(mesh.points);
            cursor["params"]["chunkSize"].setData(kChunk);
            cursor["params"]["chunkCount"].setData(chunks);
        };
        gpu::CommandBatch batch(device);
        boundsChunks_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(mesh.positions.rhi());
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
        mesh.bounds.min = {extent[0], extent[1], extent[2]};
        mesh.bounds.max = {extent[4], extent[5], extent[6]};
    }
    log::debug("{}: {} points, {} faces, {} triangles", mesh.source, mesh.points, mesh.faces, mesh.triangles);
    return mesh;
}

}   // namespace lrt::geom
