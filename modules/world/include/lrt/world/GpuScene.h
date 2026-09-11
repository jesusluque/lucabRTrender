// Copyright (c) 2026 lucabRTrender contributors.
//
// The scene on the device, as every technique reads it: meshes packed into
// shared pools (positions, triangle indices, corners, faces, normals) and one
// record per instance. Meshes are built once (lrt::geom) and shared by
// pointer; the pools are rebuilt -- GPU copies, no host data -- when the set
// of meshes changes, and the instance records every frame.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/geom/Mesh.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/render/Camera.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::world {

struct MeshInstance {
    std::shared_ptr<const geom::GpuMesh> mesh;
    render::Mat4                         objectToWorld = render::Mat4::identity();
    uint32_t                             primId = 0;       ///< the primId AOV
    uint32_t                             instanceId = 0;   ///< the instanceId AOV
    std::array<float, 3>                 displayColor{0.18F, 0.18F, 0.18F};
    float                                displayOpacity = 1.0F;
    bool                                 doubleSided = true;
};

/// Many instances of one mesh whose transforms are already on the device
/// (world::Instancing): records are written there too.
struct InstanceSet {
    std::shared_ptr<const geom::GpuMesh> mesh;
    gpu::Buffer                          chainRows;   ///< 3 float4 per instance, before `prototype`
    uint32_t                             count = 0;
    render::Mat4                         prototype = render::Mat4::identity();
    uint32_t                             primId = 0;
    std::array<float, 3>                 displayColor{0.18F, 0.18F, 0.18F};
    float                                displayOpacity = 1.0F;
    bool                                 doubleSided = true;
};

/// Consecutive instances of one mesh: one draw.
struct DrawRange {
    uint32_t mesh = 0;
    uint32_t firstInstance = 0;
    uint32_t instances = 0;
};

class GpuScene {
public:
    [[nodiscard]] static Result<GpuScene> create(gpu::ShaderLibrary& library);

    /// This frame's instances, seen from `projection`. Instances of the same
    /// mesh become consecutive records (one draw each run).
    [[nodiscard]] Result<void> update(std::span<const MeshInstance> instances, const render::Projection& projection,
                                      std::span<const InstanceSet> sets = {});

    [[nodiscard]] uint32_t instanceCount() const noexcept { return instanceCount_; }
    [[nodiscard]] uint32_t meshCount() const noexcept { return static_cast<uint32_t>(meshes_.size()); }
    [[nodiscard]] std::span<const DrawRange> draws() const noexcept { return draws_; }
    [[nodiscard]] const geom::GpuMesh& mesh(uint32_t index) const { return *meshes_[index]; }
    [[nodiscard]] uint32_t firstPoint(uint32_t mesh) const { return ranges_[mesh].firstPoint; }
    [[nodiscard]] uint32_t firstTriangle(uint32_t mesh) const { return ranges_[mesh].firstTriangle; }
    /// Changes whenever the pools are repacked (the mesh set changed).
    [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

    // Pools and records (shaders/lrt/world/scene_types.slang).
    [[nodiscard]] const gpu::Buffer& positions() const noexcept { return positions_; }
    [[nodiscard]] const gpu::Buffer& normals() const noexcept { return normals_; }
    [[nodiscard]] const gpu::Buffer& indices() const noexcept { return indices_; }
    [[nodiscard]] const gpu::Buffer& triangleCorners() const noexcept { return triangleCorners_; }
    [[nodiscard]] const gpu::Buffer& triangleFaces() const noexcept { return triangleFaces_; }
    [[nodiscard]] const gpu::Buffer& meshRecords() const noexcept { return meshRecords_; }
    [[nodiscard]] const gpu::Buffer& instanceRecords() const noexcept { return instanceRecords_; }

private:
    struct Range {
        uint32_t firstPoint = 0;
        uint32_t firstTriangle = 0;
    };
    [[nodiscard]] Result<void> repack();

    gpu::Device*                                       device_ = nullptr;
    gpu::ComputeKernel                                 records_;
    std::vector<std::shared_ptr<const geom::GpuMesh>>  meshes_;
    std::vector<Range>                                 ranges_;
    std::vector<DrawRange>                             draws_;
    uint32_t                                           instanceCount_ = 0;
    uint64_t                                           generation_ = 0;
    gpu::Buffer positions_, normals_, indices_, triangleCorners_, triangleFaces_, meshRecords_, instanceRecords_;
};

}   // namespace lrt::world
