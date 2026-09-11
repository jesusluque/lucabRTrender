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
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
    uint32_t                             material = 0;     ///< its row in the frame's material records; 0 none
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
    uint32_t                             material = 0;
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
    /// The world box around every instance's mesh box, folded on the device;
    /// nothing when the scene is empty.
    [[nodiscard]] Result<std::optional<scene::Bounds>> worldBounds() const;
    [[nodiscard]] uint32_t meshCount() const noexcept { return static_cast<uint32_t>(meshes_.size()); }
    [[nodiscard]] std::span<const DrawRange> draws() const noexcept { return draws_; }
    [[nodiscard]] const geom::GpuMesh& mesh(uint32_t index) const { return *meshes_[index]; }
    [[nodiscard]] uint32_t firstPoint(uint32_t mesh) const { return ranges_[mesh].firstPoint; }
    [[nodiscard]] uint32_t firstTriangle(uint32_t mesh) const { return ranges_[mesh].firstTriangle; }
    /// Changes whenever the pools are repacked (the mesh set changed).
    [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

    // Pools and records (shaders/lrt/world/scene_types.slang).
    [[nodiscard]] const gpu::Buffer& positions() const noexcept { return positions_; }
    [[nodiscard]] const gpu::Buffer& primvarValues() const noexcept { return primvarValues_; }
    [[nodiscard]] const gpu::Buffer& primvarRecords() const noexcept { return primvarRecords_; }
    [[nodiscard]] const gpu::Buffer& primvarSlots() const noexcept { return primvarSlots_; }

    /// The primvar slots after the fixed four (displayColor, displayOpacity,
    /// normals, st): names a frame looks up by slot (AOVs, materials).
    void setExtraPrimvarSlots(std::vector<std::string> names);
    [[nodiscard]] static constexpr uint32_t fixedSlots() noexcept { return 4; }
    /// The slot a primvar name has, or ~0.
    [[nodiscard]] uint32_t slotOf(std::string_view name) const noexcept;
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
    gpu::ComputeKernel                                 worldBoxes_, boundsChunks_, boundsReduce_;
    std::vector<std::shared_ptr<const geom::GpuMesh>>  meshes_;
    std::vector<Range>                                 ranges_;
    std::vector<DrawRange>                             draws_;
    uint32_t                                           instanceCount_ = 0;
    uint64_t                                           generation_ = 0;
    [[nodiscard]] Result<void> writeSlots();
    std::vector<std::string>                           slotNames_{"displayColor", "displayOpacity", "normals", "st"};
    bool                                               slotsDirty_ = true;
    std::vector<uint32_t>                              firstPrimvar_;   ///< per mesh: its first record
    std::vector<uint32_t>                              meshRecordWords_;   ///< the mesh records, as uploaded
    gpu::Buffer positions_, indices_, triangleCorners_, triangleFaces_, meshRecords_, instanceRecords_;
    gpu::Buffer primvarValues_, primvarRecords_, primvarSlots_;
    gpu::Buffer setRows_, setRecords_;
    /// The chains pooled in setRows_, in order. Held, so that a chain made
    /// later cannot take a pooled one's address.
    std::vector<std::pair<gpu::Buffer, uint32_t>>      setLayout_;
};

}   // namespace lrt::world
