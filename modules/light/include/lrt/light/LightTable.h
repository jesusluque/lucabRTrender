// Copyright (c) 2026 lucabRTrender contributors.
//
// The scene's lights on the device: one record per light, in world space and
// in the units USD authored, since what a record becomes -- exposure, the
// blackbody of a colour temperature, the area a `normalize` divides by -- is
// derived in the shader (shaders/lrt/light/lights.slang).
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/RadixSort.h"

#include <memory>
#include <optional>
#include "lrt/io/Ies.h"
#include "lrt/render/Camera.h"

namespace lrt::gpu {
class ShaderLibrary;
class Device;
}

namespace lrt::light {

enum class LightKind : uint32_t {
    Distant = 0,
    Sphere = 1,
    Disk = 2,
    Rect = 3,
    Dome = 4,
    Cylinder = 5,   ///< along its own x axis, as UsdLux has it; emits outward
};

/// A light as USD authored it. Angles are radians; `lightToWorld` puts the
/// light where UsdLux has it, shining down its own -Z.
struct Light {
    LightKind    kind = LightKind::Distant;
    render::Mat4 lightToWorld = render::Mat4::identity();
    float        colour[3] = {1.0F, 1.0F, 1.0F};
    float        intensity = 1.0F;
    float        exposure = 0.0F;
    float        radius = 0.5F;        ///< sphere and disk
    float        width = 1.0F;         ///< rect
    float        height = 1.0F;        ///< rect
    float        angle = 0.0F;         ///< distant: the sun's angular diameter
    float        length = 1.0F;        ///< cylinder, along its x axis
    /// UsdLux shaping: an IES profile, read by io::readIes; angleScale as
    /// UsdLux defines it (positive divides theta, negative scales from
    /// 180, zero none); normalize divides by the profile's power so its mean
    /// intensity over the sphere is one.
    std::shared_ptr<const io::IesProfile> ies;
    std::string  iesFile;              ///< where the profile comes from (Hydra); the engine reads it into `ies`
    float        iesAngleScale = 0.0F;
    bool         iesNormalize = false;
    /// Instanced: `instanceCount` placements, 3 float4 rows each, before the
    /// light's own transform -- world::Instancing's rows, handed over as a
    /// buffer since this module sits below world. The table copies the
    /// record once per instance and a kernel places each copy.
    const gpu::Buffer* instanceRows = nullptr;
    uint32_t     instanceCount = 0;
    /// Under a shutter, when an instancer in the chain moves: the chain's
    /// rows at the samples bracketing it, and when those are. The copies are
    /// placed between them as `instanceRows` places them at the frame.
    const gpu::Buffer* instanceRowsStart = nullptr;
    const gpu::Buffer* instanceRowsEnd = nullptr;
    float        instanceTimeStart = 0.0F;
    float        instanceTimeEnd = 1.0F;
    /// Whether the path tracer places this light between shutter samples:
    /// the light itself moves, or the instancer chain above it does.
    [[nodiscard]] bool movesUnderShutter() const noexcept {
        return moves || (instanceRows != nullptr && instanceCount > 0 && instanceRowsStart != nullptr &&
                         instanceRowsEnd != nullptr);
    }
    /// Under a shutter, when the light moves: its transform at the samples
    /// bracketing it, and when those are (in the shutter's units). The
    /// record keeps `lightToWorld`, the frame's; the path tracer's samples
    /// place the light between these. Under an instancer, the copies take
    /// these as the prototype's own at the samples.
    bool         moves = false;
    render::Mat4 lightToWorldStart = render::Mat4::identity();
    render::Mat4 lightToWorldEnd = render::Mat4::identity();
    float        timeStart = 0.0F;
    float        timeEnd = 1.0F;
    float        temperature = 6500.0F;
    bool         enableTemperature = false;
    bool         normalize = false;
    bool         shadow = true;
    float        coneAngle = 0.0F;     ///< shaping: the half angle; 0 or 180 means none
    float        coneSoftness = 0.0F;
    /// Which category this light lights, and which casts its shadows: the
    /// bit a prim must have for the light to reach it. kLightUnlinked: every
    /// prim, which is what a light with no collection means.
    uint32_t     lightCategory = 0xFFFFFFFFU;
    uint32_t     shadowCategory = 0xFFFFFFFFU;
    /// A dome's lat-long image, resolved. Empty: the light is its colour.
    std::string  texture;
    /// The collections USD resolved into category names. Empty: unlinked,
    /// which reaches every prim. Whoever owns both lights and prims turns
    /// these into the bits below.
    std::string  lightLink;
    std::string  shadowLink;
    /// Filled in by whoever owns the texture store, before the table is set.
    uint32_t     textureId = 0xFFFFFFFFU;
    uint32_t     sampler = 0;
    /// The light group it belongs to (`lrt:lightGroup`, or RenderMan's
    /// `ri:light:lightGroup`); empty for none. Whoever renders the frame
    /// numbers the groups a frame asks for into `groupIndex`: 0 none, else
    /// 1 + the group's index, which a kernel accumulates the light's direct
    /// contribution under.
    std::string  group;
    uint32_t     groupIndex = 0;
};

/// What shaders/lrt/light/lights.slang reads.
struct LightRecord {
    uint32_t kind = 0;
    uint32_t flags = 0;
    float    sizeX = 0.0F;
    float    sizeY = 0.0F;
    float    colour[3] = {0.0F, 0.0F, 0.0F};
    float    exposure = 0.0F;
    float    temperature = 6500.0F;
    float    coneCos = -1.0F;
    float    coneSoftness = 0.0F;
    uint32_t texture = 0xFFFFFFFFU;   ///< a dome's image in the texture table; none: its colour alone
    uint32_t sampler = 0;
    uint32_t lightCategory = 0xFFFFFFFFU;    ///< the bit a prim needs for this light to light it
    uint32_t shadowCategory = 0xFFFFFFFFU;   ///< and for it to cast this light's shadow
    /// Its share of the frame's power, accumulated: what a sample searches to
    /// choose one light instead of visiting them all.
    float    cumulative = 0.0F;
    uint32_t ies = 0xFFFFFFFFU;       ///< its IES profile's row; none: no profile
    float    iesAngleScale = 0.0F;
    uint32_t pad0 = 0;
    uint32_t group = 0;   ///< 0: no light group; else 1 + its index among the frame's
    float    rows[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};   ///< light to world, rows of a 3x4
};

/// One IES profile on the device: where its angle lists and candela table
/// sit in the values buffer, and its power, which ies_prepare computes.
struct IesRecord {
    uint32_t verticalOffset = 0;
    uint32_t verticalCount = 0;
    uint32_t horizontalOffset = 0;
    uint32_t horizontalCount = 0;
    uint32_t candelaOffset = 0;
    uint32_t photometricType = 1;
    float    multiplier = 1.0F;
    float    power = 1.0F;   ///< mean intensity over the sphere, for normalize
};

/// LightRecord::flags
inline constexpr uint32_t kLightShadow = 1;
inline constexpr uint32_t kLightNormalize = 2;
inline constexpr uint32_t kLightTemperature = 4;
constexpr uint32_t kLightIesNormalize = 8;   ///< divide the IES intensity by the profile's power

/// A light with no collection: it reaches every prim.
inline constexpr uint32_t kLightUnlinked = 0xFFFFFFFFU;

class LightTable {
public:
    /// The table needs the library for the kernel that accumulates each
    /// light's share of the frame's power on the device.
    [[nodiscard]] static Result<LightTable> create(gpu::ShaderLibrary& library);

    /// The frame's lights, uploaded. The buffer is remade when it must grow.
    /// `sceneRadius` is how far the scene reaches, in its own units: a dome
    /// or a sun spreads its light over all of it, so its power, set against
    /// an area light's, is its radiance over the scene's cross-section.
    [[nodiscard]] Result<void> set(std::span<const Light> lights, float sceneRadius = 1.0F);

    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] bool     anyShadow() const noexcept { return shadows_; }
    /// Whether any of them is a dome, which a frame paints where it drew nothing.
    [[nodiscard]] bool     anyDome() const noexcept { return domes_; }
    [[nodiscard]] const gpu::Buffer& records() const noexcept { return records_; }

    /// `lights` and `lightCount`, by name; and the light BVH's `lightNodes`,
    /// `lightTreeNodes` and `lightUnboundedCount`.
    void bind(rhi::ShaderCursor cursor) const;
    /// Whether the frame's lights have a tree to choose from: any bounded
    /// light at all. A frame of only domes and suns chooses by power.
    [[nodiscard]] bool hasBvh() const noexcept { return treeNodes_ > 0; }
    /// The tree's nodes live in the IES values buffer, sixteen floats each
    /// from `nodeBase`: a kernel on Metal binds thirty-one buffers at most.
    [[nodiscard]] const gpu::Buffer& nodeValues() const noexcept { return iesValues_; }
    [[nodiscard]] uint32_t nodeBase() const noexcept { return nodeBase_; }
    [[nodiscard]] uint32_t treeNodes() const noexcept { return treeNodes_; }
    [[nodiscard]] uint32_t unboundedCount() const noexcept { return unbounded_; }

    /// The record a light becomes, for tests and for the table itself.
    [[nodiscard]] static LightRecord recordOf(const Light& light);

private:
    gpu::Device* device_ = nullptr;
    gpu::Buffer  records_;
    std::optional<gpu::ComputeKernel> prefix_;   ///< light_prefix: each light's cumulative share, on the device
    std::optional<gpu::ComputeKernel> iesPrepare_;   ///< ies_prepare: each profile's power
    std::optional<gpu::ComputeKernel> instancesMotion_;   ///< light_instances: a moving instanced light's samples
    std::optional<gpu::ComputeKernel> instances_;    ///< light_instances: an instanced light's placements
    gpu::Buffer  iesRecords_;
    gpu::Buffer  iesValues_;
    /// The light BVH (light_bvh.slang): built after the prefix, from the
    /// bounded lights, with the compute LBVH's own hierarchy and refit
    /// kernels; the unbounded lights follow the tree in the same buffer.
    std::optional<gpu::ComputeKernel> bvhLeaves_, bvhPack_, bvhParents_, bvhSettle_, bvhUnbounded_, bvhLeafIndices_;
    std::optional<gpu::ComputeKernel> hierarchy_, refit_, countNonzero_, boundsChunks_, boundsReduce_;
    std::optional<gpu::RadixSort>     sort_;
    uint32_t     nodeBase_ = 0;
    uint32_t     motionBase_ = 0;     ///< in iesValues: 26 floats a record (rows at both samples, their times); 0: none moves
    bool         anyMoves_ = false;
    uint32_t     treeNodes_ = 0;
    uint32_t     unbounded_ = 0;
    [[nodiscard]] Result<void> buildBvh(const std::vector<uint32_t>& bounded, const std::vector<uint32_t>& unbounded);
    uint32_t     iesCount_ = 0;
    uint32_t     count_ = 0;
    uint32_t     capacity_ = 0;
    bool         shadows_ = false;
    bool         domes_ = false;
};

}   // namespace lrt::light
