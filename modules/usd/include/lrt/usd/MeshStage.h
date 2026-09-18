// Copyright (c) 2026 lucabRTrender contributors.
//
// The meshes of a stage, read without Hydra.
//
// WHY NOT HYDRA
//
// Every other route into this engine goes through Hydra, and should: a render
// wants instancing, visibility, purposes, time samples and a render index that
// knows what changed. A conversion wants none of that. It wants the meshes a
// file holds, once, with their world transforms and what their materials say
// -- and it wants them from a process that is not rendering anything, so there
// is no render index to ask.
//
// So this walks the stage itself: `UsdGeomMesh` prims, `UsdGeomXformCache` for
// the transforms, `UsdShadeMaterialBindingAPI` for the material, and the
// arrays handed straight to `geom::MeshBuilder`, which triangulates them on
// the device in `HdMeshUtil`'s order -- the same triangles a render would have
// drawn, from the same kernel.
//
// What a material says is narrowed here to what a gaussian can carry: a
// colour, a metallic, a roughness, a normal, a transmission, and the files
// those come from. A MaterialX graph is not evaluated -- that is
// `material::MaterialCompiler`'s work and it needs a shading point -- so a
// value that is computed rather than authored comes back as the texture it
// is read from, or as the default.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <utility>
#include <memory>
#include <string>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/geom/Mesh.h"

namespace lrt::geom {
class MeshBuilder;
}

namespace lrt::usd {

/// A file a material names, and whether its values are sRGB encoded (which is
/// what USD's `colorSpace` says, and what MaterialX's `colorspace` attribute
/// says for these assets).
struct StageTexture {
    std::string file;
    bool        srgb = false;

    [[nodiscard]] bool empty() const noexcept { return file.empty(); }
};

/// What one material says, in the words a splat understands.
struct StageMaterial {
    std::string            path;   ///< the prim, for messages
    std::array<float, 3>   baseColour{1.0F, 1.0F, 1.0F};
    float                  metallic = 0.0F;
    float                  roughness = 0.5F;
    /// Ours, not mesh2splat's: how much of what stands behind the surface
    /// comes through it, and the colour it takes doing so.
    float                  transmission = 0.0F;
    std::array<float, 3>   transmissionColour{1.0F, 1.0F, 1.0F};
    StageTexture           albedo;
    StageTexture           normal;
    StageTexture           metallicMap;
    StageTexture           roughnessMap;
};

/// WHAT CARRIES A MESH WHEN ITS SKELETON MOVES.
///
/// The joints each of its points is held by and how much, in the skeleton's
/// own joint order, and the transform out of the mesh's space into the bind
/// space those joints are measured from. It is the same thing
/// `geom::SkinningInput` takes, read here out of USD rather than out of
/// Hydra's ext computation, so that a conversion can give each gaussian the
/// influences of the triangle it stands on.
struct StageSkinning {
    bool                  bound = false;
    /// The Skeleton prim's path: what the cloud is written against.
    std::string           skeleton;
    /// Its joints, in order, which is the order the indices below are in.
    std::vector<std::string> joints;
    uint32_t              perPoint = 0;   ///< influences a point
    /// `(joint, weight)`, `perPoint` of them a point, point major -- the
    /// layout `geom::SkinningInput::influences` reads.
    std::vector<float>    influences;
    /// Mesh space to bind space, row major, four rows of four.
    std::array<float, 16> geomBindTransform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    /// Dual quaternion rather than linear blend, if the mesh asked for it.
    bool                  dualQuaternion = false;
};

/// One mesh of the stage, already on the device.
struct StageMesh {
    std::string           path;
    geom::GpuMesh         mesh;
    /// Object to world, row major: three rows of four, the fourth implied.
    std::array<float, 12> toWorld{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    /// Its inverse transpose, for the normals; the same twelve numbers.
    std::array<float, 12> normalToWorld{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                                        0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    StageMaterial         material;
    /// Filled when `MeshStageOptions::skinned` asked for it and the mesh is
    /// bound to a skeleton.
    StageSkinning         skinning;
};

struct MeshStageOptions {
    /// A prim path: only meshes at or under it are read. Empty is the lot.
    std::string prim;
    /// Meshes with no `st` primvar are read anyway; their gaussians then take
    /// the triangle's own size rather than a texel's. False leaves them out.
    bool        withoutTexcoords = true;
    /// The USD time code the geometry is read at: the pose a conversion turns
    /// into gaussians. A skinned stage is posed for it first, into a session
    /// layer, so the file on disk is not touched; a stage with no animation in
    /// it reads exactly as it did, because an attribute with no time samples
    /// answers with its default whatever time is asked for.
    double      time = 0.0;
    /// Read the **bind** pose and each point's joint influences, rather than
    /// the pose at `time`. What a cloud that is to be carried by a skeleton
    /// needs: its gaussians stand where the skeleton's transforms expect
    /// them, and posing the stage first would apply the skinning twice.
    bool        skinned = false;
};

/// Opens `path` and reads its meshes. The stage stays open for as long as this
/// object does, because the arrays handed to the builder are spans over it.
class MeshStage {
public:
    [[nodiscard]] static Result<MeshStage> open(const std::filesystem::path& path);

    MeshStage(MeshStage&&) noexcept;
    MeshStage& operator=(MeshStage&&) noexcept;
    ~MeshStage();

    /// Builds every mesh the options ask for. The order is the stage's.
    [[nodiscard]] Result<std::vector<StageMesh>> read(geom::MeshBuilder& builder,
                                                      const MeshStageOptions& options = {});

    /// The joints of `skeleton` at each of `times`, in the skeleton's own
    /// order: `times.size() * joints * 16` floats, row major as USD holds
    /// them. What a cloud that carries its rig writes, and the only thing
    /// about such a cloud that changes from one frame to the next.
    [[nodiscard]] Result<std::vector<float>> skeletonTransforms(const std::string& skeleton,
                                                                const std::vector<double>& times) const;

    /// The stage's own time range, for a conversion that was given none.
    [[nodiscard]] std::pair<double, double> timeRange() const;

    /// The layers the stage was opened from, for a message.
    [[nodiscard]] std::string source() const;

    struct Impl;

private:
    MeshStage() = default;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::usd
