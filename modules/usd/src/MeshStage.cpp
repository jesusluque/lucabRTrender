// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/MeshStage.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/base/gf/interval.h>
#include <pxr/usd/usd/editContext.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdSkel/animMapper.h>
#include <pxr/usd/usdSkel/bakeSkinning.h>
#include <pxr/usd/usdSkel/binding.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/utils.h>

#include "lrt/core/Log.h"
#include "lrt/geom/Mesh.h"
#include "lrt/usd/PrimData.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace lrt::usd {
namespace {

/// One input of a shader, followed as far as it goes: a value, or the file a
/// texture reads.
struct Resolved {
    bool         hasValue = false;
    VtValue      value;
    StageTexture texture;
};

[[nodiscard]] bool isImageNode(const TfToken& id) {
    const std::string& name = id.GetString();
    return name.rfind("ND_image_", 0) == 0 || name.rfind("ND_tiledimage_", 0) == 0 ||
           name == "UsdUVTexture";
}

[[nodiscard]] bool isNormalMapNode(const TfToken& id) {
    return id.GetString().rfind("ND_normalmap", 0) == 0;
}

/// The file a texture node reads, with what its colour space says.
[[nodiscard]] StageTexture fileOf(const UsdShadeShader& shader) {
    StageTexture texture;
    const UsdShadeInput file = shader.GetInput(TfToken("file"));
    if (!file) {
        return texture;
    }
    SdfAssetPath asset;
    if (!file.Get(&asset)) {
        return texture;
    }
    // Resolved where the resolver could, authored otherwise: the same rule
    // Light.cpp and Volume.cpp follow for every asset this engine opens.
    texture.file = !asset.GetResolvedPath().empty() ? asset.GetResolvedPath() : asset.GetAssetPath();
    const std::string space = file.GetAttr().GetColorSpace().GetString();
    texture.srgb = space.find("srgb") != std::string::npos || space.find("sRGB") != std::string::npos;
    if (space.empty()) {
        // UsdUVTexture says it another way.
        TfToken source;
        if (const UsdShadeInput hint = shader.GetInput(TfToken("sourceColorSpace"))) {
            hint.Get(&source);
            texture.srgb = source == TfToken("sRGB");
        }
    }
    return texture;
}

/// Follows `input` through node graphs, interface inputs and a normal map to
/// whatever finally produces it. Depth-limited: a graph may be a cycle, and
/// a converter chain may be long, and neither is worth following forever --
/// this reads what a splat can carry, not what a shading point would.
[[nodiscard]] Resolved resolve(const UsdShadeInput& input, int depth = 0) {
    Resolved out;
    if (!input || depth > 8) {
        return out;
    }
    const UsdShadeAttributeVector producing = input.GetValueProducingAttributes();
    for (const UsdAttribute& attribute : producing) {
        if (UsdShadeUtils::GetType(attribute.GetName()) == UsdShadeAttributeType::Output) {
            const UsdShadeShader shader(attribute.GetPrim());
            if (!shader) {
                continue;
            }
            TfToken id;
            shader.GetShaderId(&id);
            if (isImageNode(id)) {
                out.texture = fileOf(shader);
                return out;
            }
            if (isNormalMapNode(id)) {
                // A normal map node stands between the file and the surface.
                // What we want is the file; what the node does to it -- the
                // tangent frame -- the conversion does for itself.
                return resolve(shader.GetInput(TfToken("in")), depth + 1);
            }
            // Something computed: a mix, a multiply, a noise. There is no
            // answer here without shading the point, so the caller's default
            // stands and the material says so in the log.
            continue;
        }
        const UsdShadeInput held(attribute);
        if (held && held.Get(&out.value)) {
            out.hasValue = !out.value.IsEmpty();
            if (out.hasValue) {
                return out;
            }
        }
    }
    // Unconnected, with its own value.
    if (!out.hasValue && input.Get(&out.value)) {
        out.hasValue = !out.value.IsEmpty();
    }
    return out;
}

void takeFloat(const Resolved& resolved, float& into) {
    if (!resolved.hasValue) {
        return;
    }
    if (resolved.value.IsHolding<float>()) {
        into = resolved.value.UncheckedGet<float>();
    } else if (resolved.value.IsHolding<double>()) {
        into = static_cast<float>(resolved.value.UncheckedGet<double>());
    }
}

void takeColour(const Resolved& resolved, std::array<float, 3>& into) {
    if (!resolved.hasValue) {
        return;
    }
    if (resolved.value.IsHolding<GfVec3f>()) {
        const GfVec3f& v = resolved.value.UncheckedGet<GfVec3f>();
        into = {v[0], v[1], v[2]};
    } else if (resolved.value.IsHolding<GfVec3d>()) {
        const GfVec3d& v = resolved.value.UncheckedGet<GfVec3d>();
        into = {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
    }
}

/// What a material says, as far as a gaussian can carry it.
///
/// Two vocabularies are read: MaterialX's `standard_surface`, which is what
/// every asset built from a .mtlx arrives as, and `UsdPreviewSurface`, which is
/// what a file written for the viewport uses. They name the same things
/// differently and nothing else in this engine has had to ask them directly --
/// every other route goes through Hydra, which hands over a network.
[[nodiscard]] StageMaterial materialOf(const UsdPrim& prim) {
    StageMaterial out;
    const UsdShadeMaterialBindingAPI binding(prim);
    const UsdShadeMaterial material = binding.ComputeBoundMaterial();
    if (!material) {
        return out;
    }
    out.path = material.GetPath().GetString();
    UsdShadeShader surface = material.ComputeSurfaceSource({TfToken("mtlx")});
    if (!surface) {
        surface = material.ComputeSurfaceSource();
    }
    if (!surface) {
        return out;
    }
    TfToken id;
    surface.GetShaderId(&id);
    const bool preview = id == TfToken("UsdPreviewSurface");

    const auto read = [&surface](const char* name) { return resolve(surface.GetInput(TfToken(name))); };
    const Resolved colour = read(preview ? "diffuseColor" : "base_color");
    takeColour(colour, out.baseColour);
    out.albedo = colour.texture;

    const Resolved metallic = read(preview ? "metallic" : "metalness");
    takeFloat(metallic, out.metallic);
    out.metallicMap = metallic.texture;

    const Resolved roughness = read(preview ? "roughness" : "specular_roughness");
    takeFloat(roughness, out.roughness);
    out.roughnessMap = roughness.texture;

    out.normal = read("normal").texture;

    if (preview) {
        // UsdPreviewSurface has no transmission. What it has is an opacity,
        // and a surface you can see through is one whose opacity is less than
        // one -- so that is read as transmission, which is the only place a
        // gaussian can put it.
        float opacity = 1.0F;
        takeFloat(read("opacity"), opacity);
        out.transmission = std::clamp(1.0F - opacity, 0.0F, 1.0F);
    } else {
        takeFloat(read("transmission"), out.transmission);
        takeColour(read("transmission_color"), out.transmissionColour);
    }
    return out;
}

/// An affine map's three rows, in the order a kernel applies them: row `r`
/// holds the coefficients of the r-th coordinate of the result. USD stores a
/// matrix for row vectors (`v' = v M`), so a row here is a column there.
[[nodiscard]] std::array<float, 12> rowsOf(const GfMatrix4d& m) {
    std::array<float, 12> rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rows[static_cast<size_t>(r) * 4 + static_cast<size_t>(c)] = static_cast<float>(m[c][r]);
        }
        rows[static_cast<size_t>(r) * 4 + 3] = static_cast<float>(m[3][r]);
    }
    return rows;
}

/// The map a normal takes: the inverse transpose of the linear part. In USD's
/// row-vector convention that is the inverse's own rows, which is why this is
/// three lines rather than a transpose and an inverse.
[[nodiscard]] std::array<float, 12> normalRowsOf(const GfMatrix4d& m) {
    const GfMatrix4d inverse = m.GetInverse();
    std::array<float, 12> rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rows[static_cast<size_t>(r) * 4 + static_cast<size_t>(c)] = static_cast<float>(inverse[r][c]);
        }
    }
    return rows;
}

}   // namespace

struct MeshStage::Impl {
    UsdStageRefPtr        stage;
    std::string           source;
    std::optional<double> posedAt;   ///< the time the skinning was baked for
};

/// Puts a skinned stage in the pose it holds at `time`.
///
/// A conversion reads `UsdGeomMesh` directly, and a skinned mesh's `points`
/// attribute does not animate: the deformation is the skeleton's, and USD
/// resolves it through UsdSkel. `UsdSkelBakeSkinning` writes the posed points
/// as time samples on the meshes themselves, so the reads below need to know
/// nothing about skinning -- which is what keeps this away from Hydra and from
/// a second implementation of UsdSkel.
///
/// It is baked into the **session layer** and for one instant only, so the
/// file on disk is untouched and the cost is one pose and not a range. A stage
/// with no SkelRoot in it comes back unchanged.
Result<void> poseStage(const UsdStageRefPtr& stage, double time) {
    UsdEditContext edit(stage, stage->GetSessionLayer());
    if (!UsdSkelBakeSkinning(stage->Traverse(), GfInterval(time, time))) {
        return Error::make(ErrorCode::InvalidArgument, "cannot pose the stage's skinning at time {}",
                           time);
    }
    return ok();
}

Result<MeshStage> MeshStage::open(const std::filesystem::path& path) {
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    if (!stage) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not a stage this can open", path.string());
    }
    MeshStage held;
    held.impl_ = std::make_unique<Impl>();
    held.impl_->stage = stage;
    held.impl_->source = path.string();
    return held;
}

MeshStage::MeshStage(MeshStage&&) noexcept = default;
MeshStage& MeshStage::operator=(MeshStage&&) noexcept = default;
MeshStage::~MeshStage() = default;

Result<std::vector<float>> MeshStage::skeletonTransforms(const std::string& skeleton,
                                                         const std::vector<double>& times) const {
    if (impl_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "no stage");
    }
    const UsdPrim prim = impl_->stage->GetPrimAtPath(SdfPath(skeleton));
    if (!prim || !prim.IsA<UsdSkelSkeleton>()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not a Skeleton on this stage", skeleton);
    }
    UsdSkelCache cache;
    const UsdSkelSkeletonQuery query = cache.GetSkelQuery(UsdSkelSkeleton(prim));
    if (!query) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': its skeleton cannot be queried", skeleton);
    }
    VtTokenArray joints;
    UsdSkelSkeleton(prim).GetJointsAttr().Get(&joints);
    const size_t count = joints.size();
    if (count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': a skeleton with no joints", skeleton);
    }
    std::vector<float> out(times.size() * count * 16, 0.0F);
    for (size_t frame = 0; frame < times.size(); ++frame) {
        VtMatrix4fArray xforms;
        if (!query.ComputeSkinningTransforms(&xforms, UsdTimeCode(times[frame])) ||
            xforms.size() != count) {
            return Error::make(ErrorCode::InvalidArgument, "'{}': no joint transforms at time {}",
                               skeleton, times[frame]);
        }
        float* held = out.data() + frame * count * 16;
        // A copy, not a computation: USD laid the matrices out and they go
        // over as they are.
        std::memcpy(held, xforms.data(), count * 16 * sizeof(float));
    }
    return out;
}

std::pair<double, double> MeshStage::timeRange() const {
    if (impl_ == nullptr) {
        return {0.0, 0.0};
    }
    return {impl_->stage->GetStartTimeCode(), impl_->stage->GetEndTimeCode()};
}

std::string MeshStage::source() const {
    return impl_ == nullptr ? std::string{} : impl_->source;
}

/// The joints a stage's meshes are carried by, resolved once for the whole
/// stage: one skinning query a skinnable prim, and the skeleton each is bound
/// to. Inherited bindings are UsdSkel's business, not ours.
struct SkelBindings {
    UsdSkelCache                                cache;
    std::map<SdfPath, UsdSkelSkinningQuery>     queries;
    std::map<SdfPath, UsdSkelSkeleton>          skeletons;
};

SkelBindings resolveSkinning(const UsdStageRefPtr& stage) {
    SkelBindings out;
    for (const UsdPrim& prim : stage->Traverse()) {
        if (!prim.IsA<UsdSkelRoot>()) {
            continue;
        }
        const UsdSkelRoot root(prim);
        if (!out.cache.Populate(root, UsdTraverseInstanceProxies())) {
            continue;
        }
        std::vector<UsdSkelBinding> bindings;
        if (!out.cache.ComputeSkelBindings(root, &bindings, UsdTraverseInstanceProxies())) {
            continue;
        }
        for (const UsdSkelBinding& binding : bindings) {
            for (const UsdSkelSkinningQuery& query : binding.GetSkinningTargets()) {
                const SdfPath at = query.GetPrim().GetPath();
                out.queries.emplace(at, query);
                out.skeletons.emplace(at, binding.GetSkeleton());
            }
        }
    }
    return out;
}

/// One mesh's influences, in the SKELETON'S joint order.
///
/// A mesh may name its own subset of the skeleton's joints (`skel:joints`),
/// and then the indices UsdSkel hands back are into that subset. The mapper
/// that USD keeps for it goes the other way -- skeleton order to the mesh's --
/// so it is run over the identity to get the mesh index's skeleton index, and
/// the influences are written in the skeleton's order. One cloud then has one
/// joint order whatever mixture of meshes it came from.
StageSkinning skinningOf(const UsdSkelSkinningQuery& query, const UsdSkelSkeleton& skeleton,
                         size_t points, double time) {
    StageSkinning out;
    if (!query.HasJointInfluences()) {
        return out;
    }
    VtIntArray   indices;
    VtFloatArray weights;
    if (!query.ComputeVaryingJointInfluences(points, &indices, &weights)) {
        return out;
    }
    const int perPoint = query.GetNumInfluencesPerComponent();
    if (perPoint <= 0 || indices.size() != weights.size() ||
        indices.size() != points * static_cast<size_t>(perPoint)) {
        return out;
    }
    VtTokenArray joints;
    skeleton.GetJointsAttr().Get(&joints);

    // The mesh's joint index to the skeleton's.
    std::vector<int> toSkeleton;
    if (const UsdSkelAnimMapperRefPtr& mapper = query.GetJointMapper()) {
        VtIntArray identity(joints.size());
        for (size_t k = 0; k < joints.size(); ++k) {
            identity[k] = static_cast<int>(k);
        }
        VtIntArray mapped;
        if (mapper->Remap(identity, &mapped)) {
            toSkeleton.assign(mapped.begin(), mapped.end());
        }
    }

    out.bound = true;
    out.skeleton = skeleton.GetPrim().GetPath().GetString();
    out.joints.reserve(joints.size());
    for (const TfToken& joint : joints) {
        out.joints.push_back(joint.GetString());
    }
    out.perPoint = static_cast<uint32_t>(perPoint);
    out.influences.resize(indices.size() * 2);
    for (size_t k = 0; k < indices.size(); ++k) {
        int joint = indices[k];
        if (!toSkeleton.empty()) {
            joint = joint >= 0 && static_cast<size_t>(joint) < toSkeleton.size() ? toSkeleton[joint] : 0;
        }
        out.influences[k * 2] = static_cast<float>(std::max(joint, 0));
        out.influences[k * 2 + 1] = weights[k];
    }
    const GfMatrix4d bind = query.GetGeomBindTransform(UsdTimeCode(time));
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            out.geomBindTransform[static_cast<size_t>(row) * 4 + static_cast<size_t>(column)] =
                static_cast<float>(bind[row][column]);
        }
    }
    out.dualQuaternion = query.GetSkinningMethod() == UsdSkelTokens->dualQuaternion;
    return out;
}

Result<std::vector<StageMesh>> MeshStage::read(geom::MeshBuilder& builder, const MeshStageOptions& options) {
    if (impl_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "no stage");
    }
    const SdfPath under = options.prim.empty() ? SdfPath::AbsoluteRootPath() : SdfPath(options.prim);
    if (!options.prim.empty() && !under.IsAbsolutePath()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not an absolute prim path", options.prim);
    }
    const UsdTimeCode at(options.time);
    // POSED, OR CARRIED, AND NEVER BOTH. A cloud that keeps its joints is
    // built in the bind pose, because that is where the skeleton's transforms
    // expect to find it; posing the stage first would skin it twice.
    if (!options.skinned && (!impl_->posedAt.has_value() || *impl_->posedAt != options.time)) {
        LRT_TRY(poseStage(impl_->stage, options.time));
        impl_->posedAt = options.time;
    }
    const SkelBindings bindings = options.skinned ? resolveSkinning(impl_->stage) : SkelBindings{};
    UsdGeomXformCache transforms(at);
    std::vector<StageMesh> meshes;
    for (const UsdPrim& prim : impl_->stage->Traverse()) {
        if (!prim.IsA<UsdGeomMesh>() || !prim.GetPath().HasPrefix(under)) {
            continue;
        }
        const UsdGeomMesh mesh(prim);
        // Held for as long as the build takes: every stream below is a span
        // over one of these arrays.
        VtVec3fArray points;
        VtIntArray   counts;
        VtIntArray   indices;
        VtIntArray   holes;
        mesh.GetPointsAttr().Get(&points, at);
        mesh.GetFaceVertexCountsAttr().Get(&counts, at);
        mesh.GetFaceVertexIndicesAttr().Get(&indices, at);
        mesh.GetHoleIndicesAttr().Get(&holes, at);
        if (points.empty() || counts.empty() || indices.empty()) {
            lrt::log::info("mesh2splat: '{}' has no geometry, skipped", prim.GetPath().GetString());
            continue;
        }
        TfToken orientation;
        mesh.GetOrientationAttr().Get(&orientation);
        TfToken scheme;
        mesh.GetSubdivisionSchemeAttr().Get(&scheme);

        VtVec3fArray normals;
        TfToken      normalsInterpolation = UsdGeomTokens->vertex;
        const bool   hasNormals = mesh.GetNormalsAttr().Get(&normals, at) && !normals.empty();
        if (hasNormals) {
            normalsInterpolation = mesh.GetNormalsInterpolation();
        }
        const UsdGeomPrimvarsAPI primvars(prim);
        VtVec2fArray uvs;
        VtIntArray   uvIndices;
        TfToken      uvInterpolation = UsdGeomTokens->faceVarying;
        bool         hasUvs = false;
        for (const char* name : {"st", "st0", "uv", "UVMap"}) {
            const UsdGeomPrimvar primvar = primvars.GetPrimvar(TfToken(name));
            if (primvar && primvar.Get(&uvs, at) && !uvs.empty()) {
                uvInterpolation = primvar.GetInterpolation();
                primvar.GetIndices(&uvIndices);
                hasUvs = true;
                break;
            }
        }
        if (!hasUvs && !options.withoutTexcoords) {
            lrt::log::info("mesh2splat: '{}' has no texture coordinates, skipped",
                           prim.GetPath().GetString());
            continue;
        }

        const auto interpolationOf = [](const TfToken& token) {
            if (token == UsdGeomTokens->constant) return geom::Interpolation::Constant;
            if (token == UsdGeomTokens->uniform) return geom::Interpolation::Uniform;
            if (token == UsdGeomTokens->varying) return geom::Interpolation::Varying;
            if (token == UsdGeomTokens->faceVarying) return geom::Interpolation::FaceVarying;
            return geom::Interpolation::Vertex;
        };

        std::vector<geom::PrimvarInput> inputs;
        if (hasNormals) {
            geom::PrimvarInput primvar;
            primvar.name = "normals";
            primvar.interpolation = interpolationOf(normalsInterpolation);
            primvar.components = 3;
            primvar.values = {std::as_bytes(std::span<const GfVec3f>(normals.cdata(), normals.size())), false};
            inputs.push_back(std::move(primvar));
        }
        if (hasUvs) {
            geom::PrimvarInput primvar;
            primvar.name = "st";
            primvar.interpolation = interpolationOf(uvInterpolation);
            primvar.components = 2;
            primvar.values = {std::as_bytes(std::span<const GfVec2f>(uvs.cdata(), uvs.size())), false};
            primvar.indices = std::span<const int32_t>(uvIndices.cdata(), uvIndices.size());
            inputs.push_back(std::move(primvar));
        }

        geom::MeshInput input;
        input.source = prim.GetPath().GetString();
        input.points = {std::as_bytes(std::span<const GfVec3f>(points.cdata(), points.size())), false};
        input.faceVertexCounts = std::span<const int32_t>(counts.cdata(), counts.size());
        input.faceVertexIndices = std::span<const int32_t>(indices.cdata(), indices.size());
        input.holeIndices = std::span<const int32_t>(holes.cdata(), holes.size());
        input.leftHanded = orientation == UsdGeomTokens->leftHanded;
        // Hydra's rule, and `geom::MeshBuilder`'s: a mesh that is subdivided
        // gets smooth normals, one that is not keeps its facets.
        input.smoothNormals =
            !hasNormals && scheme != UsdGeomTokens->none && scheme != UsdGeomTokens->bilinear;
        input.primvars = inputs;

        auto built = builder.build(input);
        if (!built) {
            return std::move(built).error();
        }
        StageMesh out;
        out.path = input.source;
        out.mesh = std::move(*built);
        const GfMatrix4d toWorld = transforms.GetLocalToWorldTransform(prim);
        out.toWorld = rowsOf(toWorld);
        out.normalToWorld = normalRowsOf(toWorld);
        out.material = materialOf(prim);
        if (options.skinned) {
            const auto query = bindings.queries.find(prim.GetPath());
            const auto skeleton = bindings.skeletons.find(prim.GetPath());
            if (query != bindings.queries.end() && skeleton != bindings.skeletons.end()) {
                out.skinning = skinningOf(query->second, skeleton->second, points.size(), options.time);
            }
        }
        meshes.push_back(std::move(out));
    }
    if (meshes.empty()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no meshes to convert", impl_->source);
    }
    return meshes;
}

}   // namespace lrt::usd
