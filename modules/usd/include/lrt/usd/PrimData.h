// Copyright (c) 2026 lucabRTrender contributors.
//
// USD's arrays as they are, for the device to read. A prim's Sync keeps the
// VtValues it was handed (reference counted: no copy), and the commit uploads
// each array's bytes -- float or half -- to be interleaved and decoded on the
// GPU like any file's records. Nothing here loops over elements.
#pragma once

#include <array>
#include <optional>
#include <vector>
#include <string>

#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/path.h>

#include "lrt/render/Camera.h"

#include <pxr/base/vt/value.h>

#include "lrt/scene/GpuClouds.h"

namespace lrt::usd {

struct ParticleFieldArrays {
    pxr::VtValue positions;        ///< VtVec3fArray or VtVec3hArray
    pxr::VtValue orientations;     ///< VtQuatfArray or VtQuathArray; empty: identity
    pxr::VtValue scales;           ///< VtVec3fArray or VtVec3hArray; empty: unit
    pxr::VtValue opacities;        ///< VtFloatArray or VtHalfArray; empty: opaque
    int          shDegree = 0;
    pxr::VtValue shCoefficients;   ///< VtVec3fArray or VtVec3hArray, (degree+1)^2 per particle, DC first
    /// What a relit gaussian reflects with, one value each
    /// (`primvars:lrt:splat:metallic` and `:roughness`, LrtSplatLightingAPI).
    /// Empty for a capture, which was trained with its light already in it.
    pxr::VtValue metallic;         ///< VtFloatArray or VtHalfArray
    pxr::VtValue roughness;
    pxr::VtValue transmission;
};

/// A volume's field asset, as UsdVolOpenVDBAsset names it.
struct VolumeFieldAsset {
    std::string path;       ///< the .vdb, resolved
    std::string gridName;   ///< the grid in it; empty: the first
};

/// A volume as Sync reads it: which field is its medium, where it stands,
/// how it scatters.
struct VolumeArrays {
    pxr::SdfPath         field;       ///< the field prim (an openvdbAsset bprim)
    std::string          fieldName;   ///< the volume's name for it ("density")
    render::Mat4         objectToWorld = render::Mat4::identity();
    float                densityScale = 1.0F;
    std::array<float, 3> albedo{0.8F, 0.8F, 0.8F};
    float                g = 0.0F;
    bool                 visible = true;
};

struct PointsArrays {
    pxr::VtValue positions;   ///< VtVec3fArray or VtVec3hArray
    pxr::VtValue colours;     ///< displayColor: VtVec3fArray (one or per point) or GfVec3f; linear
};

/// A primvar as Hydra holds it.
struct PrimvarArrays {
    std::string        name;
    uint32_t           interpolation = 0;   ///< geom::Interpolation's values
    pxr::VtValue       values;
    pxr::VtArray<int>  indices;             ///< empty unless indexed
};

/// A numeric primvar's bytes (float, half or double; scalar or array) and
/// components per element. Empty for anything else.
[[nodiscard]] scene::FloatStream primvarStreamOf(const pxr::VtValue& value, uint32_t* components);

/// A mesh's topology and points as Hydra holds them (all reference counted).
/// A GeomSubset of a mesh (materialBind family): its faces and its material.
struct MeshSubset {
    pxr::VtArray<int> faces;
    pxr::SdfPath      material;
};

/// What skins a mesh, read from usdSkelImaging's ext computation prims (the
/// aggregator's and the computation's inputs), as VtValues: uploaded as
/// they are and skinned on the device (geom::Skinner).
struct SkinningArrays {
    pxr::VtValue restPoints;               ///< VtVec3fArray
    pxr::VtValue geomBindXform;            ///< GfMatrix4f or GfMatrix4d
    pxr::VtValue influences;               ///< VtVec2fArray (joint, weight)
    int          numInfluencesPerComponent = 0;
    bool         hasConstantInfluences = false;
    pxr::VtValue blendShapeOffsets;        ///< VtVec4fArray
    pxr::VtValue blendShapeOffsetRanges;   ///< VtVec2iArray, a point
    pxr::VtValue blendShapeWeights;        ///< VtFloatArray, a sub-shape
    pxr::VtValue skinningXforms;           ///< VtMatrix4fArray, a joint
    pxr::VtValue skinningDualQuats;        ///< VtVec4fArray, two a joint, or VtQuatfArray pairs
    pxr::VtValue skinningScaleXforms;      ///< VtMatrix3fArray, a joint (empty for none)
    pxr::VtValue skelLocalToWorld;         ///< GfMatrix4d
    pxr::VtValue primWorldToLocal;         ///< GfMatrix4d
    bool         dualQuaternion = false;
};

struct MeshArrays {
    pxr::VtValue           points;              ///< VtVec3fArray or VtVec3hArray
    pxr::VtArray<int>      faceVertexCounts;
    pxr::VtArray<int>      faceVertexIndices;
    pxr::VtArray<int>      holeIndices;
    pxr::VtArray<int>      invisibleFaces;   ///< Hydra's: kept in the topology, not drawn
    /// Subdivision: the scheme (none, bilinear, catmullClark, loop), the
    /// display style's refine level, and UsdGeomMesh's creases and corners.
    pxr::TfToken           scheme;
    int                    refineLevel = 0;
    pxr::VtArray<int>      creaseIndices, creaseLengths, cornerIndices;
    pxr::VtArray<float>    creaseSharpnesses, cornerSharpnesses;
    bool                   leftHanded = false;
    bool                   smoothNormals = true;
    /// Hydra marked the topology dirty: the mesh is a new one, not the last
    /// one deformed.
    bool                   topologyChanged = true;
    /// The points at the shutter's open and close, when they differ from
    /// `points`: empty otherwise.
    pxr::VtValue           pointsStart;
    pxr::VtValue           pointsEnd;
    double                 pointsTimeStart = 0.0;   ///< the samples' times, in frames about the frame
    double                 pointsTimeEnd = 0.0;
    /// The mesh is skinned: `points` are its rest points and these skin them.
    std::optional<SkinningArrays> skinning;
    std::vector<PrimvarArrays> primvars;
    std::vector<MeshSubset>    subsets;
};

/// UsdGeomBasisCurves as Hydra holds it: the engine lays a tube over each
/// span on the device and draws it as a mesh.
struct CurveArrays {
    pxr::VtValue           points;              ///< VtVec3fArray or VtVec3hArray
    pxr::VtArray<int>      curveVertexCounts;
    pxr::VtArray<int>      curveIndices;
    pxr::TfToken           type;                ///< linear | cubic
    pxr::TfToken           basis;               ///< bezier | bspline | catmullRom
    pxr::TfToken           wrap;                ///< nonperiodic | periodic | pinned
    pxr::VtValue           widths;              ///< VtFloatArray, or empty
    uint32_t               widthsInterpolation = 0;   ///< HdInterpolation's values
    bool                   topologyChanged = true;
    std::vector<PrimvarArrays> primvars;
};

/// An instancer's primvars as Hydra holds them.
/// An instancer's per-instance arrays and its own transform at one time.
struct InstancerSample {
    pxr::VtValue translations;   ///< VtVec3fArray / VtVec3dArray / VtVec3hArray
    pxr::VtValue rotations;      ///< VtQuathArray / VtQuatfArray / VtQuatdArray (ix, iy, iz, real)
    pxr::VtValue scales;         ///< VtVec3fArray / VtVec3dArray / VtVec3hArray
    pxr::VtValue transforms;     ///< VtMatrix4dArray / VtMatrix4fArray, row-major
    render::Mat4 instancerTransform = render::Mat4::identity();
};

struct InstancerArrays {
    pxr::VtValue translations;   ///< VtVec3fArray / VtVec3dArray / VtVec3hArray
    pxr::VtValue rotations;      ///< VtQuathArray / VtQuatfArray / VtQuatdArray (ix, iy, iz, real)
    pxr::VtValue scales;         ///< VtVec3fArray / VtVec3dArray / VtVec3hArray
    pxr::VtValue transforms;     ///< VtMatrix4dArray / VtMatrix4fArray, row-major
    render::Mat4 instancerTransform = render::Mat4::identity();
    /// Under a shutter, when anything of the instancer moves: the samples
    /// that bracket it, at their own times (as Hydra hands them). An array a
    /// sample lacks, or that did not move, is the frame's.
    std::optional<InstancerSample> start;
    std::optional<InstancerSample> end;
    double                         timeStart = 0.0;
    double                         timeEnd = 0.0;
};

/// One level of a prototype's instancing: the instancer, and which of its
/// elements the level below (the prototype, or a nested instancer) takes.
/// A prim's transform at the shutter's open and close, each only when it
/// differs from the frame's.
struct MeshTransforms {
    std::optional<render::Mat4> start;
    std::optional<render::Mat4> end;
    double                      timeStart = 0.0;   ///< the samples' times, in frames about the frame
    double                      timeEnd = 0.0;
};

struct InstancerLink {
    pxr::SdfPath      instancer;
    pxr::VtArray<int> indices;
};

/// A coordinate system bound to a prim (UsdShadeCoordSysAPI), by the name a
/// material refers to it by, with its transform to world -- resolved by the
/// coordSys prim hdsi makes under the target, not by parsing paths.
struct CoordSysBinding {
    std::string  name;
    render::Mat4 toWorld = render::Mat4::identity();
};

/// What a mesh looks like before materials: displayColor, displayOpacity.
struct MeshLook {
    std::array<float, 3> displayColor{0.18F, 0.18F, 0.18F};
    float                displayOpacity = 1.0F;
    bool                 doubleSided = false;
    pxr::SdfPath         material;   ///< the bound material; empty: displayColor
    /// What the light linking scene index resolved this prim's collections
    /// into: the categories a light's link is tested against. Empty: only an
    /// unlinked light reaches it.
    std::vector<pxr::TfToken> categories;
    std::vector<CoordSysBinding> coordSys;   ///< the prim's bound coordinate systems
};

/// The bytes of a float array a VtValue holds (float or half, any tuple
/// size), and how many elements it has. Empty for anything else.
[[nodiscard]] scene::FloatStream streamOf(const pxr::VtValue& value, size_t* elements = nullptr);

/// Streams over `arrays`, which must outlive the upload.
[[nodiscard]] scene::SplatStreams splatStreams(const ParticleFieldArrays& arrays, std::string source);
[[nodiscard]] scene::PointStreams pointStreams(const PointsArrays& arrays, std::string source);

}   // namespace lrt::usd
