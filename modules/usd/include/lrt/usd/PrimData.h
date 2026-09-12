// Copyright (c) 2026 lucabRTrender contributors.
//
// USD's arrays as they are, for the device to read. A prim's Sync keeps the
// VtValues it was handed (reference counted: no copy), and the commit uploads
// each array's bytes -- float or half -- to be interleaved and decoded on the
// GPU like any file's records. Nothing here loops over elements.
#pragma once

#include <array>
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

struct MeshArrays {
    pxr::VtValue           points;              ///< VtVec3fArray or VtVec3hArray
    pxr::VtArray<int>      faceVertexCounts;
    pxr::VtArray<int>      faceVertexIndices;
    pxr::VtArray<int>      holeIndices;
    bool                   leftHanded = false;
    bool                   smoothNormals = true;
    std::vector<PrimvarArrays> primvars;
    std::vector<MeshSubset>    subsets;
};

/// An instancer's primvars as Hydra holds them.
struct InstancerArrays {
    pxr::VtValue translations;   ///< VtVec3fArray / VtVec3dArray / VtVec3hArray
    pxr::VtValue rotations;      ///< VtQuathArray / VtQuatfArray / VtQuatdArray (ix, iy, iz, real)
    pxr::VtValue scales;         ///< VtVec3fArray / VtVec3dArray / VtVec3hArray
    pxr::VtValue transforms;     ///< VtMatrix4dArray / VtMatrix4fArray, row-major
    render::Mat4 instancerTransform = render::Mat4::identity();
};

/// One level of a prototype's instancing: the instancer, and which of its
/// elements the level below (the prototype, or a nested instancer) takes.
struct InstancerLink {
    pxr::SdfPath      instancer;
    pxr::VtArray<int> indices;
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
};

/// The bytes of a float array a VtValue holds (float or half, any tuple
/// size), and how many elements it has. Empty for anything else.
[[nodiscard]] scene::FloatStream streamOf(const pxr::VtValue& value, size_t* elements = nullptr);

/// Streams over `arrays`, which must outlive the upload.
[[nodiscard]] scene::SplatStreams splatStreams(const ParticleFieldArrays& arrays, std::string source);
[[nodiscard]] scene::PointStreams pointStreams(const PointsArrays& arrays, std::string source);

}   // namespace lrt::usd
