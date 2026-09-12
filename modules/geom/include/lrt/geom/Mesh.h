// Copyright (c) 2026 lucabRTrender contributors.
//
// Polygon meshes onto the device: points, triangles in Hydra's order, and
// smooth normals as Hydra computes them -- every step a kernel. The host hands
// over the arrays USD holds (spans over the VtArrays, uploaded as they are)
// and gets triangles back with what every primvar interpolation needs to find
// its value: the point (vertex, varying), the face-vertex corner (faceVarying)
// and the authored face (uniform).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/PrefixSum.h"
#include "lrt/gpu/algo/RadixSort.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::geom {

/// How a primvar's values map onto a mesh (Hydra's HdInterpolation for meshes).
enum class Interpolation : uint32_t { Constant = 0, Uniform = 1, Varying = 2, Vertex = 3, FaceVarying = 4 };

struct PrimvarInput {
    std::string              name;
    Interpolation            interpolation = Interpolation::Constant;
    uint32_t                 components = 1;   ///< 1..4 per element
    scene::FloatStream       values;           ///< float, half or double
    std::span<const int32_t> indices;          ///< empty unless indexed
};

/// A primvar on the device: one float4 per element (indices resolved).
struct GpuPrimvar {
    std::string   name;
    Interpolation interpolation = Interpolation::Constant;
    uint32_t      components = 1;
    uint32_t      count = 0;
    gpu::Buffer   values;
};

struct MeshInput {
    std::string                source;
    scene::FloatStream         points;              ///< xyz per point, float or half
    /// Positions already on the device, float4 a point (a Skinner's), taken
    /// instead of `points`; `devicePoints` says how many.
    const gpu::Buffer*         devicePositions = nullptr;
    uint32_t                   devicePoints = 0;
    std::span<const int32_t>   faceVertexCounts;
    std::span<const int32_t>   faceVertexIndices;
    std::span<const int32_t>   holeIndices;
    /// Faces kept in the topology but not drawn (Hydra's invisible faces):
    /// unlike a hole, an invisible face keeps its triangles and their
    /// numbering, so it can be shown again without a rebuild.
    std::span<const int32_t>   invisibleFaces;
    bool                       leftHanded = false;
    /// Smooth normals are computed when true (a subdivision scheme that is
    /// not "none" or "bilinear", as Hydra decides) and no "normals" primvar
    /// is given; they become the "normals" vertex primvar.
    bool                       smoothNormals = true;
    std::span<const PrimvarInput> primvars;
    /// GeomSubsets of one family (materialBind): each one's authored face indices.
    std::vector<std::span<const int32_t>> subsets;
    /// A key the caller keeps for the topology, 0 for none: a mesh built with
    /// the same key, counts and primvar layout as one already in a scene is
    /// taken as that mesh deformed, and replaces its positions in place.
    uint64_t                   topology = 0;
};

struct GpuMesh {
    std::string   source;
    uint32_t      points = 0;
    uint32_t      faces = 0;
    uint32_t      corners = 0;      ///< face-vertex indices the faces use
    uint32_t      triangles = 0;
    gpu::Buffer   positions;        ///< float4 per point
    gpu::Buffer   indices;          ///< uint, 3 per triangle: points
    gpu::Buffer   triangleCorners;  ///< uint, 3 per triangle: face-vertex indices
    gpu::Buffer   triangleFaces;    ///< uint per triangle: authored face
    gpu::Buffer   triangleHidden;   ///< uint per triangle: 1 where its face is invisible
    bool          hidden = false;   ///< any face invisible
    uint32_t      subsets = 0;      ///< GeomSubsets given
    gpu::Buffer   triangleSubsets;  ///< uint per triangle: 0, or k + 1 for the k-th subset (when subsets > 0)
    std::vector<GpuPrimvar> primvars;   ///< authored, and "normals" when computed
    scene::Bounds bounds;
    /// The caller's key for the mesh's topology (MeshInput::topology): two
    /// meshes of the same key and layout are one mesh deformed.
    uint64_t      topology = 0;

    [[nodiscard]] const GpuPrimvar* primvar(std::string_view name) const noexcept {
        for (const GpuPrimvar& p : primvars) {
            if (p.name == name) {
                return &p;
            }
        }
        return nullptr;
    }
};

class MeshBuilder {
public:
    [[nodiscard]] static Result<MeshBuilder> create(gpu::ShaderLibrary& library);

    [[nodiscard]] Result<GpuMesh> build(const MeshInput& input);

private:
    gpu::Device*       device_ = nullptr;
    gpu::PrefixSum     prefix_;
    gpu::RadixSort     sort_;
    gpu::ComputeKernel points_, holes_, faceCounts_, triangulate_;
    gpu::ComputeKernel cornerKeys_, clearRuns_, pointRuns_, pointNormals_;
    gpu::ComputeKernel boundsChunks_, boundsReduce_;
    gpu::ComputeKernel expand_;
    gpu::ComputeKernel subsetClear_, subsetScatter_, subsetTriangles_;
};

}   // namespace lrt::geom
