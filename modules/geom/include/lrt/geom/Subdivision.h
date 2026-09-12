// Copyright (c) 2026 lucabRTrender contributors.
//
// Subdivision surfaces on the device: Catmull-Clark, Loop and bilinear, with
// semi-sharp creases and corners, boundaries as creases, face-varying
// channels refined over their own topology, and the vertices pushed onto
// the limit surface at the end. The host lays out each level's topology --
// which corners make which edges, what each vertex touches -- as tables of
// indices (bookkeeping), and kernels place every point and value
// (subdivision.slang). The result is what a MeshBuilder takes: positions on
// the device, faces as index lists, primvars refined alongside.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/geom/Mesh.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::geom {

enum class SubdivisionScheme : uint32_t { Bilinear = 0, CatmullClark = 1, Loop = 2 };

struct SubdivisionInput {
    std::string                   source;
    scene::FloatStream            points;            ///< xyz a point, float or half (or `devicePositions`)
    const gpu::Buffer*            devicePositions = nullptr;
    uint32_t                      devicePoints = 0;
    std::span<const int32_t>      faceVertexCounts;
    std::span<const int32_t>      faceVertexIndices;
    std::span<const int32_t>      holeIndices;
    SubdivisionScheme             scheme = SubdivisionScheme::CatmullClark;
    uint32_t                      levels = 1;
    bool                          limit = true;       ///< push the last level onto the limit surface
    /// UsdGeomMesh's creases: vertex chains of `creaseLengths[k]` vertices
    /// each, a sharpness per crease or per edge; and corners.
    std::span<const int32_t>      creaseIndices;
    std::span<const int32_t>      creaseLengths;
    std::span<const float>        creaseSharpnesses;
    std::span<const int32_t>      cornerIndices;
    std::span<const float>        cornerSharpnesses;
    std::span<const PrimvarInput> primvars;
};

/// A primvar after refinement: vertex and face-varying ones on the device,
/// one float4 an element; uniform ones as the coarse values with an index a
/// refined face; constant ones as they were.
struct RefinedPrimvar {
    std::string          name;
    Interpolation        interpolation = Interpolation::Constant;
    uint32_t             components = 1;
    gpu::Buffer          values;           ///< device: float4 an element
    uint32_t             count = 0;
    scene::FloatStream   hostValues;       ///< uniform, constant: the coarse stream
    std::vector<int32_t> indices;          ///< uniform: per refined face, its coarse face
};

struct Refined {
    gpu::Buffer                 positions;   ///< float4 a point
    uint32_t                    points = 0;
    uint32_t                    edges = 0;   ///< of the last level, for a count that must be exact
    std::vector<int32_t>        faceVertexCounts;
    std::vector<int32_t>        faceVertexIndices;
    std::vector<int32_t>        holeIndices;
    std::vector<RefinedPrimvar> primvars;

    /// The refined mesh as a builder takes it; `inputs` holds what the spans point at.
    struct AsInput {
        MeshInput                 mesh;
        std::vector<PrimvarInput> primvars;
    };
    [[nodiscard]] AsInput asMeshInput(const std::string& source, uint64_t topology) const;
};

class Subdivider {
public:
    [[nodiscard]] static Result<Subdivider> create(gpu::ShaderLibrary& library);

    [[nodiscard]] Result<Refined> refine(const SubdivisionInput& input);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel facePoints_, edgePoints_, vertexPoints_, limit_, gather_, points_, expand_;
};

}   // namespace lrt::geom
