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

struct MeshInput {
    std::string                source;
    scene::FloatStream         points;              ///< xyz per point, float or half
    std::span<const int32_t>   faceVertexCounts;
    std::span<const int32_t>   faceVertexIndices;
    std::span<const int32_t>   holeIndices;
    bool                       leftHanded = false;
    /// Smooth normals are computed when true (no authored normals, and a
    /// subdivision scheme that is not "none" or "bilinear", as Hydra decides).
    bool                       smoothNormals = true;
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
    gpu::Buffer   normals;          ///< float4 per point when computed; invalid otherwise
    scene::Bounds bounds;
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
};

}   // namespace lrt::geom
