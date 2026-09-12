// Copyright (c) 2026 lucabRTrender contributors.
//
// Basis curves on the device, as tubes: each span evaluated at a few
// parameters, a ring of vertices at half the width about each, the rings
// joined into quads. The tube is a GpuMesh like any other, so every route
// draws it, shades it and picks it as a mesh; the host lays out the rings'
// topology (a pattern) and a kernel places every vertex.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "lrt/core/Result.h"
#include "lrt/geom/Mesh.h"
#include "lrt/gpu/ComputeKernel.h"

namespace lrt::geom {

enum class CurveBasis : uint32_t { Linear = 0, Bezier = 1, BSpline = 2, CatmullRom = 3 };
enum class CurveWrap : uint32_t { Nonperiodic = 0, Periodic = 1 };
/// How `widths` map onto the curves (Hydra's interpolations for curves).
enum class WidthInterpolation : uint32_t { Constant = 0, Uniform = 1, Vertex = 2, Varying = 3 };

struct CurveInput {
    std::string                source;
    scene::FloatStream         points;             ///< xyz a control point, float or half
    std::span<const int32_t>   curveVertexCounts;  ///< control points a curve
    std::span<const int32_t>   curveIndices;       ///< optional: indices into points, else 0..n
    CurveBasis                 basis = CurveBasis::Linear;   ///< linear type: Linear; cubic: one of the three
    CurveWrap                  wrap = CurveWrap::Nonperiodic;
    std::span<const float>     widths;             ///< by `widthInterpolation`; empty: `width`
    WidthInterpolation         widthInterpolation = WidthInterpolation::Constant;
    float                      width = 0.01F;
    uint32_t                   sides = 8;          ///< vertices a ring
    uint32_t                   segments = 0;       ///< sub-segments a span; 0: 8 for cubic, 1 for linear
    uint64_t                   topology = 0;       ///< MeshInput::topology
    std::span<const PrimvarInput> primvars;        ///< constant or uniform (a curve) primvars, carried to the tube's faces
};

class CurveBuilder {
public:
    [[nodiscard]] static Result<CurveBuilder> create(gpu::ShaderLibrary& library);

    /// The tube mesh: `spans` and `curves` say how it was laid out.
    struct Built {
        GpuMesh  mesh;
        uint32_t curves = 0;
        uint32_t spans = 0;
    };
    [[nodiscard]] Result<Built> build(const CurveInput& input);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel tube_;
    gpu::ComputeKernel points_;   ///< the mesh builder's decode, control points to float4
    MeshBuilder        meshes_;
};

}   // namespace lrt::geom
