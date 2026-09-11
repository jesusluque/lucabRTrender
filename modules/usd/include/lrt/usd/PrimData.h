// Copyright (c) 2026 lucabRTrender contributors.
//
// USD's arrays into the records the GPU decodes. Arranging, not decoding:
// ParticleField stores linear opacity and scale and raw SH coefficients, and
// the encoding descriptor says so; the numbers are turned into splats on the
// GPU like any file's.
#pragma once

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/quatf.h>

#include "lrt/io/RawSplats.h"

namespace lrt::usd {

struct ParticleFieldArrays {
    pxr::VtArray<pxr::GfVec3f> positions;
    pxr::VtArray<pxr::GfQuatf> orientations;   ///< may be empty: identity
    pxr::VtArray<pxr::GfVec3f> scales;         ///< may be empty: unit
    pxr::VtArray<float>        opacities;      ///< may be empty: opaque
    int                        shDegree = 0;
    pxr::VtArray<pxr::GfVec3f> shCoefficients; ///< (degree+1)^2 per particle, DC first; may be empty
};

[[nodiscard]] io::RawSplats rawSplatsFrom(const ParticleFieldArrays& arrays, std::string source);

struct PointsArrays {
    pxr::VtArray<pxr::GfVec3f> positions;
    pxr::VtArray<pxr::GfVec3f> colours;   ///< displayColor: one (constant) or per point; linear
};

[[nodiscard]] io::RawPoints rawPointsFrom(const PointsArrays& arrays, std::string source);

}   // namespace lrt::usd
