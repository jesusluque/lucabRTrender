// Copyright (c) 2026 lucabRTrender contributors.
//
// USD's arrays as they are, for the device to read. A prim's Sync keeps the
// VtValues it was handed (reference counted: no copy), and the commit uploads
// each array's bytes -- float or half -- to be interleaved and decoded on the
// GPU like any file's records. Nothing here loops over elements.
#pragma once

#include <string>

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

/// The bytes of a float array a VtValue holds (float or half, any tuple
/// size), and how many elements it has. Empty for anything else.
[[nodiscard]] scene::FloatStream streamOf(const pxr::VtValue& value, size_t* elements = nullptr);

/// Streams over `arrays`, which must outlive the upload.
[[nodiscard]] scene::SplatStreams splatStreams(const ParticleFieldArrays& arrays, std::string source);
[[nodiscard]] scene::PointStreams pointStreams(const PointsArrays& arrays, std::string source);

}   // namespace lrt::usd
