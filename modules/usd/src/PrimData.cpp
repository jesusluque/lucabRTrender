// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/PrimData.h"

#include <algorithm>
#include <span>

#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>

namespace lrt::usd {
namespace {

template <typename Element>
scene::FloatStream bytesOf(const pxr::VtArray<Element>& array, bool half, size_t* elements) {
    if (elements != nullptr) {
        *elements = array.size();
    }
    return {std::as_bytes(std::span<const Element>(array.cdata(), array.size())), half};
}

}   // namespace

scene::FloatStream streamOf(const pxr::VtValue& value, size_t* elements) {
    using namespace pxr;
    if (elements != nullptr) {
        *elements = 0;
    }
    if (value.IsHolding<VtVec3fArray>()) return bytesOf(value.UncheckedGet<VtVec3fArray>(), false, elements);
    if (value.IsHolding<VtVec3hArray>()) return bytesOf(value.UncheckedGet<VtVec3hArray>(), true, elements);
    if (value.IsHolding<VtQuatfArray>()) return bytesOf(value.UncheckedGet<VtQuatfArray>(), false, elements);
    if (value.IsHolding<VtQuathArray>()) return bytesOf(value.UncheckedGet<VtQuathArray>(), true, elements);
    if (value.IsHolding<VtFloatArray>()) return bytesOf(value.UncheckedGet<VtFloatArray>(), false, elements);
    if (value.IsHolding<VtHalfArray>()) return bytesOf(value.UncheckedGet<VtHalfArray>(), true, elements);
    if (value.IsHolding<GfVec3f>()) {
        if (elements != nullptr) {
            *elements = 1;
        }
        const GfVec3f& v = value.UncheckedGet<GfVec3f>();
        return {std::as_bytes(std::span<const float>(v.data(), 3)), false};
    }
    return {};
}

scene::SplatStreams splatStreams(const ParticleFieldArrays& a, std::string source) {
    scene::SplatStreams s;
    s.source = std::move(source);
    size_t count = 0;
    s.positions = streamOf(a.positions, &count);
    s.count = static_cast<uint32_t>(count);
    s.rotations = streamOf(a.orientations);
    s.scales = streamOf(a.scales);
    s.opacities = streamOf(a.opacities);
    const int degree = std::clamp(a.shDegree, 0, 3);
    s.coefficients = static_cast<uint32_t>((degree + 1) * (degree + 1));
    s.sh = streamOf(a.shCoefficients);
    if (s.sh.values() < uint64_t{s.count} * s.coefficients * 3) {
        s.sh = {};   // fewer coefficients than the degree says: none, as before
    }
    return s;
}

scene::PointStreams pointStreams(const PointsArrays& a, std::string source) {
    scene::PointStreams s;
    s.source = std::move(source);
    size_t count = 0;
    s.positions = streamOf(a.positions, &count);
    s.count = static_cast<uint32_t>(count);
    s.colours = streamOf(a.colours);
    return s;
}

}   // namespace lrt::usd
