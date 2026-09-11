// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/PrimData.h"

#include <algorithm>

namespace lrt::usd {

io::RawSplats rawSplatsFrom(const ParticleFieldArrays& a, std::string source) {
    io::RawSplats raw;
    raw.source = std::move(source);
    const size_t n = a.positions.size();
    const bool haveRotation = a.orientations.size() >= n;
    const bool haveScale = a.scales.size() >= n;
    const bool haveOpacity = a.opacities.size() >= n;
    const int degree = std::clamp(a.shDegree, 0, 3);
    const size_t perParticle = static_cast<size_t>((degree + 1) * (degree + 1));
    const bool haveSh = a.shCoefficients.size() >= n * perParticle && perParticle > 0;
    const uint32_t rest = haveSh ? static_cast<uint32_t>(perParticle - 1) : 0;

    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 14 + rest * 3;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = rest; e.restColourOuter = 0;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    e.colour = io::SplatEncoding::Colour::ShDc;
    e.rotation = io::SplatEncoding::Rotation::Float;

    raw.count = static_cast<uint32_t>(n);
    raw.records.resize(n * e.floatsPerRecord);
    for (size_t i = 0; i < n; ++i) {
        float* r = raw.records.data() + i * e.floatsPerRecord;
        const pxr::GfVec3f& p = a.positions[i];
        r[0] = p[0]; r[1] = p[1]; r[2] = p[2];
        r[3] = haveOpacity ? a.opacities[i] : 1.0F;
        const pxr::GfVec3f s = haveScale ? a.scales[i] : pxr::GfVec3f(1.0F);
        r[4] = s[0]; r[5] = s[1]; r[6] = s[2];
        if (haveRotation) {
            const pxr::GfQuatf& q = a.orientations[i];
            r[7] = q.GetReal();
            r[8] = q.GetImaginary()[0];
            r[9] = q.GetImaginary()[1];
            r[10] = q.GetImaginary()[2];
        } else {
            r[7] = 1.0F; r[8] = 0.0F; r[9] = 0.0F; r[10] = 0.0F;
        }
        if (haveSh) {
            const size_t base = i * perParticle;
            for (size_t k = 0; k < perParticle; ++k) {
                const pxr::GfVec3f& c = a.shCoefficients[base + k];
                const size_t at = 11 + k * 3;
                r[at] = c[0]; r[at + 1] = c[1]; r[at + 2] = c[2];
            }
        } else {
            r[11] = 0.0F; r[12] = 0.0F; r[13] = 0.0F;   // DC 0: base colour 0.5, the schema's default
        }
    }
    return raw;
}

io::RawPoints rawPointsFrom(const PointsArrays& a, std::string source) {
    io::RawPoints raw;
    raw.source = std::move(source);
    const size_t n = a.positions.size();
    raw.count = static_cast<uint32_t>(n);
    raw.colourKind = a.colours.empty() ? 0u : 2u;
    raw.records.resize(n * 6);
    const bool perPoint = a.colours.size() >= n;
    for (size_t i = 0; i < n; ++i) {
        float* r = raw.records.data() + i * 6;
        const pxr::GfVec3f& p = a.positions[i];
        r[0] = p[0]; r[1] = p[1]; r[2] = p[2];
        const pxr::GfVec3f c = a.colours.empty() ? pxr::GfVec3f(1.0F)
                               : a.colours[perPoint ? i : 0];
        r[3] = c[0]; r[4] = c[1]; r[5] = c[2];
    }
    return raw;
}

}   // namespace lrt::usd
