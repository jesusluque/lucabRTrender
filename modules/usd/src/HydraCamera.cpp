// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/usd/HydraCamera.h"

#include <cmath>

namespace lrt::usd {

render::Mat4 fromUsd(const pxr::GfMatrix4d& m) {
    // USD: p' = p * M, translation in row 3. Engine: p' = M p, translation in column 3.
    render::Mat4 out;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.at(r, c) = m[c][r];
        }
    }
    return out;
}

render::Projection projectionFromHydra(const pxr::GfMatrix4d& worldToView,
                                       const pxr::GfMatrix4d& p, uint32_t width,
                                       uint32_t height) {
    render::Projection out;
    // USD's view looks down -Z; the kernels look down +z (openFXplayer D36).
    out.worldToView = aofx::xform::scaling({1.0, 1.0, -1.0}) * fromUsd(worldToView);
    out.eyeWorld = aofx::xform::inverseAffine(fromUsd(worldToView)).translation();
    const double w = static_cast<double>(width);
    const double h = static_cast<double>(height);
    out.orthographic = std::abs(p[3][3] - 1.0) < 1e-9 && std::abs(p[2][3]) < 1e-9;
    if (out.orthographic) {
        // ndc = x * P00 + P30, view z = -z'.
        out.focalX = w * 0.5 * p[0][0];
        out.focalY = h * 0.5 * p[1][1];
        out.centreX = w * 0.5 * (1.0 + p[3][0]);
        out.centreY = h * 0.5 * (1.0 + p[3][1]);
        // ndc z = -z' * P22 + P32 ranges -1..1 over near..far.
        out.nearZ = (p[3][2] + 1.0) / p[2][2];
        out.farZ = (p[3][2] - 1.0) / p[2][2];
    } else {
        // ndc.x = P00 * x / z' - P20, with z' = -z the distance in front.
        out.focalX = w * 0.5 * p[0][0];
        out.focalY = h * 0.5 * p[1][1];
        out.centreX = w * 0.5 * (1.0 - p[2][0]);
        out.centreY = h * 0.5 * (1.0 - p[2][1]);
        out.nearZ = p[3][2] / (p[2][2] - 1.0);
        out.farZ = p[3][2] / (p[2][2] + 1.0);
    }
    if (out.nearZ > out.farZ) {
        std::swap(out.nearZ, out.farZ);
    }
    out.nearZ = std::abs(out.nearZ);
    out.farZ = std::abs(out.farZ);
    return out;
}

}   // namespace lrt::usd
