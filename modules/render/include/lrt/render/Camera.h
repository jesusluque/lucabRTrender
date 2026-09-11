// Copyright (c) 2026 lucabRTrender contributors.
//
// A camera the way openFXplayer and Nuke describe one, and what the kernels
// need from it.
//
// World: right-handed, +Y up, a camera looks down its own -Z (openFXplayer
// D36). The kernels project along +z with x right and y up, so the view is the
// camera's inverse with z negated -- `viewFromCamera`, the one place that
// convention is written. Lens: a focal length and a horizontal aperture, not a
// field of view, because that is what a camera report says; square pixels
// (vaperture travels for the record and the frustum, not the projection).
#pragma once

#include <cstdint>

#include "aofx/Transform.h"

namespace lrt::render {

using Mat4 = aofx::xform::Mat4;
using Vec3 = aofx::xform::Vec3;

struct Lens {
    enum class Projection { Perspective, Orthographic };
    Projection projection = Projection::Perspective;
    double focal = 50.0;          ///< mm
    double haperture = 24.576;    ///< mm
    double vaperture = 18.672;    ///< mm
    double nearZ = 0.1;
    double farZ = 10000.0;
    double windowTranslate[2] = {0.0, 0.0};   ///< in half-widths, both axes
    double windowScale[2] = {1.0, 1.0};
    double windowRoll = 0.0;                  ///< degrees, counter-clockwise
};

struct Camera {
    Mat4 cameraToWorld = Mat4::identity();
    Lens lens;

    /// A camera at `eye` looking at `target`.
    [[nodiscard]] static Camera lookingAt(const Vec3& eye, const Vec3& target,
                                          const Vec3& up = {0.0, 1.0, 0.0});
};

/// The kernel's view of a camera for one image size.
struct Projection {
    Mat4   worldToView;   ///< +z forward
    double focalX = 0.0;  ///< pixels per unit of x/z (perspective) or per world unit (ortho)
    double focalY = 0.0;
    double centreX = 0.0;
    double centreY = 0.0;
    double nearZ = 0.1;
    double farZ = 10000.0;
    bool   orthographic = false;
    Vec3   eyeWorld;
};

[[nodiscard]] Mat4 viewFromCamera(const Mat4& cameraToWorld);
[[nodiscard]] Projection projectionFor(const Camera& camera, uint32_t width, uint32_t height);

}   // namespace lrt::render
