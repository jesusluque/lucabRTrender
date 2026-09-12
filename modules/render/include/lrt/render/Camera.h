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

    double exposure = 0.0;   ///< stops, as UsdGeomCamera authors it
    /// The diaphragm, as UsdGeomCamera authors it: 0 is a pinhole. The lens
    /// radius is focal / (2 fStop) with focal in tenths of a scene unit, the
    /// unit UsdGeomCamera's focalLength is in.
    double fStop = 0.0;
    double focusDistance = 0.0;   ///< scene units
    /// Radial distortion, in ndc radius: p' = p (1 + k1 r^2 + k2 r^4).
    double distortionK1 = 0.0;
    double distortionK2 = 0.0;
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
    double exposure = 0.0;   ///< stops: the frame is scaled by 2^exposure
    /// A thin lens: its radius in scene units (0 for a pinhole) and the depth
    /// in focus. Only the path tracer honours it, since it is the one route
    /// that can cast a ray from somewhere other than the pixel's centre.
    double lensRadius = 0.0;
    double focusDistance = 0.0;
    double distortionK1 = 0.0;   ///< radial, in ndc radius: p' = p (1 + k1 r^2 + k2 r^4)
    double distortionK2 = 0.0;
};

[[nodiscard]] Mat4 viewFromCamera(const Mat4& cameraToWorld);
[[nodiscard]] Projection projectionFor(const Camera& camera, uint32_t width, uint32_t height);

}   // namespace lrt::render
