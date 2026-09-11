// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/render/Camera.h"

#include <algorithm>
#include <cmath>

namespace lrt::render {

Camera Camera::lookingAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    Camera camera;
    camera.cameraToWorld = aofx::xform::lookAt(eye, target, up);
    return camera;
}

Mat4 viewFromCamera(const Mat4& cameraToWorld) {
    return aofx::xform::scaling(Vec3{1.0, 1.0, -1.0}) *
           aofx::xform::viewFromCameraWorld(cameraToWorld);
}

Projection projectionFor(const Camera& camera, uint32_t width, uint32_t height) {
    Projection out;
    out.worldToView = viewFromCamera(camera.cameraToWorld);
    if (std::abs(camera.lens.windowRoll) > 1e-12) {
        // A turn of the picture about the view's own z, after the flip.
        out.worldToView = aofx::xform::rotationZ(camera.lens.windowRoll) * out.worldToView;
    }
    const double w = static_cast<double>(width);
    const double h = static_cast<double>(height);
    // W * focal / haperture: pixels per unit of x/z, and pixels per world unit
    // for an orthographic camera -- Nuke's rule, why focal is an ortho zoom.
    const double focal = w * std::max(camera.lens.focal, 1e-9) /
                         std::max(camera.lens.haperture, 1e-9);
    out.focalX = focal * camera.lens.windowScale[0];
    out.focalY = focal * camera.lens.windowScale[1];
    out.centreX = w * 0.5 + camera.lens.windowTranslate[0] * w * 0.5;
    out.centreY = h * 0.5 + camera.lens.windowTranslate[1] * w * 0.5;
    out.nearZ = camera.lens.nearZ;
    out.farZ = camera.lens.farZ;
    out.orthographic = camera.lens.projection == Lens::Projection::Orthographic;
    out.eyeWorld = camera.cameraToWorld.translation();
    return out;
}

}   // namespace lrt::render
