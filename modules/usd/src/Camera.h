// Copyright (c) 2026 lucabRTrender contributors.
//
// The camera sprim: hd's HdCamera, which reads the camera's parameters and
// its transform at the frame, and beside it the transform at the samples
// that bracket the shutter -- a camera that moves while the shutter is open
// blurs everything it sees, and only the samples say so.
#pragma once

#include <pxr/base/gf/matrix4d.h>
#include <pxr/imaging/hd/camera.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdLrtCamera final : public HdCamera {
public:
    explicit HdLrtCamera(SdfPath const& id) : HdCamera(id) {}

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

    /// Whether the camera's transform differs between the shutter's samples.
    [[nodiscard]] bool Moves() const { return _moves; }
    [[nodiscard]] const GfMatrix4d& GetTransformStart() const { return _start; }
    [[nodiscard]] const GfMatrix4d& GetTransformEnd() const { return _end; }
    [[nodiscard]] double GetTimeStart() const { return _timeStart; }
    [[nodiscard]] double GetTimeEnd() const { return _timeEnd; }

private:
    bool       _moves = false;
    GfMatrix4d _start{1.0};
    GfMatrix4d _end{1.0};
    double     _timeStart = 0.0;
    double     _timeEnd = 0.0;
};

PXR_NAMESPACE_CLOSE_SCOPE
