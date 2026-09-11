// Copyright (c) 2026 lucabRTrender contributors.
//
// Setting shaders/lrt/splat/frame.slang's FrameParams by name, shared by every
// renderer that speaks it (the tile rasteriser and the reference).
#pragma once

#include <array>
#include <cstdint>

#include <slang-rhi/shader-cursor.h>

#include "lrt/render/Camera.h"
#include "lrt/render/Points.h"
#include "lrt/render/TileRasterizer.h"

namespace lrt::render {

inline uint32_t bitsFor(uint32_t values) {
    uint32_t bits = 1;
    while (bits < 32 && (uint64_t{1} << bits) < values) {
        ++bits;
    }
    return bits;
}

struct FrameCommon {
    uint32_t width, height, tilesX, tilesY;
    const Projection* projection;
    const RenderSettings* settings;
};

inline void setFrame(rhi::ShaderCursor cursor, const FrameCommon& f) {
    rhi::ShaderCursor p = cursor["params"];
    p["width"].setData(f.width);
    p["height"].setData(f.height);
    p["tilesX"].setData(f.tilesX);
    p["tilesY"].setData(f.tilesY);
    p["focalX"].setData(static_cast<float>(f.projection->focalX));
    p["focalY"].setData(static_cast<float>(f.projection->focalY));
    p["centreX"].setData(static_cast<float>(f.projection->centreX));
    p["centreY"].setData(static_cast<float>(f.projection->centreY));
    p["nearZ"].setData(static_cast<float>(f.projection->nearZ));
    p["farZ"].setData(static_cast<float>(f.projection->farZ));
    p["orthographic"].setData(uint32_t{f.projection->orthographic ? 1u : 0u});
    p["antialias"].setData(uint32_t{f.settings->antialias ? 1u : 0u});
    p["shLimit"].setData(f.settings->maxShDegree);
    p["depthMode"].setData(uint32_t{f.settings->depth == RenderSettings::Depth::Mean ? 0u : 1u});
    p["depthThreshold"].setData(f.settings->depthThreshold);
    p["bgR"].setData(f.settings->background[0]);
    p["bgG"].setData(f.settings->background[1]);
    p["bgB"].setData(f.settings->background[2]);
    p["bgA"].setData(f.settings->background[3]);
    p["linearise"].setData(uint32_t{f.settings->linearise ? 1u : 0u});
    p["hasUnder"].setData(uint32_t{0});
    p["tileBits"].setData(bitsFor(f.tilesX * f.tilesY));
}

inline void setObject(rhi::ShaderCursor cursor, const Mat4& objectToView, const Vec3& eyeObject) {
    rhi::ShaderCursor p = cursor["params"];
    static constexpr const char* kNames[12] = {"m00", "m01", "m02", "m03", "m10", "m11",
                                              "m12", "m13", "m20", "m21", "m22", "m23"};
    const std::array<float, 12> rows = objectToView.rows3x4();
    for (size_t k = 0; k < 12; ++k) {
        p[kNames[k]].setData(rows[k]);
    }
    p["eyeX"].setData(static_cast<float>(eyeObject.x));
    p["eyeY"].setData(static_cast<float>(eyeObject.y));
    p["eyeZ"].setData(static_cast<float>(eyeObject.z));
}


/// EditParams (shaders/lrt/common/edit.slang) under `e`: the struct nested as
/// `edit` in FrameParams and RtParams.
inline void setEdit(rhi::ShaderCursor e, const SplatEdit& edit) {
    e["active"].setData(uint32_t{edit.active ? 1u : 0u});
    e["shape"].setData(static_cast<uint32_t>(edit.shape));
    e["mode"].setData(static_cast<uint32_t>(edit.mode));
    e["invert"].setData(uint32_t{edit.invert ? 1u : 0u});
    e["centreX"].setData(edit.centre[0]);
    e["centreY"].setData(edit.centre[1]);
    e["centreZ"].setData(edit.centre[2]);
    e["sizeX"].setData(edit.size[0]);
    e["sizeY"].setData(edit.size[1]);
    e["sizeZ"].setData(edit.size[2]);
    e["tintR"].setData(edit.tint[0]);
    e["tintG"].setData(edit.tint[1]);
    e["tintB"].setData(edit.tint[2]);
    e["saturation"].setData(edit.saturation);
    e["brightness"].setData(edit.brightness);
    e["opacity"].setData(edit.opacity);
    e["minOpacity"].setData(edit.minOpacity);
    e["maxScale"].setData(edit.maxScale);
}

/// PointParams (shaders/lrt/points/point_frame.slang) under `cursor`, which is
/// the root's "params" for the raster and EDL passes and "points" for the
/// disc projection.
inline void setPointParams(rhi::ShaderCursor p, const Projection& projection, uint32_t width,
                           uint32_t height, const Mat4& objectToView, const PointStyle& style,
                           uint32_t count) {
    p["width"].setData(width);
    p["height"].setData(height);
    p["focalX"].setData(static_cast<float>(projection.focalX));
    p["focalY"].setData(static_cast<float>(projection.focalY));
    p["centreX"].setData(static_cast<float>(projection.centreX));
    p["centreY"].setData(static_cast<float>(projection.centreY));
    p["nearZ"].setData(static_cast<float>(projection.nearZ));
    p["farZ"].setData(static_cast<float>(projection.farZ));
    p["orthographic"].setData(uint32_t{projection.orthographic ? 1u : 0u});
    p["sizeMode"].setData(uint32_t{style.sizeMode == PointStyle::Size::Pixels ? 1u : 0u});
    p["size"].setData(style.size);
    p["count"].setData(count);
    static constexpr const char* kNames[12] = {"m00", "m01", "m02", "m03", "m10", "m11",
                                              "m12", "m13", "m20", "m21", "m22", "m23"};
    const std::array<float, 12> rows = objectToView.rows3x4();
    for (size_t k = 0; k < 12; ++k) {
        p[kNames[k]].setData(rows[k]);
    }
    p["colourMode"].setData(uint32_t{style.constantColour ? 1u : 0u});
    p["colourR"].setData(style.colour[0]);
    p["colourG"].setData(style.colour[1]);
    p["colourB"].setData(style.colour[2]);
    p["edlStrength"].setData(style.edlStrength);
    p["edlRadius"].setData(style.edlRadius);
    p["surface"].setData(uint32_t{style.surfaceDepthOffset > 0.0F ? 1u : 0u});
    p["depthOffset"].setData(style.surfaceDepthOffset);
}

}   // namespace lrt::render
