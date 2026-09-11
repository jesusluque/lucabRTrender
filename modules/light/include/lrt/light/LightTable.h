// Copyright (c) 2026 lucabRTrender contributors.
//
// The scene's lights on the device: one record per light, in world space and
// in the units USD authored, since what a record becomes -- exposure, the
// blackbody of a colour temperature, the area a `normalize` divides by -- is
// derived in the shader (shaders/lrt/light/lights.slang).
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/render/Camera.h"

namespace lrt::gpu {
class Device;
}

namespace lrt::light {

enum class LightKind : uint32_t {
    Distant = 0,
    Sphere = 1,
    Disk = 2,
    Rect = 3,
    Dome = 4,
};

/// A light as USD authored it. Angles are radians; `lightToWorld` puts the
/// light where UsdLux has it, shining down its own -Z.
struct Light {
    LightKind    kind = LightKind::Distant;
    render::Mat4 lightToWorld = render::Mat4::identity();
    float        colour[3] = {1.0F, 1.0F, 1.0F};
    float        intensity = 1.0F;
    float        exposure = 0.0F;
    float        radius = 0.5F;        ///< sphere and disk
    float        width = 1.0F;         ///< rect
    float        height = 1.0F;        ///< rect
    float        angle = 0.0F;         ///< distant: the sun's angular diameter
    float        temperature = 6500.0F;
    bool         enableTemperature = false;
    bool         normalize = false;
    bool         shadow = true;
    float        coneAngle = 0.0F;     ///< shaping: the half angle; 0 or 180 means none
    float        coneSoftness = 0.0F;
};

/// What shaders/lrt/light/lights.slang reads.
struct LightRecord {
    uint32_t kind = 0;
    uint32_t flags = 0;
    float    sizeX = 0.0F;
    float    sizeY = 0.0F;
    float    colour[3] = {0.0F, 0.0F, 0.0F};
    float    exposure = 0.0F;
    float    temperature = 6500.0F;
    float    coneCos = -1.0F;
    float    coneSoftness = 0.0F;
    float    pad0 = 0.0F;
    float    rows[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};   ///< light to world, rows of a 3x4
};

/// LightRecord::flags
inline constexpr uint32_t kLightShadow = 1;
inline constexpr uint32_t kLightNormalize = 2;
inline constexpr uint32_t kLightTemperature = 4;

class LightTable {
public:
    [[nodiscard]] static Result<LightTable> create(gpu::Device& device);

    /// The frame's lights, uploaded. The buffer is remade when it must grow.
    [[nodiscard]] Result<void> set(std::span<const Light> lights);

    [[nodiscard]] uint32_t count() const noexcept { return count_; }
    [[nodiscard]] bool     anyShadow() const noexcept { return shadows_; }
    [[nodiscard]] const gpu::Buffer& records() const noexcept { return records_; }

    /// `lights` and `lightCount`, by name.
    void bind(rhi::ShaderCursor cursor) const;

    /// The record a light becomes, for tests and for the table itself.
    [[nodiscard]] static LightRecord recordOf(const Light& light);

private:
    gpu::Device* device_ = nullptr;
    gpu::Buffer  records_;
    uint32_t     count_ = 0;
    uint32_t     capacity_ = 0;
    bool         shadows_ = false;
};

}   // namespace lrt::light
