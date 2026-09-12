// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/light/LightTable.h"

#include <cmath>
#include <vector>

#include "lrt/gpu/Device.h"

namespace lrt::light {

Result<LightTable> LightTable::create(gpu::Device& device) {
    LightTable table;
    table.device_ = &device;
    return table;
}

LightRecord LightTable::recordOf(const Light& light) {
    LightRecord record;
    record.kind = static_cast<uint32_t>(light.kind);
    record.flags = (light.shadow ? kLightShadow : 0u) | (light.normalize ? kLightNormalize : 0u) |
                   (light.enableTemperature ? kLightTemperature : 0u);
    switch (light.kind) {
        case LightKind::Distant:
            record.sizeX = light.angle;
            break;
        case LightKind::Sphere:
        case LightKind::Disk:
            record.sizeX = light.radius;
            break;
        case LightKind::Rect:
            record.sizeX = light.width;
            record.sizeY = light.height;
            break;
        case LightKind::Cylinder:
            record.sizeX = light.radius;
            record.sizeY = light.length;
            break;
        case LightKind::Dome:
            break;
    }
    for (int c = 0; c < 3; ++c) {
        record.colour[c] = light.colour[c] * light.intensity;
    }
    record.exposure = light.exposure;
    record.temperature = light.temperature;
    // A cone of a half angle at or above a right angle shapes nothing.
    const bool shaped = light.coneAngle > 0.0F && light.coneAngle < 3.14159265358979F / 2.0F;
    record.coneCos = shaped ? std::cos(light.coneAngle) : -1.0F;
    record.coneSoftness = light.coneSoftness;
    record.texture = light.textureId;
    record.sampler = light.sampler;
    record.lightCategory = light.lightCategory;
    record.shadowCategory = light.shadowCategory;
    const std::array<float, 12> rows = light.lightToWorld.rows3x4();
    for (size_t k = 0; k < rows.size(); ++k) {
        record.rows[k] = rows[k];
    }
    return record;
}

/// What a light is worth to a frame: its emission times what it emits over.
/// A rough estimate is all a choice needs -- it only has to be positive and
/// roughly proportional, since the density it implies is divided back out.
static float powerOf(const Light& light) {
    const float luminance = 0.2126F * light.colour[0] + 0.7152F * light.colour[1] + 0.0722F * light.colour[2];
    float power = std::max(luminance, 0.0F) * std::max(light.intensity, 0.0F) * std::exp2(light.exposure);
    switch (light.kind) {
        case LightKind::Sphere:
            power *= light.normalize ? 1.0F : 4.0F * 3.14159265358979F * light.radius * light.radius;
            break;
        case LightKind::Disk:
            power *= light.normalize ? 1.0F : 3.14159265358979F * light.radius * light.radius;
            break;
        case LightKind::Rect:
            power *= light.normalize ? 1.0F : light.width * light.height;
            break;
        case LightKind::Cylinder:
            power *= light.normalize ? 1.0F : 2.0F * 3.14159265358979F * light.radius * light.length;
            break;
        case LightKind::Distant:
        case LightKind::Dome:
            break;   // over the whole sky either way
    }
    return std::max(power, 1.0e-6F);
}

Result<void> LightTable::set(std::span<const Light> lights) {
    std::vector<LightRecord> records;
    records.reserve(lights.size() + 1);
    shadows_ = false;
    domes_ = false;
    for (const Light& light : lights) {
        records.push_back(recordOf(light));
        shadows_ = shadows_ || light.shadow;
        domes_ = domes_ || light.kind == LightKind::Dome;
    }
    if (records.empty()) {
        records.emplace_back();   // a buffer to bind, which nothing reads
    }
    // Cumulative shares of the frame's power, for choosing one light.
    power_ = 0.0F;
    for (size_t k = 0; k < lights.size(); ++k) {
        power_ += powerOf(lights[k]);
        records[k].cumulative = power_;
    }
    count_ = static_cast<uint32_t>(lights.size());
    if (records.size() > capacity_ || !records_.valid()) {
        auto made = gpu::Buffer::fromSpan<LightRecord>(*device_, records, "lights.records");
        if (!made) return std::move(made).error();
        records_ = std::move(*made);
        capacity_ = static_cast<uint32_t>(records.size());
        return ok();
    }
    if (!records_.write(*device_, 0, records.size() * sizeof(LightRecord), records.data())) {
        return Error(ErrorCode::DeviceFailure, "lights: cannot upload the frame's records");
    }
    return ok();
}

void LightTable::bind(rhi::ShaderCursor cursor) const {
    cursor["lights"].setBinding(records_.rhi());
    cursor["lightCount"].setData(count_);
}

}   // namespace lrt::light
