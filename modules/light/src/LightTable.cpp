// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/light/LightTable.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ShaderLibrary.h"

#include <cmath>
#include <vector>

#include "lrt/gpu/Device.h"

namespace lrt::light {

Result<LightTable> LightTable::create(gpu::ShaderLibrary& library) {
    gpu::Device& device = library.device();
    LightTable table;
    table.device_ = &device;
    auto prefix = gpu::ComputeKernel::create(library, "lrt/light/light_prefix", "lightPrefix");
    if (!prefix) return std::move(prefix).error();
    table.prefix_.emplace(std::move(*prefix));
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
    count_ = static_cast<uint32_t>(lights.size());
    if (records.size() > capacity_ || !records_.valid()) {
        auto made = gpu::Buffer::fromSpan<LightRecord>(*device_, records, "lights.records");
        if (!made) return std::move(made).error();
        records_ = std::move(*made);
        capacity_ = static_cast<uint32_t>(records.size());
    } else if (!records_.write(*device_, 0, records.size() * sizeof(LightRecord), records.data())) {
        return Error(ErrorCode::DeviceFailure, "lights: cannot upload the frame's records");
    }
    // Each light's share of the frame's power, accumulated on the device:
    // the host uploads what was authored and computes nothing on it.
    if (count_ > 0) {
        gpu::CommandBatch batch(*device_);
        prefix_->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["records"].setBinding(records_.rhi());
            cursor["count"].setData(count_);
        });
        LRT_TRY(batch.submit(true));
    }
    return ok();
}

void LightTable::bind(rhi::ShaderCursor cursor) const {
    cursor["lights"].setBinding(records_.rhi());
    cursor["lightCount"].setData(count_);
}

}   // namespace lrt::light
