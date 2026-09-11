// Copyright (c) 2026 lucabRTrender contributors.
#include "Engine.h"

#include "lrt/core/Log.h"

namespace lrt::usd {

std::unique_ptr<Engine> Engine::create(std::string& why) {
    auto device = gpu::Device::create();
    if (!device) {
        why = device.error().toString();
        return nullptr;
    }
    auto engine = std::unique_ptr<Engine>(new Engine());
    engine->device_ = *device;
    engine->library_ = std::make_unique<gpu::ShaderLibrary>(engine->device_);
    auto loader = scene::CloudLoader::create(*engine->library_);
    if (!loader) {
        why = loader.error().toString();
        return nullptr;
    }
    engine->loader_.emplace(std::move(*loader));
    auto rasterizer = render::TileRasterizer::create(*engine->library_);
    if (!rasterizer) {
        why = rasterizer.error().toString();
        return nullptr;
    }
    engine->rasterizer_.emplace(std::move(*rasterizer));
    if (engine->device_->caps().rasterization) {
        if (auto points = render::PointRasterizer::create(*engine->library_)) {
            engine->pointRasterizer_.emplace(std::move(*points));
        } else {
            log::warn("hdLrt: points will be drawn as discs: {}", points.error().toString());
        }
    }
    return engine;
}

void Engine::setSplats(const pxr::SdfPath& id, std::optional<io::RawSplats> raw,
                       const render::Mat4* transform, std::optional<bool> visible) {
    const std::lock_guard<std::mutex> held(guard_);
    SplatEntry& entry = splats_[id];
    if (raw.has_value()) {
        entry.pending = std::move(raw);
    }
    if (transform != nullptr) {
        entry.objectToWorld = *transform;
    }
    if (visible.has_value()) {
        entry.visible = *visible;
    }
}

void Engine::setPoints(const pxr::SdfPath& id, std::optional<io::RawPoints> raw,
                       const render::Mat4* transform, std::optional<bool> visible,
                       std::optional<render::PointStyle> style) {
    const std::lock_guard<std::mutex> held(guard_);
    PointsEntry& entry = points_[id];
    if (raw.has_value()) {
        entry.pending = std::move(raw);
    }
    if (transform != nullptr) {
        entry.objectToWorld = *transform;
    }
    if (visible.has_value()) {
        entry.visible = *visible;
    }
    if (style.has_value()) {
        entry.style = *style;
    }
}

void Engine::remove(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    splats_.erase(id);
    points_.erase(id);
}

Result<size_t> Engine::commit() {
    const std::lock_guard<std::mutex> held(guard_);
    size_t uploaded = 0;
    for (auto& [id, entry] : splats_) {
        if (!entry.pending.has_value()) {
            continue;
        }
        if (entry.pending->count == 0) {
            entry.gpu.reset();
        } else {
            auto splats = loader_->upload(*entry.pending);
            if (!splats) {
                log::warn("hdLrt: {}: {}", id.GetString(), splats.error().toString());
                entry.gpu.reset();
            } else {
                entry.gpu = std::make_unique<scene::GpuSplats>(std::move(*splats));
            }
        }
        entry.pending.reset();
        ++uploaded;
    }
    for (auto& [id, entry] : points_) {
        if (!entry.pending.has_value()) {
            continue;
        }
        if (entry.pending->count == 0) {
            entry.gpu.reset();
        } else {
            auto points = loader_->upload(*entry.pending);
            if (!points) {
                log::warn("hdLrt: {}: {}", id.GetString(), points.error().toString());
                entry.gpu.reset();
            } else {
                entry.gpu = std::make_unique<scene::GpuPoints>(std::move(*points));
            }
        }
        entry.pending.reset();
        ++uploaded;
    }
    return uploaded;
}

Result<void> Engine::render(const render::Projection& projection, const render::RenderSettings& settings,
                            render::RenderTargets& targets, Technique technique) {
    std::vector<render::SplatInstance> splats;
    std::vector<render::PointInstance> points;
    {
        const std::lock_guard<std::mutex> held(guard_);
        for (const auto& [id, entry] : splats_) {
            if (entry.visible && entry.gpu != nullptr) {
                splats.push_back({entry.gpu.get(), entry.objectToWorld});
            }
        }
        for (const auto& [id, entry] : points_) {
            if (entry.visible && entry.gpu != nullptr) {
                points.push_back({entry.gpu.get(), entry.objectToWorld, entry.style});
            }
        }
    }
    if (technique == Technique::RayTraced) {
        if (!points.empty()) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                log::warn("hdLrt: the ray traced technique draws splats only; points are left out");
            }
        }
        if (!rayTracer_.has_value()) {
            auto made = render::GaussianRayTracer::create(*library_);
            if (!made) return std::move(made).error();
            rayTracer_.emplace(std::move(*made));
        }
        LRT_TRY(rayTracer_->render(projection, splats, settings, targets));
        return ok();
    }
    if (!points.empty() && pointRasterizer_.has_value()) {
        LRT_TRY(pointRasterizer_->render(projection, points, settings, pointLayer_));
        LRT_TRY(rasterizer_->render(projection, splats, settings, targets, {}, &pointLayer_));
        return ok();
    }
    LRT_TRY(rasterizer_->render(projection, splats, settings, targets, points));
    return ok();
}

}   // namespace lrt::usd
