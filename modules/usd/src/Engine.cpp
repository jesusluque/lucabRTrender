// Copyright (c) 2026 lucabRTrender contributors.
#include "Engine.h"

#include <algorithm>

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
                       const render::Mat4* transform, std::optional<bool> visible,
                       std::optional<render::SplatEdit> edit, std::optional<StreamedAsset> asset) {
    const std::lock_guard<std::mutex> held(guard_);
    SplatEntry& entry = splats_[id];
    if (asset.has_value()) {
        entry.assetPending = std::move(asset);
    }
    if (edit.has_value()) {
        entry.edit = *edit;
    }
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
        if (entry.assetPending.has_value()) {
            StreamedAsset asset = std::move(*entry.assetPending);
            entry.assetPending.reset();
            if (asset.path != entry.asset.path || asset.budget != entry.asset.budget) {
                entry.lodCloud.reset();
                entry.pool.reset();
                if (!asset.path.empty()) {
                    if (asset.budget > 0) {
                        auto pool = lod::StreamingPool::open(*device_, asset.path, {asset.budget, 2});
                        if (pool) {
                            entry.pool = std::move(*pool);
                        } else {
                            log::warn("hdLrt: {}: {}", id.GetString(), pool.error().toString());
                        }
                    } else {
                        auto read = lod::readLrtc(*device_, asset.path);
                        if (read) {
                            entry.lodCloud = std::make_unique<lod::LodCloud>(std::move(*read));
                        } else {
                            log::warn("hdLrt: {}: {}", id.GetString(), read.error().toString());
                        }
                    }
                }
                ++uploaded;
            }
            entry.asset = std::move(asset);
        }
        if (!entry.asset.path.empty()) {
            // The asset stands in for the prim's own arrays.
            entry.pending.reset();
            entry.gpu.reset();
            continue;
        }
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
                            render::RenderTargets& targets, Technique technique, bool settleStreams) {
    std::vector<render::SplatInstance> splats;
    std::vector<render::PointInstance> points;
    std::vector<lod::LodInstance> cuts;
    std::vector<lod::StreamingPool*> poolOf;   // per cut: its pool, if streamed
    {
        const std::lock_guard<std::mutex> held(guard_);
        for (const auto& [id, entry] : splats_) {
            if (!entry.visible) {
                continue;
            }
            if (entry.gpu != nullptr) {
                splats.push_back({entry.gpu.get(), entry.objectToWorld, entry.edit});
            }
            const lod::LodCloud* cloud = entry.pool != nullptr ? &entry.pool->cloud() : entry.lodCloud.get();
            if (cloud == nullptr) {
                continue;
            }
            if (technique == Technique::RayTraced) {
                // A cut changes every frame, and the ray tracer would rebuild
                // every frame: it draws the whole cloud, when it is whole.
                if (entry.lodCloud != nullptr) {
                    splats.push_back({&entry.lodCloud->splats, entry.objectToWorld, entry.edit});
                } else {
                    log::warn("hdLrt: {}: a streamed asset is drawn by the rasteriser only", id.GetString());
                }
                continue;
            }
            cuts.push_back({cloud, entry.objectToWorld, entry.edit, entry.asset.threshold});
            poolOf.push_back(entry.pool.get());
        }
        for (const auto& [id, entry] : points_) {
            if (entry.visible && entry.gpu != nullptr) {
                points.push_back({entry.gpu.get(), entry.objectToWorld, entry.style});
            }
        }
    }
    if (!cuts.empty()) {
        if (!cutter_.has_value()) {
            auto made = lod::CutSelector::create(*library_);
            if (!made) return std::move(made).error();
            cutter_.emplace(std::move(*made));
        }
        const bool streamed = std::any_of(poolOf.begin(), poolOf.end(), [](auto* p) { return p != nullptr; });
        std::vector<lod::CutStats> stats;
        for (int round = 0;; ++round) {
            auto selected = cutter_->select(projection, cuts, 0.0F, streamed ? &stats : nullptr);
            if (!selected) return std::move(selected).error();
            if (!streamed) {
                splats.insert(splats.end(), selected->begin(), selected->end());
                break;
            }
            for (size_t k = 0; k < poolOf.size(); ++k) {
                if (poolOf[k] != nullptr) {
                    poolOf[k]->want(stats[k].needs);
                }
            }
            uint32_t placed = 0;
            for (size_t k = 0; k < poolOf.size(); ++k) {
                const auto earlier = poolOf.begin() + static_cast<std::ptrdiff_t>(k);
                // A pool two prims share is updated once.
                if (poolOf[k] != nullptr && std::find(poolOf.begin(), earlier, poolOf[k]) == earlier) {
                    auto n = poolOf[k]->update(settleStreams);
                    if (!n) return std::move(n).error();
                    placed += *n;
                }
            }
            if (!settleStreams || placed == 0 || round >= 64) {
                splats.insert(splats.end(), selected->begin(), selected->end());
                break;
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
