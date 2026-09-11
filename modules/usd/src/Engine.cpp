// Copyright (c) 2026 lucabRTrender contributors.
#include "Engine.h"

#include <algorithm>

#include "lrt/core/Log.h"
#include "lrt/gpu/CommandBatch.h"

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

void Engine::setSplats(const pxr::SdfPath& id, std::optional<ParticleFieldArrays> raw,
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

void Engine::setPoints(const pxr::SdfPath& id, std::optional<PointsArrays> raw,
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

void Engine::setMesh(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                     std::optional<MeshArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                     std::optional<MeshLook> look, std::optional<std::vector<InstancerLink>> instancing) {
    const std::lock_guard<std::mutex> held(guard_);
    MeshEntry& entry = meshes_[id];
    if (instancing.has_value()) {
        entry.instancing = std::move(*instancing);
        entry.chainDirty = true;
    }
    entry.primId = static_cast<uint32_t>(primId);
    entry.renderTag = renderTag;
    if (arrays.has_value()) {
        entry.pending = std::move(arrays);
    }
    if (transform != nullptr) {
        entry.objectToWorld = *transform;
    }
    if (visible.has_value()) {
        entry.visible = *visible;
    }
    if (look.has_value()) {
        entry.look = *look;
    }
}

void Engine::setInstancer(const pxr::SdfPath& id, const pxr::SdfPath& parent, InstancerArrays arrays) {
    const std::lock_guard<std::mutex> held(guard_);
    InstancerEntry& entry = instancers_[id];
    entry.arrays = std::move(arrays);
    entry.parent = parent;
    entry.version = ++instancerVersion_;
}

void Engine::removeInstancer(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    instancers_.erase(id);
}

void Engine::remove(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    splats_.erase(id);
    points_.erase(id);
    meshes_.erase(id);
}

Result<size_t> Engine::commit() {
    const std::lock_guard<std::mutex> held(guard_);
    size_t uploaded = 0;
    for (auto& [id, entry] : meshes_) {
        if (!entry.pending.has_value()) {
            continue;
        }
        const MeshArrays& a = *entry.pending;
        geom::MeshInput input;
        input.source = id.GetString();
        input.points = streamOf(a.points);
        input.faceVertexCounts = std::span<const int32_t>(a.faceVertexCounts.cdata(), a.faceVertexCounts.size());
        input.faceVertexIndices = std::span<const int32_t>(a.faceVertexIndices.cdata(), a.faceVertexIndices.size());
        input.holeIndices = std::span<const int32_t>(a.holeIndices.cdata(), a.holeIndices.size());
        input.leftHanded = a.leftHanded;
        input.smoothNormals = a.smoothNormals;
        std::vector<geom::PrimvarInput> primvars;
        for (const PrimvarArrays& p : a.primvars) {
            geom::PrimvarInput primvar;
            primvar.name = p.name;
            // HdInterpolation's order is geom::Interpolation's.
            primvar.interpolation = static_cast<geom::Interpolation>(p.interpolation);
            primvar.values = primvarStreamOf(p.values, &primvar.components);
            primvar.indices = std::span<const int32_t>(p.indices.cdata(), p.indices.size());
            primvars.push_back(std::move(primvar));
        }
        input.primvars = primvars;
        entry.gpu.reset();
        if (input.points.values() >= 3 && !a.faceVertexCounts.empty()) {
            if (!meshBuilder_.has_value()) {
                auto made = geom::MeshBuilder::create(*library_);
                if (!made) return std::move(made).error();
                meshBuilder_.emplace(std::move(*made));
            }
            auto mesh = meshBuilder_->build(input);
            if (mesh) {
                entry.gpu = std::make_shared<const geom::GpuMesh>(std::move(*mesh));
            } else {
                log::warn("hdLrt: {}: {}", id.GetString(), mesh.error().toString());
            }
        }
        entry.pending.reset();
        ++uploaded;
    }
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
        const scene::SplatStreams streams = splatStreams(*entry.pending, id.GetString());
        if (streams.count == 0) {
            entry.gpu.reset();
        } else {
            auto splats = loader_->upload(streams);
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
        const scene::PointStreams streams = pointStreams(*entry.pending, id.GetString());
        if (streams.count == 0) {
            entry.gpu.reset();
        } else {
            auto points = loader_->upload(streams);
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

Result<void> Engine::writeAov(const render::RenderTargets& targets, bool depth, const AovLayout& layout,
                              const double* projection, std::span<uint8_t> into) {
    const uint64_t bytes = uint64_t{targets.width} * targets.height * layout.channels * layout.componentBytes;
    if (into.size() < bytes || layout.channels == 0 || layout.channels > 4 ||
        (layout.componentBytes != 1 && layout.componentBytes != 2 && layout.componentBytes != 4)) {
        return Error(ErrorCode::InvalidArgument, "a render buffer the engine cannot fill");
    }
    const gpu::Buffer& source = depth ? targets.depth : targets.colour;
    if (!source.valid()) {
        return Error(ErrorCode::InvalidArgument, "nothing rendered to convert");
    }
    if (!aovConvert_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "lrt/usd/aov_convert", "aovConvert");
        if (!made) return std::move(made).error();
        aovConvert_.emplace(std::move(*made));
    }
    const uint32_t words = static_cast<uint32_t>((bytes + 3) / 4);
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{words} * 4;
    desc.elementBytes = 4;
    desc.label = "hydra.aov";
    auto out = gpu::Buffer::create(*device_, desc);
    if (!out) return std::move(out).error();
    gpu::BufferDesc one;
    one.bytes = 16;
    one.elementBytes = 16;
    one.label = "hydra.placeholder";
    auto placeholder = gpu::Buffer::create(*device_, one);
    if (!placeholder) return std::move(placeholder).error();
    gpu::CommandBatch batch(*device_);
    aovConvert_->dispatch(batch, {words, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(depth ? placeholder->rhi() : source.rhi());
        cursor["depth"].setBinding(depth ? source.rhi() : placeholder->rhi());
        cursor["words"].setBinding(out->rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(targets.width);
        p["height"].setData(targets.height);
        p["words"].setData(words);
        p["source"].setData(uint32_t{depth ? 1u : 0u});
        p["channels"].setData(layout.channels);
        p["componentBytes"].setData(layout.componentBytes);
        p["componentKind"].setData(layout.componentKind);
        // Matrix terms, not data: the host's projection, as floats.
        p["p22"].setData(static_cast<float>(projection[10]));
        p["p32"].setData(static_cast<float>(projection[14]));
        p["p23"].setData(static_cast<float>(projection[11]));
        p["p33"].setData(static_cast<float>(projection[15]));
    });
    LRT_TRY(batch.submit(true));
    // A readback for a host that maps the buffer: output IO, the only copy.
    return out->read(*device_, 0, bytes, into.data());
}

Result<void> Engine::render(const render::Projection& projection, const render::RenderSettings& settings,
                            render::RenderTargets& targets, Technique technique, bool settleStreams,
                            const pxr::TfTokenVector* renderTags) {
    lastTargets_ = &targets;
    std::vector<world::MeshInstance> meshInstances;
    std::vector<world::InstanceSet> meshSets;
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
        for (const auto& [id, entry] : meshes_) {
            if (!entry.visible || entry.gpu == nullptr) {
                continue;
            }
            if (renderTags != nullptr && !renderTags->empty() &&
                std::find(renderTags->begin(), renderTags->end(), entry.renderTag) == renderTags->end()) {
                continue;
            }
            if (!entry.instancing.empty()) {
                // Instanced: the chain, recomposed on the device when an
                // instancer in it changed.
                std::vector<uint64_t> versions;
                bool complete = true;
                for (const InstancerLink& link : entry.instancing) {
                    const auto found = instancers_.find(link.instancer);
                    complete = complete && found != instancers_.end();
                    versions.push_back(found != instancers_.end() ? found->second.version : 0);
                }
                if (!complete) {
                    continue;
                }
                auto& mutableEntry = const_cast<MeshEntry&>(entry);
                if (entry.chainDirty || versions != entry.chainVersions) {
                    if (!instancing_.has_value()) {
                        auto made = world::Instancing::create(*library_);
                        if (!made) return std::move(made).error();
                        instancing_.emplace(std::move(*made));
                    }
                    std::vector<world::InstancerLevel> levels;
                    for (const InstancerLink& link : entry.instancing) {
                        const InstancerArrays& a = instancers_.at(link.instancer).arrays;
                        world::InstancerLevel level;
                        level.indices = std::span<const int32_t>(link.indices.cdata(), link.indices.size());
                        level.translations = streamOf(a.translations);
                        level.rotations = streamOf(a.rotations);
                        level.scales = streamOf(a.scales);
                        level.transforms = streamOf(a.transforms);
                        level.instancerTransform = a.instancerTransform;
                        levels.push_back(level);
                    }
                    auto chain = instancing_->compose(levels);
                    if (!chain) return std::move(chain).error();
                    mutableEntry.chain = std::move(*chain);
                    mutableEntry.chainVersions = std::move(versions);
                    mutableEntry.chainDirty = false;
                }
                world::InstanceSet set;
                set.mesh = entry.gpu;
                set.chainRows = entry.chain.rows;
                set.count = entry.chain.count;
                set.prototype = entry.objectToWorld;
                set.primId = entry.primId;
                set.displayColor = entry.look.displayColor;
                set.displayOpacity = entry.look.displayOpacity;
                set.doubleSided = entry.look.doubleSided;
                meshSets.push_back(std::move(set));
                continue;
            }
            world::MeshInstance instance;
            instance.mesh = entry.gpu;
            instance.objectToWorld = entry.objectToWorld;
            instance.primId = entry.primId;
            instance.displayColor = entry.look.displayColor;
            instance.displayOpacity = entry.look.displayOpacity;
            instance.doubleSided = entry.look.doubleSided;
            meshInstances.push_back(std::move(instance));
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
    // Opaque layers first -- meshes, points -- then splats blended over them.
    const bool drawMeshes = !meshInstances.empty() || !meshSets.empty();
    if (drawMeshes && !device_->caps().rasterization) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            log::warn("hdLrt: meshes need a rasterising device until ray traced visibility lands");
        }
    }
    const bool meshLayer = drawMeshes && device_->caps().rasterization;
    if (meshLayer) {
        if (!scene_.has_value()) {
            auto scene = world::GpuScene::create(*library_);
            if (!scene) return std::move(scene).error();
            scene_.emplace(std::move(*scene));
            auto raster = technique::VisibilityRaster::create(*library_);
            if (!raster) return std::move(raster).error();
            visibilityRaster_.emplace(std::move(*raster));
            auto shading = technique::HeadlightShading::create(*library_);
            if (!shading) return std::move(shading).error();
            headlight_.emplace(std::move(*shading));
        }
        LRT_TRY(scene_->update(meshInstances, projection, meshSets));
        gpu::CommandBatch batch(*device_);
        LRT_TRY(visibilityRaster_->render(batch, *scene_, projection, settings.width, settings.height, visibility_));
        LRT_TRY(headlight_->shade(batch, *scene_, visibility_, projection, meshLayer_));
        LRT_TRY(batch.submit(true));
    }
    const bool pointLayer = !points.empty() && pointRasterizer_.has_value();
    if (pointLayer) {
        LRT_TRY(pointRasterizer_->render(projection, points, settings, pointLayer_));
    }
    const render::RenderTargets* under = nullptr;
    if (meshLayer && pointLayer) {
        const uint64_t pixels = uint64_t{settings.width} * settings.height;
        if (opaqueLayer_.width != settings.width || opaqueLayer_.height != settings.height ||
            !opaqueLayer_.colour.valid()) {
            gpu::BufferDesc colour;
            colour.bytes = pixels * 16;
            colour.elementBytes = 16;
            colour.label = "engine.opaque.colour";
            auto madeColour = gpu::Buffer::create(*device_, colour);
            if (!madeColour) return std::move(madeColour).error();
            gpu::BufferDesc depth;
            depth.bytes = pixels * 4;
            depth.elementBytes = 4;
            depth.label = "engine.opaque.depth";
            auto madeDepth = gpu::Buffer::create(*device_, depth);
            if (!madeDepth) return std::move(madeDepth).error();
            opaqueLayer_ = {settings.width, settings.height, std::move(*madeColour), std::move(*madeDepth)};
        }
        if (!nearest_.has_value()) {
            auto made = gpu::ComputeKernel::create(*library_, "lrt/technique/layers_nearest", "layersNearest");
            if (!made) return std::move(made).error();
            nearest_.emplace(std::move(*made));
        }
        gpu::CommandBatch batch(*device_);
        nearest_->dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colourA"].setBinding(meshLayer_.colour.rhi());
            cursor["depthA"].setBinding(meshLayer_.depth.rhi());
            cursor["colourB"].setBinding(pointLayer_.colour.rhi());
            cursor["depthB"].setBinding(pointLayer_.depth.rhi());
            cursor["colour"].setBinding(opaqueLayer_.colour.rhi());
            cursor["depth"].setBinding(opaqueLayer_.depth.rhi());
            cursor["params"]["pixels"].setData(static_cast<uint32_t>(pixels));
        });
        LRT_TRY(batch.submit(true));
        under = &opaqueLayer_;
    } else if (meshLayer) {
        under = &meshLayer_;
    } else if (pointLayer) {
        under = &pointLayer_;
    }
    if (under != nullptr) {
        LRT_TRY(rasterizer_->render(projection, splats, settings, targets, {}, under));
        return ok();
    }
    LRT_TRY(rasterizer_->render(projection, splats, settings, targets, points));
    return ok();
}

}   // namespace lrt::usd
