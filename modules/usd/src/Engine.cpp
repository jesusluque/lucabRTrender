// Copyright (c) 2026 lucabRTrender contributors.
#include "Engine.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>

#include <MaterialXFormat/File.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

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
                       std::optional<render::SplatEdit> edit, std::optional<StreamedAsset> asset,
                       std::optional<bool> relight,
                       std::optional<std::vector<pxr::TfToken>> categories) {
    const std::lock_guard<std::mutex> held(guard_);
    SplatEntry& entry = splats_[id];
    if (asset.has_value()) {
        entry.assetPending = std::move(asset);
    }
    if (edit.has_value()) {
        entry.edit = *edit;
    }
    if (relight) {
        entry.relight = *relight;
    }
    if (categories) {
        entry.categories = std::move(*categories);
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
        entry.subsetMaterials.clear();
        for (const MeshSubset& subset : arrays->subsets) {
            entry.subsetMaterials.push_back(subset.material);
        }
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

void Engine::setMaterial(const pxr::SdfPath& id, std::shared_ptr<void> mtlxDocument) {
    const std::lock_guard<std::mutex> held(guard_);
    MaterialEntry& entry = materials_[id];
    entry.document = std::move(mtlxDocument);
    entry.cutout = material::MaterialCompiler::cutsOut(entry.document);
    entry.pending = true;
}

void Engine::setLight(const pxr::SdfPath& id, const light::Light& lamp) {
    const std::lock_guard<std::mutex> held(guard_);
    lights_[id] = lamp;
}

uint32_t Engine::categoryBit(const std::string& name) {
    if (name.empty()) {
        return light::kLightUnlinked;
    }
    const auto found = categoryBits_.find(name);
    if (found != categoryBits_.end()) {
        return found->second;
    }
    if (categoryBits_.size() >= 64) {
        lrt::log::warn("hdLrt: more than 64 light linking categories; '{}' links to nothing", name);
        return light::kLightUnlinked;
    }
    const uint32_t bit = static_cast<uint32_t>(categoryBits_.size());
    categoryBits_.emplace(name, bit);
    return bit;
}

uint64_t Engine::categoryMask(const std::vector<pxr::TfToken>& names) {
    uint64_t mask = 0;
    for (const pxr::TfToken& name : names) {
        const uint32_t bit = categoryBit(name.GetString());
        if (bit != light::kLightUnlinked) {
            mask |= uint64_t{1} << bit;
        }
    }
    return mask;
}

void Engine::setLightSamples(uint32_t samples) { lightSamples_.store(std::max(samples, 1u)); }

void Engine::setChooseLights(bool choose) { chooseLights_.store(choose); }

void Engine::setPathSamples(uint32_t samples) { pathSamples_.store(std::max(samples, 1u)); }

void Engine::setPathBounces(uint32_t bounces) { pathBounces_.store(bounces); }

void Engine::removeLight(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    lights_.erase(id);
}

void Engine::removeMaterial(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    materials_.erase(id);
    materialsChanged_ = true;
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
    for (auto& [id, entry] : materials_) {
        if (!entry.pending) {
            continue;
        }
        entry.pending = false;
        entry.compiled.reset();
        materialsChanged_ = true;
        ++uploaded;
        if (!entry.document) {
            continue;
        }
        if (!compiler_ && !compilerFailed_) {
            std::vector<std::filesystem::path> shaders;
            for (const std::string& path : device_->shaderSearchPaths()) {
                shaders.emplace_back(path);
            }
            std::vector<std::filesystem::path> sources;
            for (const MaterialX::FilePath& path : pxr::HdMtlxSearchPaths()) {
                sources.emplace_back(path.asString());
                sources.emplace_back(std::filesystem::path(path.asString()).parent_path());
            }
            auto made = material::MaterialCompiler::create(pxr::HdMtlxStdLibraries(), sources, shaders);
            if (made) {
                compiler_ = std::move(*made);
            } else {
                compilerFailed_ = true;
                log::warn("hdLrt: materials show displayColor: {}", made.error().toString());
            }
        }
        if (!compiler_) {
            continue;
        }
        auto compiled = compiler_->compileDocument(entry.document);
        if (compiled) {
            entry.compiled = std::move(*compiled);
        } else {
            log::warn("hdLrt: material {}: {}", id.GetString(), compiled.error().toString());
        }
    }
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
        for (const MeshSubset& subset : a.subsets) {
            input.subsets.emplace_back(subset.faces.cdata(), subset.faces.size());
        }
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

Result<std::optional<scene::Bounds>> Engine::bounds() {
    std::optional<scene::Bounds> all;
    const auto grow = [&](const scene::Bounds& box, const render::Mat4& toWorld) {
        // A box through an affine map, as scene_bounds.slang does per instance:
        // one prim's matrix and its cloud's box, not the cloud.
        scene::Bounds moved;
        for (int r = 0; r < 3; ++r) {
            double centre = toWorld.at(r, 3);
            double extent = 0.0;
            for (int c = 0; c < 3; ++c) {
                const double mid = 0.5 * (double(box.min[size_t(c)]) + double(box.max[size_t(c)]));
                const double half = 0.5 * (double(box.max[size_t(c)]) - double(box.min[size_t(c)]));
                centre += toWorld.at(r, c) * mid;
                extent += std::abs(toWorld.at(r, c)) * half;
            }
            moved.min[size_t(r)] = static_cast<float>(centre - extent);
            moved.max[size_t(r)] = static_cast<float>(centre + extent);
        }
        if (!all) {
            all = moved;
            return;
        }
        for (size_t k = 0; k < 3; ++k) {
            all->min[k] = std::min(all->min[k], moved.min[k]);
            all->max[k] = std::max(all->max[k], moved.max[k]);
        }
    };
    {
        const std::lock_guard<std::mutex> held(guard_);
        for (const auto& [id, entry] : splats_) {
            if (!entry.visible) {
                continue;
            }
            if (entry.gpu != nullptr) {
                grow(entry.gpu->bounds, entry.objectToWorld);
            } else if (entry.lodCloud != nullptr) {
                grow(entry.lodCloud->splats.bounds, entry.objectToWorld);
            } else if (entry.pool != nullptr) {
                grow(entry.pool->cloud().splats.bounds, entry.objectToWorld);
            }
        }
        for (const auto& [id, entry] : points_) {
            if (entry.visible && entry.gpu != nullptr) {
                grow(entry.gpu->bounds, entry.objectToWorld);
            }
        }
    }
    if (scene_.has_value()) {
        auto meshes = scene_->worldBounds();
        if (!meshes) return std::move(meshes).error();
        if (meshes->has_value()) {
            grow(**meshes, render::Mat4::identity());
        }
    }
    return all;
}

Result<void> Engine::prepareMaterials(const std::vector<std::string>& aovPrimvars) {
    if (!scene_.has_value()) {
        auto scene = world::GpuScene::create(*library_);
        if (!scene) return std::move(scene).error();
        scene_.emplace(std::move(*scene));
    }
    if (!textures_) {
        auto made = material::TextureStore::create(*library_);
        if (!made) return std::move(made).error();
        textures_ = std::move(*made);
    }
    if (!materialPrograms_.has_value()) {
        auto made = technique::MaterialPrograms::create(*library_);
        if (!made) return std::move(made).error();
        materialPrograms_.emplace(std::move(*made));
    }
    if (!materialShading_.has_value()) {
        auto made = technique::MaterialShading::create(*library_);
        if (!made) return std::move(made).error();
        materialShading_.emplace(std::move(*made));
    }
    // Primvar slots: the AOVs' and every primvar a material reads.
    std::vector<std::string> names = aovPrimvars;
    const auto fixed = [](const std::string& name) {
        return name == "displayColor" || name == "displayOpacity" || name == "normals" || name == "st";
    };
    for (const auto& [id, entry] : materials_) {
        if (!entry.compiled) {
            continue;
        }
        for (const material::MaterialSlot& slot : entry.compiled->slots) {
            if (slot.kind == material::MaterialSlot::Kind::Primvar && !fixed(slot.name) &&
                std::find(names.begin(), names.end(), slot.name) == names.end()) {
                names.push_back(slot.name);
            }
        }
    }
    scene_->setExtraPrimvarSlots(names);
    if (!materialsChanged_ && names == materialSlotNames_ && materialRecords_.valid()) {
        return ok();
    }
    // One module per distinct structure; a row and blob words per material.
    std::vector<material::CompiledMaterial> modules;
    std::vector<technique::MaterialRecord> rows{{0, 0, 0, 0}};
    std::vector<float> blob;
    materialRows_.clear();
    materialCutouts_ = false;
    for (const auto& [id, entry] : materials_) {
        if (!entry.compiled) {
            continue;
        }
        uint32_t function = 0;
        for (size_t k = 0; k < modules.size(); ++k) {
            if (modules[k].module == entry.compiled->module) {
                function = static_cast<uint32_t>(k + 1);
            }
        }
        if (function == 0) {
            modules.push_back(*entry.compiled);
            function = static_cast<uint32_t>(modules.size());
        }
        const std::vector<float> words = material::MaterialCompiler::parameters(
            *entry.compiled, *textures_, [&](const std::string& name) { return scene_->slotOf(name); });
        materialRows_[id] = static_cast<uint32_t>(rows.size());
        const uint32_t flags = entry.cutout ? technique::kMaterialCutout : 0u;
        materialCutouts_ = materialCutouts_ || entry.cutout;
        rows.push_back({function, static_cast<uint32_t>(blob.size()), flags, 0});
        blob.insert(blob.end(), words.begin(), words.end());
    }
    if (blob.empty()) {
        blob.push_back(0.0F);
    }
    if (auto loaded = textures_->commit(); !loaded) {
        return std::move(loaded).error();
    }
    auto records = gpu::Buffer::fromSpan<technique::MaterialRecord>(*device_, rows, "materials.records");
    if (!records) return std::move(records).error();
    auto words = gpu::Buffer::fromSpan<float>(*device_, blob, "materials.blob");
    if (!words) return std::move(words).error();
    materialRecords_ = std::move(*records);
    materialBlob_ = std::move(*words);
    LRT_TRY(materialPrograms_->setModules(modules));
    LRT_TRY(materialShading_->setPrograms(*materialPrograms_));
    materialsChanged_ = false;
    materialSlotNames_ = std::move(names);
    return ok();
}

AovView Engine::aovView(const render::RenderTargets& targets, AovSource aov) const {
    AovView view;
    view.ids = aov.kind == AovKind::PrimId || aov.kind == AovKind::InstanceId || aov.kind == AovKind::ElementId;
    switch (aov.kind) {
    case AovKind::Colour: view.buffer = &targets.colour; return view;
    case AovKind::Depth: view.buffer = &targets.depth; view.source = 1; return view;
    default: break;
    }
    if (!aovsValid_ || aovs_.width != targets.width || aovs_.height != targets.height ||
        (aov.kind == AovKind::Primvar && aov.primvar >= aovs_.primvarSlots)) {
        return view;   // nothing a mesh drew
    }
    switch (aov.kind) {
    case AovKind::PrimId: view.buffer = &aovs_.ids; view.source = 2; view.stride = 3; view.offset = 0; break;
    case AovKind::InstanceId: view.buffer = &aovs_.ids; view.source = 2; view.stride = 3; view.offset = 1; break;
    case AovKind::ElementId: view.buffer = &aovs_.ids; view.source = 2; view.stride = 3; view.offset = 2; break;
    case AovKind::EyeNormal: view.buffer = &aovs_.eyeNormals; break;
    case AovKind::WorldNormal: view.buffer = &aovs_.worldNormals; break;
    default: view.buffer = &aovs_.primvars; view.stride = aovs_.primvarSlots; view.offset = aov.primvar; break;
    }
    return view;
}

Result<void> Engine::writeAov(const render::RenderTargets& targets, AovSource aov, const AovLayout& layout,
                              const double* projection, std::span<uint8_t> into) {
    const uint64_t bytes = uint64_t{targets.width} * targets.height * layout.channels * layout.componentBytes;
    if (into.size() < bytes || layout.channels == 0 || layout.channels > 4 ||
        (layout.componentBytes != 1 && layout.componentBytes != 2 && layout.componentBytes != 4)) {
        return Error(ErrorCode::InvalidArgument, "a render buffer the engine cannot fill");
    }
    const AovView view = aovView(targets, aov);
    if (view.buffer == nullptr) {
        // Nothing a mesh drew: the clear value, -1 for ids and 0 otherwise.
        std::fill(into.begin(), into.begin() + static_cast<std::ptrdiff_t>(bytes), static_cast<uint8_t>(view.ids ? 0xFF : 0));
        return ok();
    }
    const gpu::Buffer* source = view.buffer;
    const uint32_t kind = view.source;
    const uint32_t stride = view.stride;
    const uint32_t offset = view.offset;
    if (!source->valid()) {
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
        cursor["colour"].setBinding(kind == 0 ? source->rhi() : placeholder->rhi());
        cursor["depth"].setBinding(kind == 1 ? source->rhi() : placeholder->rhi());
        cursor["idSource"].setBinding(kind == 2 ? source->rhi() : placeholder->rhi());
        cursor["words"].setBinding(out->rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(targets.width);
        p["height"].setData(targets.height);
        p["words"].setData(words);
        p["source"].setData(kind);
        p["stride"].setData(stride);
        p["offset"].setData(offset);
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
                            const pxr::TfTokenVector* renderTags, const AovRequest& aovRequest,
                            MeshVisibility visibility) {
    lastTargets_ = &targets;
    aovsValid_ = false;
    {
        bool anyMesh = false;
        {
            const std::lock_guard<std::mutex> held(guard_);
            anyMesh = !meshes_.empty();
        }
        if (anyMesh) {
            LRT_TRY(prepareMaterials(aovRequest.primvars));
        }
    }
    const auto rowOf = [&](const pxr::SdfPath& material) -> uint32_t {
        const auto found = materialRows_.find(material);
        return found != materialRows_.end() ? found->second : 0u;
    };
    const auto subsetRowsOf = [&](const MeshEntry& entry) {
        std::vector<uint32_t> rows;
        rows.reserve(entry.subsetMaterials.size());
        for (const pxr::SdfPath& material : entry.subsetMaterials) {
            rows.push_back(rowOf(material));
        }
        return rows;
    };
    std::vector<light::Light> lamps;
    std::vector<world::MeshInstance> meshInstances;
    std::vector<world::InstanceSet> meshSets;
    std::vector<render::SplatInstance> splats;
    std::vector<render::PointInstance> points;
    std::vector<lod::LodInstance> cuts;
    std::vector<lod::StreamingPool*> poolOf;   // per cut: its pool, if streamed
    {
        const std::lock_guard<std::mutex> held(guard_);
        lamps.reserve(lights_.size());
        for (const auto& [id, lamp] : lights_) {
            lamps.push_back(lamp);
            lamps.back().lightCategory = categoryBit(lamp.lightLink);
            lamps.back().shadowCategory = categoryBit(lamp.shadowLink);
        }
        for (const auto& [id, entry] : splats_) {
            if (!entry.visible) {
                continue;
            }
            if (entry.gpu != nullptr) {
                splats.push_back({entry.gpu.get(), entry.objectToWorld, entry.edit, entry.relight,
                                  categoryMask(entry.categories)});
            }
            const lod::LodCloud* cloud = entry.pool != nullptr ? &entry.pool->cloud() : entry.lodCloud.get();
            if (cloud == nullptr) {
                continue;
            }
            if (technique == Technique::RayTraced) {
                // A cut changes every frame, and the ray tracer would rebuild
                // every frame: it draws the whole cloud, when it is whole.
                if (entry.lodCloud != nullptr) {
                    splats.push_back({&entry.lodCloud->splats, entry.objectToWorld, entry.edit, entry.relight,
                                      categoryMask(entry.categories)});
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
                set.material = rowOf(entry.look.material);
                set.subsetMaterials = subsetRowsOf(entry);
                set.categories = categoryMask(entry.look.categories);
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
            instance.material = rowOf(entry.look.material);
            instance.subsetMaterials = subsetRowsOf(entry);
            instance.categories = categoryMask(entry.look.categories);
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
    // Opaque layers first -- meshes, points -- then splats blended over them.
    const bool drawMeshes = !meshInstances.empty() || !meshSets.empty();
    const gpu::Caps& caps = device_->caps();
    // A frame of nothing but splats is GaussianRayTracer's, and it writes the
    // whole image: there is no layer to compose under it, so it returns here.
    // With meshes in the frame the traced technique means something else --
    // the surfaces are path traced below and the splats composed over them by
    // the rasteriser, because the tracer takes no `under` layer. Splats inside
    // the rays is still to be written (docs/decisions.md, M6).
    if (technique == Technique::RayTraced && !drawMeshes) {
        if (!points.empty()) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                log::warn("hdLrt: a traced frame of splats alone leaves the points out");
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
    if (visibility == MeshVisibility::Automatic) {
        // Rays first: a draw costs the host a few microseconds to record, and
        // Kitchen_set's 1800 of them outweigh a frame of rays at any size
        // measured (docs/decisions.md, M2).
        visibility = caps.rayQuery && caps.accelerationStructure ? MeshVisibility::Rays
                     : caps.rasterization                         ? MeshVisibility::Raster
                                                                  : MeshVisibility::Bvh;
    }
    if (drawMeshes && visibility == MeshVisibility::Raster && !caps.rasterization) {
        return Error(ErrorCode::Unsupported, "mesh visibility by raster: the device does not rasterise");
    }
    if (drawMeshes && visibility == MeshVisibility::Rays && !(caps.rayQuery && caps.accelerationStructure)) {
        return Error(ErrorCode::Unsupported, "mesh visibility by rays: the device has no ray queries");
    }
    const bool meshLayer = drawMeshes;
    const bool pathTracing = meshLayer && technique == Technique::RayTraced;
    if (meshLayer) {
        LRT_TRY(scene_->update(meshInstances, projection, meshSets));
        // The frame's lights, and what a shadow ray traces against: rays
        // shadow whatever route found the visibility, so the structure is
        // built even where the rasteriser drew. All of this uploads and
        // builds -- each submitting a batch of its own -- so it happens
        // before the frame's batch opens: a submit inside a batch that has
        // already recorded work releases what that work still refers to.
        if (!lightTable_.has_value()) {
            auto made = light::LightTable::create(*device_);
            if (!made) return std::move(made).error();
            lightTable_.emplace(std::move(*made));
        }
        // A dome's image goes through the same texture table the materials
        // sample, so it is requested here and committed before the frame's
        // records are uploaded.
        if (!textures_) {
            auto made = material::TextureStore::create(*library_);
            if (!made) return std::move(made).error();
            textures_ = std::move(*made);
        }
        bool domeTextures = false;
        for (light::Light& lamp : lamps) {
            if (lamp.texture.empty()) {
                continue;
            }
            lamp.textureId = textures_->request(lamp.texture, material::ColourSpace::Auto);
            // Lat-long: around in u, clamped at the poles.
            lamp.sampler = textures_->sampler(material::Wrap::Repeat, material::Wrap::Clamp);
            domeTextures = true;
        }
        if (domeTextures) {
            if (auto loaded = textures_->commit(); !loaded) {
                return std::move(loaded).error();
            }
        }
        LRT_TRY(lightTable_->set(lamps));
        // A path traced surface needs a structure whatever the lights do --
        // its bounce is a ray -- while shading needs one only where a light
        // casts a shadow. The same structure serves both, and a path traced
        // frame without it would trace against nothing and never know.
        bool raysWanted = false;
        if ((lightTable_->anyShadow() || pathTracing) && caps.rayQuery && caps.accelerationStructure) {
            if (!rayTracingScene_.has_value()) {
                auto accel = world::RayTracingScene::create(*library_);
                if (!accel) return std::move(accel).error();
                rayTracingScene_.emplace(std::move(*accel));
            }
            if (visibility != MeshVisibility::Rays) {
                LRT_TRY(rayTracingScene_->build(*scene_));   // the rays route builds it in the pass below
            }
            raysWanted = true;
        }
        // What a material needs wherever it is evaluated: shading always,
        // visibility only where a material cuts its samples away.
        technique::MaterialFrame frame;
        frame.programs = &*materialPrograms_;
        frame.scene = &*scene_;
        frame.records = &materialRecords_;
        frame.blob = &materialBlob_;
        frame.textures = textures_.get();
        frame.lights = &*lightTable_;
        frame.samples = lightSamples_.load();
        frame.chooseLights = chooseLights_.load();
        const technique::MaterialFrame* cutouts = materialCutouts_ ? &frame : nullptr;
            gpu::CommandBatch batch(*device_);
        switch (visibility) {
        case MeshVisibility::Automatic:
        case MeshVisibility::Raster:
            if (!visibilityRaster_.has_value()) {
                auto made = technique::VisibilityRaster::create(*library_);
                if (!made) return std::move(made).error();
                visibilityRaster_.emplace(std::move(*made));
            }
            LRT_TRY(visibilityRaster_->render(batch, *scene_, projection, settings.width, settings.height,
                                              visibility_, cutouts));
            break;
        case MeshVisibility::Rays:
            if (!visibilityTrace_.has_value()) {
                auto accel = world::RayTracingScene::create(*library_);
                if (!accel) return std::move(accel).error();
                rayTracingScene_.emplace(std::move(*accel));
                auto made = technique::VisibilityTrace::create(*library_);
                if (!made) return std::move(made).error();
                visibilityTrace_.emplace(std::move(*made));
            }
            LRT_TRY(rayTracingScene_->build(*scene_));
            LRT_TRY(visibilityTrace_->render(batch, *rayTracingScene_, projection, settings.width,
                                             settings.height, visibility_, cutouts));
            break;
        case MeshVisibility::Bvh:
            if (!visibilityBvh_.has_value()) {
                auto bvh = world::BvhScene::create(*library_);
                if (!bvh) return std::move(bvh).error();
                bvhScene_.emplace(std::move(*bvh));
                auto made = technique::VisibilityBvh::create(*library_);
                if (!made) return std::move(made).error();
                visibilityBvh_.emplace(std::move(*made));
            }
            LRT_TRY(bvhScene_->build(*scene_));
            LRT_TRY(visibilityBvh_->render(batch, *scene_, *bvhScene_, projection, settings.width,
                                           settings.height, visibility_, cutouts));
            break;
        }
        // Read after the visibility pass, never before it: the rays route
        // rebuilds this structure there, and the one it replaces is released
        // with it -- taking the pointer earlier left shading tracing against
        // a freed structure.
        frame.shadows = raysWanted ? rayTracingScene_->topLevel() : nullptr;
        if (pathTracing) {
            if (!pathTracer_.has_value()) {
                auto made = technique::PathTracer::create(*library_);
                if (!made) return std::move(made).error();
                pathTracer_.emplace(std::move(*made));
            }
            LRT_TRY(pathTracer_->setPrograms(*materialPrograms_));
            technique::PathSettings paths;
            paths.samples = pathSamples_.load();
            paths.bounces = pathBounces_.load();
            // A frame of its own, not a sample of the last one: the camera and
            // the scene may both have moved. Progressive accumulation is the
            // render thread's to ask for (M6, not done).
            paths.seed = pathSeed_++ * 7919u;
            LRT_TRY(pathTracer_->trace(batch, visibility_, projection, frame, paths, meshLayer_));
        } else {
            LRT_TRY(materialShading_->shade(batch, visibility_, projection, frame, meshLayer_));
        }
        if (aovRequest.ids || aovRequest.normals || !aovRequest.primvars.empty()) {
            if (!aovShading_.has_value()) {
                auto made = technique::AovShading::create(*library_);
                if (!made) return std::move(made).error();
                aovShading_.emplace(std::move(*made));
            }
            std::vector<uint32_t> slots;
            for (const std::string& name : aovRequest.primvars) {
                slots.push_back(scene_->slotOf(name));
            }
            LRT_TRY(aovShading_->shade(batch, *scene_, visibility_, projection, slots, aovs_));
            aovsValid_ = true;
        }
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
    // What a splat relights from, where its prim asked to be relit
    // (LrtSplatLightingAPI). Handed over as buffers and counts, since render
    // sits below light in the module order and cannot name its types.
    render::SplatLights splatLights;
    if (lightTable_.has_value() && lightTable_->count() > 0) {
        splatLights.records = &lightTable_->records();
        splatLights.count = lightTable_->count();
        splatLights.power = lightTable_->power();
    }
    if (under != nullptr) {
        LRT_TRY(rasterizer_->render(projection, splats, settings, targets, {}, under, &splatLights));
    } else {
        LRT_TRY(rasterizer_->render(projection, splats, settings, targets, points, nullptr, &splatLights));
    }
    LRT_TRY(paintDomes(projection, settings.width, settings.height, targets));
    return ok();
}

Result<void> Engine::paintDomes(const render::Projection& projection, uint32_t width, uint32_t height,
                                render::RenderTargets& targets) {
    if (!lightTable_.has_value() || !lightTable_->anyDome() || !targets.colour.valid()) {
        return ok();
    }
    if (!domeBackground_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "lrt/technique/dome_background", "domeBackground");
        if (!made) return std::move(made).error();
        domeBackground_.emplace(std::move(*made));
    }
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    gpu::CommandBatch batch(*device_);
    domeBackground_->dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
        lightTable_->bind(cursor);
        // A dome reads its image through the same table the materials sample,
        // so the background pass binds it too: without it every dome is the
        // white a missing file falls back to.
        if (textures_) {
            textures_->bind(cursor["gTextures"]);
        }
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["depth"].setBinding(targets.depth.rhi());
        technique::setCamera(cursor["camera"], projection, width, height);
        static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                   "v12", "v13", "v20", "v21", "v22", "v23"};
        for (size_t k = 0; k < 12; ++k) {
            cursor["background"][kNames[k]].setData(toWorld[k]);
        }
    });
    LRT_TRY(batch.submit(true));
    return ok();
}

}   // namespace lrt::usd
