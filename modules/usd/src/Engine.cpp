// Copyright (c) 2026 lucabRTrender contributors.
#include "Engine.h"
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/matrix3f.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>

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

namespace {

/// An instancer chain composed on the device at the frame, and -- where an
/// instancer in it moves under the shutter -- at the samples about it: each
/// level from its own samples where it has them, the frame's arrays where it
/// does not. `start` and `end` are left empty when nothing in the chain moves;
/// the times are the first moving level's.
Result<void> composeChains(world::Instancing& instancing, const std::map<pxr::SdfPath, InstancerEntry>& instancers,
                           const std::vector<InstancerLink>& links, world::InstanceChain& chain,
                           world::InstanceChain& start, world::InstanceChain& end, double& timeStart,
                           double& timeEnd) {
    enum class At { Frame, Start, End };
    bool anyMoves = false;
    const auto compose = [&](At at) -> Result<world::InstanceChain> {
        std::vector<world::InstancerLevel> levels;
        for (const InstancerLink& link : links) {
            const InstancerArrays& a = instancers.at(link.instancer).arrays;
            const InstancerSample* sample = at == At::Start && a.start.has_value() ? &*a.start
                                            : at == At::End && a.end.has_value()   ? &*a.end
                                                                                    : nullptr;
            if (at == At::Frame && a.start.has_value() && a.end.has_value() && !anyMoves) {
                anyMoves = true;
                timeStart = a.timeStart;
                timeEnd = a.timeEnd;
            }
            world::InstancerLevel level;
            level.indices = std::span<const int32_t>(link.indices.cdata(), link.indices.size());
            level.translations = streamOf(sample != nullptr ? sample->translations : a.translations);
            level.rotations = streamOf(sample != nullptr ? sample->rotations : a.rotations);
            level.scales = streamOf(sample != nullptr ? sample->scales : a.scales);
            level.transforms = streamOf(sample != nullptr ? sample->transforms : a.transforms);
            level.instancerTransform = sample != nullptr ? sample->instancerTransform : a.instancerTransform;
            levels.push_back(level);
        }
        return instancing.compose(levels);
    };
    auto frame = compose(At::Frame);
    if (!frame) return std::move(frame).error();
    chain = std::move(*frame);
    start = {};
    end = {};
    if (anyMoves) {
        auto atStart = compose(At::Start);
        if (!atStart) return std::move(atStart).error();
        auto atEnd = compose(At::End);
        if (!atEnd) return std::move(atEnd).error();
        if (atStart->count == chain.count && atEnd->count == chain.count) {
            start = std::move(*atStart);
            end = std::move(*atEnd);
        }
    }
    return ok();
}

std::array<float, 16> matrixOf(const pxr::VtValue& value) {
    std::array<float, 16> out{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (value.IsHolding<pxr::GfMatrix4f>()) {
        const pxr::GfMatrix4f& m = value.UncheckedGet<pxr::GfMatrix4f>();
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out[static_cast<size_t>(r * 4 + c)] = m[r][c];
    } else if (value.IsHolding<pxr::GfMatrix4d>()) {
        const pxr::GfMatrix4d& m = value.UncheckedGet<pxr::GfMatrix4d>();
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out[static_cast<size_t>(r * 4 + c)] = static_cast<float>(m[r][c]);
    }
    return out;
}

template <typename Array>
std::span<const std::byte> bytesOf(const pxr::VtValue& value) {
    if (!value.IsHolding<Array>()) {
        return {};
    }
    const Array& array = value.UncheckedGet<Array>();
    return {reinterpret_cast<const std::byte*>(array.cdata()), array.size() * sizeof(typename Array::value_type)};
}

/// The skinner's input over Hydra's values: byte views, nothing converted
/// but the three matrices (and a double matrix to float, as Storm's kernel
/// casts them).
geom::SkinningInput skinningInputOf(const SkinningArrays& s, const scene::FloatStream& rest) {
    geom::SkinningInput in;
    in.points = static_cast<uint32_t>(rest.values() / 3);
    in.restPoints = rest.bytes;
    in.blendShapeOffsets = bytesOf<pxr::VtVec4fArray>(s.blendShapeOffsets);
    in.blendShapeOffsetRanges = bytesOf<pxr::VtVec2iArray>(s.blendShapeOffsetRanges);
    in.blendShapeWeights = bytesOf<pxr::VtFloatArray>(s.blendShapeWeights);
    in.influences = bytesOf<pxr::VtVec2fArray>(s.influences);
    in.numInfluencesPerPoint = static_cast<uint32_t>(std::max(s.numInfluencesPerComponent, 0));
    in.constantInfluences = s.hasConstantInfluences;
    in.method = s.dualQuaternion ? geom::SkinningMethod::DualQuaternion : geom::SkinningMethod::LinearBlend;
    in.skinningXforms = bytesOf<pxr::VtMatrix4fArray>(s.skinningXforms);
    in.skinningDualQuats = bytesOf<pxr::VtVec4fArray>(s.skinningDualQuats);
    if (in.skinningDualQuats.empty()) {
        in.skinningDualQuats = bytesOf<pxr::VtQuatfArray>(s.skinningDualQuats);
    }
    in.skinningScaleXforms = bytesOf<pxr::VtMatrix3fArray>(s.skinningScaleXforms);
    in.geomBindXform = matrixOf(s.geomBindXform);
    in.skelLocalToWorld = matrixOf(s.skelLocalToWorld);
    in.primWorldToLocal = matrixOf(s.primWorldToLocal);
    return in;
}

}   // namespace

void Engine::setCurves(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                       std::optional<CurveArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                       std::optional<MeshLook> look) {
    const std::lock_guard<std::mutex> held(guard_);
    MeshEntry& entry = meshes_[id];
    entry.primId = static_cast<uint32_t>(primId);
    entry.renderTag = renderTag;
    if (arrays.has_value()) {
        entry.pendingCurves = std::move(arrays);
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
    revision_.fetch_add(1);
}

void Engine::setMesh(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                     std::optional<MeshArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                     std::optional<MeshLook> look, std::optional<std::vector<InstancerLink>> instancing,
                     std::optional<MeshTransforms> shutter) {
    const std::lock_guard<std::mutex> held(guard_);
    MeshEntry& entry = meshes_[id];
    if (shutter.has_value()) {
        entry.shutter = *shutter;
    }
    if (arrays.has_value()) {
        entry.pendingStart = arrays->pointsStart;
        entry.pendingEnd = arrays->pointsEnd;
        entry.pointsTimeStart = arrays->pointsTimeStart;
        entry.pointsTimeEnd = arrays->pointsTimeEnd;
    }
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

void Engine::setLight(const pxr::SdfPath& id, const light::Light& lamp, std::vector<InstancerLink> instancing) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    LightEntry& entry = lights_[id];
    entry.lamp = lamp;
    // Sync gives the chain whole each time; a change is any link differing.
    bool same = entry.instancing.size() == instancing.size();
    for (size_t i = 0; same && i < instancing.size(); ++i) {
        same = entry.instancing[i].instancer == instancing[i].instancer && entry.instancing[i].indices == instancing[i].indices;
    }
    if (!same) {
        entry.instancing = std::move(instancing);
        entry.chainDirty = true;
    }
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

// These four change what a path finds, so a mean gathered under the old ones
// is not the same mean; each raises the revision and the accumulation starts
// again. setPathTotal does not: moving the finish line leaves what has been
// gathered still valid.
void Engine::setLightSamples(uint32_t samples) {
    if (lightSamples_.exchange(std::max(samples, 1u)) != std::max(samples, 1u)) {
        revision_.fetch_add(1);
    }
}

void Engine::setChooseLights(bool choose) {
    if (chooseLights_.exchange(choose) != choose) {
        revision_.fetch_add(1);
    }
}

void Engine::setPathSamples(uint32_t samples) {
    if (pathSamples_.exchange(std::max(samples, 1u)) != std::max(samples, 1u)) {
        revision_.fetch_add(1);
    }
}

void Engine::setPathBounces(uint32_t bounces) {
    if (pathBounces_.exchange(bounces) != bounces) {
        revision_.fetch_add(1);
    }
}

void Engine::setShutter(double open, double close) {
    const bool changed = shutterOpen_.exchange(open) != open || shutterClose_.exchange(close) != close;
    if (changed) {
        revision_.fetch_add(1);
        shutterSettle_.store(2);   // this frame, and the one that resamples
    }
}

void Engine::setMotionBuckets(uint32_t buckets) {
    const uint32_t clamped = std::min(std::max(buckets, 1u), 8u);
    if (motionBuckets_.exchange(clamped) != clamped) {
        revision_.fetch_add(1);
    }
}

void Engine::setPathTotal(uint32_t total) { pathTotal_.store(std::max(total, 1u)); }

void Engine::setDenoise(bool denoise) { denoise_.store(denoise); }

void Engine::setPathAdaptive(bool adaptive) { pathAdaptive_.store(adaptive); }
void Engine::setPathMis(bool mis) { pathMis_.store(mis); }

void Engine::setPathError(float error) { pathError_.store(std::max(error, 1.0e-4F)); }

uint32_t Engine::pathAccumulated() const noexcept {
    return pathTracer_.has_value() && pathState_.traced ? pathTracer_->accumulated() : 0;
}

bool Engine::pathConverged() const noexcept {
    if (shutterSettle_.load() > 0) {
        return false;   // the prims are still being resampled about it
    }
    if (!pathState_.traced) {
        return true;   // nothing being gathered
    }
    if (pathAccumulated() >= pathTotal_.load()) {
        return true;
    }
    return pathState_.adaptive && pathProgress_.covered > 0 && pathProgress_.converged == pathProgress_.covered;
}

std::vector<CoordSysBinding> Engine::coordSysOf(const pxr::SdfPath& id) const {
    const std::lock_guard<std::mutex> held(guard_);
    const auto found = meshes_.find(id);
    return found != meshes_.end() ? found->second.look.coordSys : std::vector<CoordSysBinding>{};
}

uint64_t Engine::meshGeneration() const noexcept {
    return scene_.has_value() ? scene_->generation() : 0;
}

uint64_t Engine::meshPositionsRevision() const noexcept {
    return scene_.has_value() ? scene_->positionsRevision() : 0;
}

void Engine::removeLight(const pxr::SdfPath& id) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    lights_.erase(id);
}

void Engine::removeMaterial(const pxr::SdfPath& id) {
    const std::lock_guard<std::mutex> held(guard_);
    materials_.erase(id);
    materialsChanged_ = true;
}

void Engine::setVolume(const pxr::SdfPath& id, VolumeArrays arrays) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumes_[id] = std::move(arrays);
    ++volumesVersion_;
}

void Engine::removeVolume(const pxr::SdfPath& id) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumes_.erase(id);
    ++volumesVersion_;
}

void Engine::setVolumeField(const pxr::SdfPath& id, VolumeFieldAsset asset) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumeFields_[id] = std::move(asset);
    ++volumesVersion_;
}

void Engine::removeVolumeField(const pxr::SdfPath& id) {
    revision_.fetch_add(1);
    const std::lock_guard<std::mutex> held(guard_);
    volumeFields_.erase(id);
    ++volumesVersion_;
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
        if (entry.pendingCurves.has_value()) {
            const CurveArrays& c = *entry.pendingCurves;
            geom::CurveInput input;
            input.source = id.GetString();
            input.points = streamOf(c.points);
            input.curveVertexCounts = std::span<const int32_t>(c.curveVertexCounts.cdata(), c.curveVertexCounts.size());
            input.curveIndices = std::span<const int32_t>(c.curveIndices.cdata(), c.curveIndices.size());
            if (c.type == pxr::HdTokens->cubic) {
                input.basis = c.basis == pxr::HdTokens->bspline      ? geom::CurveBasis::BSpline
                              : c.basis == pxr::HdTokens->catmullRom ? geom::CurveBasis::CatmullRom
                                                                     : geom::CurveBasis::Bezier;
            } else {
                input.basis = geom::CurveBasis::Linear;
            }
            input.wrap = c.wrap == pxr::HdTokens->periodic ? geom::CurveWrap::Periodic : geom::CurveWrap::Nonperiodic;
            std::vector<float> widths;
            if (c.widths.IsHolding<pxr::VtFloatArray>()) {
                const auto& w = c.widths.UncheckedGet<pxr::VtFloatArray>();
                widths.assign(w.begin(), w.end());
            } else if (c.widths.IsHolding<float>()) {
                widths.push_back(c.widths.UncheckedGet<float>());
            }
            input.widths = widths;
            // HdInterpolation: constant 0, uniform 1, varying 2, vertex 3.
            input.widthInterpolation = c.widthsInterpolation == 1   ? geom::WidthInterpolation::Uniform
                                       : c.widthsInterpolation == 2 ? geom::WidthInterpolation::Varying
                                       : c.widthsInterpolation == 3 ? geom::WidthInterpolation::Vertex
                                                                    : geom::WidthInterpolation::Constant;
            if (widths.size() == 1) {
                input.widthInterpolation = geom::WidthInterpolation::Constant;
                input.width = widths.front();
            }
            std::vector<geom::PrimvarInput> primvars;
            for (const PrimvarArrays& p : c.primvars) {
                geom::PrimvarInput primvar;
                primvar.name = p.name;
                primvar.interpolation = static_cast<geom::Interpolation>(p.interpolation);
                primvar.values = primvarStreamOf(p.values, &primvar.components);
                primvars.push_back(std::move(primvar));
            }
            input.primvars = primvars;
            if (c.topologyChanged || entry.topologyKey == 0) {
                entry.topologyKey = ++nextTopologyKey_;
            }
            input.topology = entry.topologyKey;
            entry.gpu.reset();
            if (!curveBuilder_.has_value()) {
                auto made = geom::CurveBuilder::create(*library_);
                if (!made) return std::move(made).error();
                curveBuilder_.emplace(std::move(*made));
            }
            auto built = curveBuilder_->build(input);
            if (built) {
                entry.gpu = std::make_shared<const geom::GpuMesh>(std::move(built->mesh));
            } else {
                log::warn("hdLrt: {}: {}", id.GetString(), built.error().toString());
            }
            entry.pendingCurves.reset();
            ++uploaded;
            continue;
        }
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
        input.invisibleFaces = std::span<const int32_t>(a.invisibleFaces.cdata(), a.invisibleFaces.size());
        input.leftHanded = a.leftHanded;
        input.smoothNormals = a.smoothNormals;
        // The same key while Hydra says the topology stands, so the scene
        // takes the rebuilt mesh as the old one deformed and refits.
        if (a.topologyChanged || entry.topologyKey == 0) {
            entry.topologyKey = ++nextTopologyKey_;
        }
        input.topology = entry.topologyKey;
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
        // The coordinate systems bound to the prim, as the material's
        // transform nodes read them: each system's transform to world, its
        // rows as three constant float4 primvars ("lrtCoordSys_NAME_0" to
        // "_2"). Values handed over as Hydra gave them; any inverse is the
        // shader's.
        std::vector<std::array<float, 12>> coordSysRows;
        coordSysRows.reserve(entry.look.coordSys.size());
        for (const CoordSysBinding& binding : entry.look.coordSys) {
            coordSysRows.push_back(binding.toWorld.rows3x4());
            const std::array<float, 12>& rows = coordSysRows.back();
            for (uint32_t r = 0; r < 3; ++r) {
                geom::PrimvarInput primvar;
                primvar.name = "lrtCoordSys_" + binding.name + "_" + std::to_string(r);
                primvar.interpolation = geom::Interpolation::Constant;
                primvar.components = 4;
                primvar.values = {std::as_bytes(std::span<const float>(rows.data() + r * 4, 4)), false};
                primvars.push_back(std::move(primvar));
            }
        }
        input.primvars = primvars;
        for (const MeshSubset& subset : a.subsets) {
            input.subsets.emplace_back(subset.faces.cdata(), subset.faces.size());
        }
        entry.gpu.reset();
        // Skinned: the rest points go through the skinner on the device, and
        // the mesh is built from what comes out, under the same topology key
        // -- a deformation of the rest mesh, refit and not rebuilt.
        gpu::Buffer skinned;
        if (a.skinning.has_value() && input.points.values() >= 3) {
            if (!skinner_.has_value()) {
                auto made = geom::Skinner::create(*library_);
                if (!made) return std::move(made).error();
                skinner_.emplace(std::move(*made));
            }
            auto result = skinner_->skin(skinningInputOf(*a.skinning, input.points));
            if (result) {
                skinned = std::move(*result);
                input.devicePositions = &skinned;
                input.devicePoints = static_cast<uint32_t>(input.points.values() / 3);
            } else {
                log::warn("hdLrt: {}: skinning: {}", id.GetString(), result.error().toString());
            }
        }
        // A subdivision surface at a refine level above zero: refined on the
        // device (after the skinning, when there is any) and built from the
        // refined mesh. The refined topology is the mesh's own key.
        std::optional<geom::Refined> refined;
        std::optional<geom::Refined::AsInput> refinedInput;
        const bool subdivides = a.refineLevel > 0 && !a.scheme.IsEmpty() &&
                                a.scheme != pxr::PxOsdOpenSubdivTokens->none && input.points.values() >= 3 &&
                                !a.faceVertexCounts.empty();
        if (subdivides) {
            if (!subdivider_.has_value()) {
                auto made = geom::Subdivider::create(*library_);
                if (!made) return std::move(made).error();
                subdivider_.emplace(std::move(*made));
            }
            geom::SubdivisionInput sub;
            sub.source = input.source;
            sub.points = input.points;
            sub.devicePositions = input.devicePositions;
            sub.devicePoints = input.devicePoints;
            sub.faceVertexCounts = input.faceVertexCounts;
            sub.faceVertexIndices = input.faceVertexIndices;
            sub.holeIndices = input.holeIndices;
            sub.scheme = a.scheme == pxr::PxOsdOpenSubdivTokens->loop       ? geom::SubdivisionScheme::Loop
                         : a.scheme == pxr::PxOsdOpenSubdivTokens->bilinear ? geom::SubdivisionScheme::Bilinear
                                                                             : geom::SubdivisionScheme::CatmullClark;
            sub.levels = static_cast<uint32_t>(std::min(a.refineLevel, 5));
            sub.creaseIndices = std::span<const int32_t>(a.creaseIndices.cdata(), a.creaseIndices.size());
            sub.creaseLengths = std::span<const int32_t>(a.creaseLengths.cdata(), a.creaseLengths.size());
            sub.creaseSharpnesses = std::span<const float>(a.creaseSharpnesses.cdata(), a.creaseSharpnesses.size());
            sub.cornerIndices = std::span<const int32_t>(a.cornerIndices.cdata(), a.cornerIndices.size());
            sub.cornerSharpnesses = std::span<const float>(a.cornerSharpnesses.cdata(), a.cornerSharpnesses.size());
            sub.primvars = input.primvars;
            auto made = subdivider_->refine(sub);
            if (made) {
                refined.emplace(std::move(*made));
                refinedInput.emplace(refined->asMeshInput(input.source, input.topology));
                refinedInput->mesh.invisibleFaces = {};   // a coarse face's flag does not survive refinement yet
                refinedInput->mesh.leftHanded = input.leftHanded;
            } else {
                log::warn("hdLrt: {}: subdivision: {}", id.GetString(), made.error().toString());
            }
        }
        const geom::MeshInput& built_input = refinedInput.has_value() ? refinedInput->mesh : input;
        if (input.points.values() >= 3 && !a.faceVertexCounts.empty()) {
            if (!meshBuilder_.has_value()) {
                auto made = geom::MeshBuilder::create(*library_);
                if (!made) return std::move(made).error();
                meshBuilder_.emplace(std::move(*made));
            }
            auto mesh = meshBuilder_->build(built_input);
            if (mesh) {
                entry.gpu = std::make_shared<const geom::GpuMesh>(std::move(*mesh));
            } else {
                log::warn("hdLrt: {}: {}", id.GetString(), mesh.error().toString());
            }
            // The shutter's meshes: the same topology, the points there.
            entry.gpuStart.reset();
            entry.gpuEnd.reset();
            for (auto [pending, into] : {std::pair{&entry.pendingStart, &entry.gpuStart},
                                         std::pair{&entry.pendingEnd, &entry.gpuEnd}}) {
                if (pending->IsEmpty() || !entry.gpu) {
                    continue;
                }
                geom::MeshInput sample = input;
                sample.points = streamOf(*pending);
                if (sample.points.values() != input.points.values()) {
                    continue;
                }
                auto built = meshBuilder_->build(sample);
                if (built) {
                    *into = std::make_shared<const geom::GpuMesh>(std::move(*built));
                }
            }
        }
        entry.pending.reset();
        entry.pendingStart = pxr::VtValue();
        entry.pendingEnd = pxr::VtValue();
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
    // Anything uploaded is something a path could find: whatever a path traced
    // frame had accumulated was of a scene that no longer exists.
    if (uploaded > 0) {
        revision_.fetch_add(1);
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
    // The path tracer's, written at its first hit: nothing where the frame
    // was not path traced.
    case AovKind::Albedo:
        if (pathAuxValid_ && pathAux_.width == targets.width && pathAux_.height == targets.height) {
            view.buffer = &pathAux_.planes;
        }
        return view;
    case AovKind::ShadingNormal:
        if (pathAuxValid_ && pathAux_.width == targets.width && pathAux_.height == targets.height) {
            view.buffer = &pathAux_.planes;
            view.offset = static_cast<uint32_t>(pathAux_.normalOffset());
        }
        return view;
    // A light group's plane, one of the frame's in one buffer: the offset
    // addresses it at a stride of one.
    case AovKind::LightGroup:
        if (aov.primvar >= lightGroupCount_ || lightGroupPixels_ != uint64_t{targets.width} * targets.height) {
            return view;
        }
        // The raster's shading writes the planes; a path traced frame's means
        // are copied there (gatherLightGroups) before the domes and the
        // exposure reach them.
        if (lightGroupColour_.valid()) {
            view.buffer = &lightGroupColour_;
            view.offset = static_cast<uint32_t>(aov.primvar * lightGroupPixels_);
        }
        return view;
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
    // A frame drawn since the shutter changed: this one draws the prims as
    // they were sampled, the next draws them resampled (docs: the shutter).
    if (const int settling = shutterSettle_.load(); settling > 0) {
        shutterSettle_.store(settling - 1);
    }
    {
        bool anyMesh = false;
        {
            const std::lock_guard<std::mutex> held(guard_);
            // A traced volume draws through the mesh layer, which wants the
            // scene and the material programs even when no mesh is in it.
            anyMesh = !meshes_.empty() || (technique == Technique::RayTraced && !volumes_.empty());
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
    const uint32_t motionBuckets = technique == Technique::RayTraced ? motionBuckets_.load() : 1u;
    const double shutterOpen = shutterOpen_.load();
    const double shutterClose = shutterClose_.load();
    {
        const std::lock_guard<std::mutex> held(guard_);
        lamps.reserve(lights_.size());
        for (auto& [id, entry] : lights_) {
            const light::Light& lamp = entry.lamp;
            lamps.push_back(lamp);
            lamps.back().lightCategory = categoryBit(lamp.lightLink);
            lamps.back().shadowCategory = categoryBit(lamp.shadowLink);
            // Its light group, numbered among the ones this frame asks for
            // (1 + the index; 0 for none, and for a group nobody asked for).
            lamps.back().groupIndex = 0;
            for (size_t g = 0; g < aovRequest.lightGroups.size() && !lamp.group.empty(); ++g) {
                if (aovRequest.lightGroups[g] == lamp.group) {
                    lamps.back().groupIndex = static_cast<uint32_t>(g + 1);
                    break;
                }
            }
            // Under an instancer: the chain composed on the device, as a
            // mesh's, once per change; its rows place the light's copies.
            if (!entry.instancing.empty()) {
                std::vector<uint64_t> versions;
                bool complete = true;
                for (const InstancerLink& link : entry.instancing) {
                    const auto found = instancers_.find(link.instancer);
                    complete = complete && found != instancers_.end();
                    versions.push_back(found != instancers_.end() ? found->second.version : 0);
                }
                if (!complete) {
                    // Its instancer has not arrived: as a mesh, not drawn yet.
                    lamps.pop_back();
                    continue;
                }
                {
                    if (entry.chainDirty || versions != entry.chainVersions) {
                        if (!instancing_.has_value()) {
                            auto made = world::Instancing::create(*library_);
                            if (!made) return std::move(made).error();
                            instancing_.emplace(std::move(*made));
                        }
                        LRT_TRY(composeChains(*instancing_, instancers_, entry.instancing, entry.chain,
                                              entry.chainStart, entry.chainEnd, entry.chainTimeStart,
                                              entry.chainTimeEnd));
                        entry.chainVersions = std::move(versions);
                        entry.chainDirty = false;
                    }
                    lamps.back().instanceRows = &entry.chain.rows;
                    lamps.back().instanceCount = entry.chain.count;
                    if (motionBuckets > 1 && entry.chain.count > 0 && entry.chainStart.count == entry.chain.count &&
                        entry.chainStart.rows.valid() && entry.chainEnd.rows.valid()) {
                        lamps.back().instanceRowsStart = &entry.chainStart.rows;
                        lamps.back().instanceRowsEnd = &entry.chainEnd.rows;
                        lamps.back().instanceTimeStart = static_cast<float>(entry.chainTimeStart);
                        lamps.back().instanceTimeEnd = static_cast<float>(entry.chainTimeEnd);
                        if (entry.chainTimeEnd == entry.chainTimeStart) {
                            lamps.back().instanceTimeStart = static_cast<float>(shutterOpen);
                            lamps.back().instanceTimeEnd = static_cast<float>(shutterClose);
                        }
                    }
                }
            }
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
                    LRT_TRY(composeChains(*instancing_, instancers_, entry.instancing, mutableEntry.chain,
                                          mutableEntry.chainStart, mutableEntry.chainEnd,
                                          mutableEntry.chainTimeStart, mutableEntry.chainTimeEnd));
                    mutableEntry.chainVersions = std::move(versions);
                    mutableEntry.chainDirty = false;
                }
                world::InstanceSet set;
                set.mesh = entry.gpu;
                set.chainRows = entry.chain.rows;
                set.count = entry.chain.count;
                if (motionBuckets > 1 && entry.chainStart.count == entry.chain.count && entry.chain.count > 0 &&
                    entry.chainStart.rows.valid() && entry.chainEnd.rows.valid()) {
                    set.chainRowsStart = entry.chainStart.rows;
                    set.chainRowsEnd = entry.chainEnd.rows;
                    set.timeStart = entry.chainTimeStart;
                    set.timeEnd = entry.chainTimeEnd;
                    if (set.timeEnd == set.timeStart) {
                        set.timeStart = shutterOpen;
                        set.timeEnd = shutterClose;
                    }
                }
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
            if (motionBuckets > 1 && (entry.shutter.start.has_value() || entry.shutter.end.has_value() ||
                                      entry.gpuStart != nullptr || entry.gpuEnd != nullptr)) {
                // The samples' times: the transform's, or the points' where
                // only they move. Where both move at different times, the
                // transform's are taken and the points are placed at them --
                // a compromise, noted in the docs.
                world::MeshMotion motion;
                motion.objectToWorldStart = entry.shutter.start.value_or(entry.objectToWorld);
                motion.objectToWorldEnd = entry.shutter.end.value_or(entry.objectToWorld);
                motion.meshStart = entry.gpuStart;
                motion.meshEnd = entry.gpuEnd;
                const bool transformMoves = entry.shutter.start.has_value() || entry.shutter.end.has_value();
                motion.timeStart = transformMoves ? entry.shutter.timeStart : entry.pointsTimeStart;
                motion.timeEnd = transformMoves ? entry.shutter.timeEnd : entry.pointsTimeEnd;
                if (motion.timeEnd == motion.timeStart) {
                    motion.timeStart = shutterOpen;
                    motion.timeEnd = shutterClose;
                }
                instance.motion = motion;
            }
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
    // Volumes are path traced through the mesh layer, whose visibility then
    // finds nothing and every sample walks its camera ray: a traced frame
    // with volumes has that layer even when it has no mesh.
    bool volumesInFrame = false;
    if (technique == Technique::RayTraced) {
        const std::lock_guard<std::mutex> held(guard_);
        for (const auto& [id, volume] : volumes_) {
            volumesInFrame = volumesInFrame || volume.visible;
        }
    }
    const gpu::Caps& caps = device_->caps();
    // A frame of nothing but splats is GaussianRayTracer's, and it writes the
    // whole image: there is no layer to compose under it, so it returns here.
    // With meshes in the frame the traced technique means something else --
    // the surfaces are path traced below and the splats composed over them by
    // the rasteriser, because the tracer takes no `under` layer. Splats inside
    // the rays is still to be written (docs/decisions.md, M6).
    if (technique == Technique::RayTraced && !drawMeshes && !volumesInFrame) {
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
    const bool meshLayer = drawMeshes || volumesInFrame;
    const bool pathTracing = meshLayer && technique == Technique::RayTraced;
    if (!pathTracing) {
        pathState_.traced = false;
        pathAuxValid_ = false;
    }
    if (meshLayer) {
        // A light that moves cuts the frame into shutter slices as geometry
        // that moves does.
        const bool lightsMove = motionBuckets > 1 && std::any_of(lamps.begin(), lamps.end(), [](const light::Light& l) {
                                    return l.movesUnderShutter();
                                });
        LRT_TRY(scene_->update(meshInstances, projection, meshSets, motionBuckets, shutterOpen, shutterClose,
                               lightsMove));
        // The frame's lights, and what a shadow ray traces against: rays
        // shadow whatever route found the visibility, so the structure is
        // built even where the rasteriser drew. All of this uploads and
        // builds -- each submitting a batch of its own -- so it happens
        // before the frame's batch opens: a submit inside a batch that has
        // already recorded work releases what that work still refers to.
        if (!lightTable_.has_value()) {
            auto made = light::LightTable::create(*library_);
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
            // An IES profile, read the first time its path is seen.
            if (!lamp.iesFile.empty() && !lamp.ies) {
                auto known = iesProfiles_.find(lamp.iesFile);
                if (known != iesProfiles_.end()) {
                    lamp.ies = known->second;
                } else if (!iesFailed_.contains(lamp.iesFile)) {
                    auto read = io::readIes(lamp.iesFile);
                    if (read) {
                        lamp.ies = std::make_shared<const io::IesProfile>(std::move(*read));
                        iesProfiles_[lamp.iesFile] = lamp.ies;
                    } else {
                        iesFailed_.insert(lamp.iesFile);
                        log::warn("hdLrt: light without its IES profile: {}", read.error().toString());
                    }
                }
            }
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
        // The scene's reach, for a dome's or a sun's share of the lights'
        // power: its bounds are a kernel's, read again when the mesh set or
        // its points change.
        if (scene_->generation() != lightRadiusGeneration_ || scene_->positionsRevision() != lightRadiusRevision_) {
            lightRadiusGeneration_ = scene_->generation();
            lightRadiusRevision_ = scene_->positionsRevision();
            auto bounds = scene_->worldBounds();
            if (!bounds) return std::move(bounds).error();
            lightSceneRadius_ = 1.0F;
            if (bounds->has_value()) {
                const scene::Bounds& b = **bounds;
                const float dx = b.max[0] - b.min[0], dy = b.max[1] - b.min[1], dz = b.max[2] - b.min[2];
                lightSceneRadius_ = std::max(0.5F * std::sqrt(dx * dx + dy * dy + dz * dz), 1.0e-3F);
            }
        }
        LRT_TRY(lightTable_->set(lamps, lightSceneRadius_));
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
        // The light groups' planes, one buffer for the frame's groups.
        if (aovRequest.lightGroups.size() > technique::kMaxLightGroups) {
            return Error::make(ErrorCode::InvalidArgument, "{} light groups asked for; at most {}",
                               aovRequest.lightGroups.size(), technique::kMaxLightGroups);
        }
        lightGroupCount_ = static_cast<uint32_t>(aovRequest.lightGroups.size());
        if (lightGroupCount_ > 0) {
            const uint64_t pixels = uint64_t{settings.width} * settings.height;
            const uint64_t bytes = pixels * lightGroupCount_ * 16;
            if (!lightGroupColour_.valid() || lightGroupColour_.bytes() < bytes || lightGroupPixels_ != pixels) {
                gpu::BufferDesc desc;
                desc.bytes = bytes;
                desc.elementBytes = 16;
                desc.label = "lights.groups.colour";
                auto colour = gpu::Buffer::create(*device_, desc);
                if (!colour) return std::move(colour).error();
                lightGroupColour_ = std::move(*colour);
            }
            lightGroupPixels_ = pixels;
            frame.groups = {&lightGroupColour_, lightGroupCount_};
        }
        // The frame's volumes, laid out again whenever a volume or a field
        // changed: each grid read once per file and name.
        if (pathTracing) {
            std::vector<world::VolumeInput> inputs;
            std::vector<std::shared_ptr<const io::NanoGrid>> holding;
            bool rebuild = false;
            {
                const std::lock_guard<std::mutex> held(guard_);
                rebuild = volumesVersion_ != volumesBuilt_;
                if (rebuild) {
                    volumesBuilt_ = volumesVersion_;
                    for (const auto& [id, volume] : volumes_) {
                        if (!volume.visible || volume.field.IsEmpty()) {
                            continue;
                        }
                        const auto field = volumeFields_.find(volume.field);
                        if (field == volumeFields_.end() || field->second.path.empty()) {
                            continue;   // its field has not arrived
                        }
                        const auto key = std::make_pair(field->second.path, field->second.gridName);
                        auto cached = nanoGrids_.find(key);
                        if (cached == nanoGrids_.end()) {
                            auto read = io::readVdbGrid(field->second.path, field->second.gridName);
                            if (!read) {
                                log::warn("hdLrt: volume {}: {}", id.GetString(), read.error().toString());
                            }
                            cached = nanoGrids_
                                         .emplace(key, read ? std::make_shared<const io::NanoGrid>(std::move(*read))
                                                            : std::shared_ptr<const io::NanoGrid>())
                                         .first;
                        }
                        if (cached->second == nullptr) {
                            continue;
                        }
                        world::VolumeInput input;
                        input.grid = cached->second.get();
                        input.objectToWorld = volume.objectToWorld;
                        input.densityScale = volume.densityScale;
                        input.albedo = volume.albedo;
                        input.g = volume.g;
                        inputs.push_back(input);
                        holding.push_back(cached->second);
                    }
                }
            }
            if (rebuild) {
                if (!volumeSet_.has_value()) {
                    auto made = world::VolumeSet::create(*library_);
                    if (!made) return std::move(made).error();
                    volumeSet_.emplace(std::move(*made));
                }
                gpu::CommandBatch volumeBatch(*device_);
                LRT_TRY(volumeSet_->set(volumeBatch, inputs));
                LRT_TRY(volumeBatch.submit(true));
                volumesDrawn_ = static_cast<uint32_t>(inputs.size());
            }
            if (volumesDrawn_ > 0) {
                frame.volumes = &volumeSet_->words();
                frame.volumeCount = volumesDrawn_;
            }
        } else if (!volumesUndrawnSaid_) {
            const std::lock_guard<std::mutex> held(guard_);
            if (!volumes_.empty()) {
                volumesUndrawnSaid_ = true;
                log::info("hdLrt: volumes are drawn by the rt technique; the raster technique draws none");
            }
        }
        // The emitting triangles, for the path tracer's next event estimation:
        // probed and weighed on the device again whenever what they depend on
        // changed -- the scene, its positions, the materials, anything the
        // revision counts. Not through media, whose kernel does not sample them.
        if (pathTracing && frame.volumeCount == 0) {
            const EmissiveKey key{scene_->generation(), scene_->positionsRevision(), revision_.load(),
                                  materialRecords_.rhi(), scene_->instanceCount()};
            if (!emissiveTable_.has_value()) {
                auto made = technique::EmissiveTable::create(*library_);
                if (!made) return std::move(made).error();
                emissiveTable_.emplace(std::move(*made));
            }
            if (!(key == emissiveKey_)) {
                LRT_TRY(emissiveTable_->build(frame, static_cast<uint32_t>(materialRecords_.count())));
                emissiveKey_ = key;
            }
            if (emissiveTable_->totalPower() > 0.0F) {
                frame.emissive = &emissiveTable_->table();
                frame.emissivePower = emissiveTable_->totalPower();
            }
        }
        const technique::MaterialFrame* cutouts = (materialCutouts_ || scene_->anyHidden()) ? &frame : nullptr;
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
            paths.adaptive = pathAdaptive_.load();
            paths.mis = pathMis_.load();
            paths.errorTarget = pathError_.load();
            paths.headlight = frame.lights == nullptr || frame.lights->count() == 0;
            PathState now;
            now.worldToView = projection.worldToView;
            now.focalX = projection.focalX;
            now.lensRadius = projection.lensRadius;
            now.focusDistance = projection.focusDistance;
            now.distortionK1 = projection.distortionK1;
            now.distortionK2 = projection.distortionK2;
            now.focalY = projection.focalY;
            now.centreX = projection.centreX;
            now.centreY = projection.centreY;
            now.nearZ = projection.nearZ;
            now.farZ = projection.farZ;
            now.orthographic = projection.orthographic;
            now.width = settings.width;
            now.height = settings.height;
            now.samples = paths.samples;
            now.bounces = paths.bounces;
            now.adaptive = paths.adaptive;
            now.mis = paths.mis;
            now.cameraMoves = projection.cameraMoves;
            now.cameraStart = projection.viewToWorldStart;
            now.cameraEnd = projection.viewToWorldEnd;
            now.error = paths.errorTarget;
            now.revision = revision_.load();
            if (renderTags != nullptr) {
                // Which purposes are drawn is part of what a path finds.
                uint64_t hash = 1469598103934665603ull;
                for (const pxr::TfToken& tag : *renderTags) {
                    for (const char c : tag.GetString()) {
                        hash = (hash ^ static_cast<uint8_t>(c)) * 1099511628211ull;
                    }
                    hash = (hash ^ 0x2Cu) * 1099511628211ull;
                }
                // And which light groups are gathered: their planes accumulate
                // with the colour and stand only while the same ones do.
                for (const std::string& group : aovRequest.lightGroups) {
                    for (const char c : group) {
                        hash = (hash ^ static_cast<uint8_t>(c)) * 1099511628211ull;
                    }
                    hash = (hash ^ 0x3Bu) * 1099511628211ull;
                }
                now.tags = hash;
            }
            now.traced = true;
            // The same frame continued, or a new one: a camera that moved, a
            // scene that changed or a setting that did all start the mean
            // again, and only an identical frame adds to it. This is what
            // makes a viewport converge while it is left alone.
            const bool same = now == pathState_;
            paths.accumulate = same;
            if (!same) {
                pathTracer_->restart();
                pathSeed_ = 0;
                pathState_ = now;
            }
            paths.seed = pathSeed_++ * 7919u;
            LRT_TRY(pathTracer_->trace(batch, visibility_, projection, frame, paths, meshLayer_, &pathAux_));
            pathAuxValid_ = true;
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
        // The adaptive gate's counters, after the pass: which covered pixels
        // have stopped. Its own dispatch and readback, so after the batch.
        if (pathTracing && pathState_.adaptive) {
            auto progress = pathTracer_->progress(visibility_);
            if (!progress) return std::move(progress).error();
            pathProgress_ = *progress;
        }
        // After the frame's batch, never inside it: the denoiser submits work
        // of its own and waits for OIDN. It runs on a path traced frame that
        // has gathered what it was asked for, in place over the mean -- the
        // input is copied to OIDN's staging before OIDN writes anything.
        if (pathTracing && denoise_.load() && !denoiserFailed_ && pathTracer_->accumulated() >= pathTotal_.load()) {
            if (!denoiser_.has_value()) {
                auto made = technique::Denoiser::create(*library_);
                if (!made) {
                    denoiserFailed_ = true;
                    log::warn("hdLrt: no denoiser: {}", made.error().toString());
                } else {
                    denoiser_.emplace(std::move(*made));
                }
            }
            if (denoiser_.has_value()) {
                if (auto ran = denoiser_->denoise(meshLayer_.colour, &pathAux_.planes, 0, pathAux_.normalOffsetBytes(),
                                                 meshLayer_.colour, settings.width, settings.height);
                    !ran) {
                    denoiserFailed_ = true;
                    log::warn("hdLrt: denoising stopped: {}", ran.error().toString());
                }
            }
        }
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
    }
    if (under != nullptr) {
        LRT_TRY(rasterizer_->render(projection, splats, settings, targets, {}, under, &splatLights));
    } else {
        LRT_TRY(rasterizer_->render(projection, splats, settings, targets, points, nullptr, &splatLights));
    }
    LRT_TRY(gatherLightGroups(pathTracing, settings.width, settings.height));
    LRT_TRY(paintDomes(projection, settings.width, settings.height, targets));
    LRT_TRY(applyExposure(projection.exposure, settings.width, settings.height, targets));
    return ok();
}

Result<void> Engine::gatherLightGroups(bool traced, uint32_t width, uint32_t height) {
    const uint64_t pixels = uint64_t{width} * height;
    if (!traced || lightGroupCount_ == 0 || !lightGroupColour_.valid() || lightGroupPixels_ != pixels ||
        !pathTracer_.has_value() || pathTracer_->lightGroups() < lightGroupCount_) {
        return ok();
    }
    if (!groupsScaled_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "lrt/technique/exposure", "copyScaled");
        if (!made) return std::move(made).error();
        groupsScaled_.emplace(std::move(*made));
    }
    gpu::CommandBatch batch(*device_);
    const uint32_t entries = static_cast<uint32_t>(pixels * lightGroupCount_);
    groupsScaled_->dispatch(batch, {entries, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["source"].setBinding(pathTracer_->sum().rhi());
        cursor["sourceBase"].setData(static_cast<uint32_t>(pathTracer_->lightGroupMeanOffset(0)));
        cursor["colour"].setBinding(lightGroupColour_.rhi());
        cursor["params"]["scale"].setData(1.0F);
        cursor["params"]["pixels"].setData(entries);
    });
    return batch.submit(true);
}

Result<void> Engine::applyExposure(double stops, uint32_t width, uint32_t height, render::RenderTargets& targets) {
    if (stops == 0.0 || !targets.colour.valid()) {
        return ok();
    }
    if (!exposure_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "lrt/technique/exposure", "applyExposure");
        if (!made) return std::move(made).error();
        exposure_.emplace(std::move(*made));
    }
    const uint32_t pixels = width * height;
    gpu::CommandBatch batch(*device_);
    exposure_->dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["params"]["scale"].setData(static_cast<float>(std::exp2(stops)));
        cursor["params"]["pixels"].setData(pixels);
    });
    // The light groups are what the colour is made of, and take its exposure.
    if (lightGroupCount_ > 0 && lightGroupColour_.valid() && lightGroupPixels_ == pixels) {
        const uint32_t entries = pixels * lightGroupCount_;
        exposure_->dispatch(batch, {entries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(lightGroupColour_.rhi());
            cursor["params"]["scale"].setData(static_cast<float>(std::exp2(stops)));
            cursor["params"]["pixels"].setData(entries);
        });
    }
    return batch.submit(true);
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
    // And in the light groups' planes: a dome seen by the camera is its group's.
    if (lightGroupCount_ > 0 && lightGroupColour_.valid() && lightGroupPixels_ == uint64_t{width} * height) {
        if (!domeGroups_.has_value()) {
            auto made = gpu::ComputeKernel::create(*library_, "lrt/technique/dome_background", "domeBackgroundGroups");
            if (!made) return std::move(made).error();
            domeGroups_.emplace(std::move(*made));
        }
        domeGroups_->dispatch(batch, {width, height, 1}, [&](rhi::ShaderCursor cursor) {
            lightTable_->bind(cursor);
            if (textures_) {
                textures_->bind(cursor["gTextures"]);
            }
            cursor["depth"].setBinding(targets.depth.rhi());
            cursor["groupPlanes"].setBinding(lightGroupColour_.rhi());
            cursor["groupBase"].setData(uint32_t{0});
            cursor["groupCount"].setData(lightGroupCount_);
            technique::setCamera(cursor["camera"], projection, width, height);
            static constexpr const char* kNames[12] = {"v00", "v01", "v02", "v03", "v10", "v11",
                                                       "v12", "v13", "v20", "v21", "v22", "v23"};
            for (size_t k = 0; k < 12; ++k) {
                cursor["background"][kNames[k]].setData(toWorld[k]);
            }
        });
    }
    LRT_TRY(batch.submit(true));
    return ok();
}

}   // namespace lrt::usd
