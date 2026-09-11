// Copyright (c) 2026 lucabRTrender contributors.
//
// What a Hydra render delegate holds: the device, the renderers, and the
// flat scene the prims sync into.
//
// THE SCENE, AND WHO MAY TOUCH IT
//
// Hydra syncs prims on worker threads. A prim's Sync only arranges its arrays
// into CPU records and hands them here under the lock; nothing touches the
// device. The render pass then, on the thread that executes it, uploads
// whatever changed (the GPU decode) and renders. So the device has one caller,
// which is the rule openFXplayer's gpu_host lives by, and the scene the render
// reads is a committed one.
//
// Entries are keyed by prim path and live in slots with generations, so a
// handle to a destroyed prim is a stale handle and never somebody else's cloud.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/path.h>

#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/usd/PrimData.h"
#include "lrt/geom/Mesh.h"
#include "lrt/lod/Lod.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/Instancing.h"
#include "lrt/lod/Lrtc.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"
#include "lrt/technique/MaterialShading.h"
#include "lrt/render/GaussianRayTracer.h"
#include "lrt/render/PointRasterizer.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::usd {

/// LrtStreamedAssetAPI: a .lrtc drawn with its levels of detail.
struct StreamedAsset {
    std::string path;              ///< resolved; empty when the prim has none
    float       threshold = 1.0F;  ///< px a merged cell may span
    uint64_t    budget = 0;        ///< splats on the device; 0 reads the file whole

    bool operator==(const StreamedAsset&) const = default;
};

struct SplatEntry {
    std::optional<ParticleFieldArrays>  pending;   ///< synced, not yet uploaded
    std::unique_ptr<scene::GpuSplats>   gpu;
    render::Mat4                        objectToWorld = render::Mat4::identity();
    bool                                visible = true;
    render::SplatEdit                   edit;
    std::optional<StreamedAsset>        assetPending;
    StreamedAsset                       asset;
    std::unique_ptr<lod::LodCloud>      lodCloud;   ///< the asset read whole
    std::unique_ptr<lod::StreamingPool> pool;       ///< or streamed
};

struct InstancerEntry {
    InstancerArrays  arrays;
    pxr::SdfPath     parent;
    uint64_t         version = 0;
};

struct MeshEntry {
    std::optional<MeshArrays>              pending;
    std::vector<InstancerLink>             instancing;      ///< innermost first; empty: not instanced
    world::InstanceChain                   chain;           ///< composed from `instancing`
    std::vector<uint64_t>                  chainVersions;   ///< the instancer versions `chain` was made from
    bool                                   chainDirty = false;
    std::shared_ptr<const geom::GpuMesh>   gpu;
    render::Mat4                           objectToWorld = render::Mat4::identity();
    MeshLook                               look;
    std::vector<pxr::SdfPath>              subsetMaterials;   ///< per GeomSubset the mesh was built with
    uint32_t                               primId = 0;
    pxr::TfToken                           renderTag;
    bool                                   visible = true;
};

struct MaterialEntry {
    std::shared_ptr<void>                   document;   ///< MaterialX::DocumentPtr; null: nothing MaterialX reads
    bool                                    pending = true;
    bool                                    cutout = false;   ///< its opacity cuts samples away: visibility evaluates it
    std::optional<material::CompiledMaterial> compiled;
};

struct PointsEntry {
    std::optional<PointsArrays>         pending;
    std::unique_ptr<scene::GpuPoints>   gpu;
    render::Mat4                        objectToWorld = render::Mat4::identity();
    render::PointStyle                  style;
    bool                                visible = true;
};

/// The layout of a Hydra render buffer's pixels, for `Engine::writeAov`.
struct AovLayout {
    uint32_t channels = 4;
    uint32_t componentBytes = 4;
    uint32_t componentKind = 2;   ///< 0 unorm8, 1 float16, 2 float32
};

/// Which image an AOV reads.
enum class AovKind { Colour, Depth, PrimId, InstanceId, ElementId, EyeNormal, WorldNormal, Primvar };

struct AovSource {
    AovKind  kind = AovKind::Colour;
    uint32_t primvar = 0;   ///< AovKind::Primvar: its index in AovRequest::primvars
};

/// Where an AOV lives on the device, as shaders/lrt/usd/aov_convert.slang
/// reads it.
struct AovView {
    const gpu::Buffer* buffer = nullptr;   ///< null: nothing drew it, so its clear value
    uint32_t           source = 0;         ///< 0 float4 (colour, normals, primvars), 1 view z, 2 uint ids
    uint32_t           stride = 1;         ///< entries per pixel
    uint32_t           offset = 0;         ///< the AOV's entry within them
    bool               ids = false;        ///< cleared to -1, not 0
};

/// What a frame should compute beyond colour and depth.
struct AovRequest {
    bool                     ids = false;       ///< primId, instanceId, elementId
    bool                     normals = false;   ///< Neye, normal
    std::vector<std::string> primvars;          ///< "primvars:NAME" outputs, by NAME
};

/// How the engine draws a frame.
enum class Technique {
    Raster,     ///< tile rasteriser, points composited
    RayTraced,  ///< GaussianRayTracer, the device's faster route; splats only
};

/// Which route finds what meshes a pixel sees. All three fill the same
/// visibility targets with the same ids (tests/technique/test_visibility.cpp).
enum class MeshVisibility {
    Automatic,  ///< rays where the device has ray queries, else raster, else the compute BVH
    Raster,     ///< VisibilityRaster
    Rays,       ///< VisibilityTrace: the device's acceleration structures
    Bvh,        ///< VisibilityBvh: compute BVHs, for devices with neither
};

class Engine {
public:
    /// Null with a reason when there is no device; Hydra then gets nothing drawn.
    static std::unique_ptr<Engine> create(std::string& why);

    // --- from Sync (any thread) ---
    void setSplats(const pxr::SdfPath& id, std::optional<ParticleFieldArrays> raw,
                   const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<render::SplatEdit> edit = std::nullopt,
                   std::optional<StreamedAsset> asset = std::nullopt);
    void setPoints(const pxr::SdfPath& id, std::optional<PointsArrays> raw,
                   const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<render::PointStyle> style);
    void setMesh(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                 std::optional<MeshArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                 std::optional<MeshLook> look,
                 std::optional<std::vector<InstancerLink>> instancing = std::nullopt);
    /// A material's network as a MaterialX document (null where hdMtlx could not
    /// read it: its meshes show displayColor). Compiled at the next commit.
    void setMaterial(const pxr::SdfPath& id, std::shared_ptr<void> mtlxDocument);
    void removeMaterial(const pxr::SdfPath& id);
    void setInstancer(const pxr::SdfPath& id, const pxr::SdfPath& parent, InstancerArrays arrays);
    void removeInstancer(const pxr::SdfPath& id);
    void remove(const pxr::SdfPath& id);

    // --- from the render pass (one thread) ---
    /// Uploads what changed. Returns how many entries were uploaded.
    Result<size_t> commit();
    /// `settleStreams`: before drawing, cut and load until the streamed
    /// assets hold what this view wants (as much as their budgets allow) --
    /// for an image that must be complete. Otherwise streams fill in over
    /// the frames that follow.
    Result<void> render(const render::Projection& projection, const render::RenderSettings& settings,
                        render::RenderTargets& targets, Technique technique = Technique::Raster,
                        bool settleStreams = false, const pxr::TfTokenVector* renderTags = nullptr,
                        const AovRequest& aovs = {}, MeshVisibility visibility = MeshVisibility::Automatic);

    [[nodiscard]] gpu::Device& device() noexcept { return *device_; }
    [[nodiscard]] gpu::ShaderLibrary& library() noexcept { return *library_; }

    /// The targets the last render drew into (owned by the render pass).
    [[nodiscard]] const render::RenderTargets* lastTargets() const noexcept { return lastTargets_; }

    /// Where everything drawn in the last frame is, in world space: mesh boxes
    /// folded on the device, cloud boxes through their prims' transforms.
    /// Nothing before a frame has drawn anything.
    [[nodiscard]] Result<std::optional<scene::Bounds>> bounds();

    /// The last frame's `aov` on the device, for a caller that shows it
    /// there (lrt view) rather than reading it back.
    [[nodiscard]] AovView aovView(const render::RenderTargets& targets, AovSource aov) const;

    /// A render target as a Hydra render buffer's bytes, converted on the
    /// device -- format, row order, and for depth the host projection's [0, 1]
    /// from view z (`projection` is the host's row-vector matrix, 16 values) --
    /// and read into `into`.
    [[nodiscard]] Result<void> writeAov(const render::RenderTargets& targets, AovSource source,
                                        const AovLayout& layout, const double* projection, std::span<uint8_t> into);

private:
    Engine() = default;

    std::shared_ptr<gpu::Device>              device_;
    std::unique_ptr<gpu::ShaderLibrary>       library_;
    std::optional<scene::CloudLoader>         loader_;
    std::optional<render::TileRasterizer>     rasterizer_;
    std::optional<render::PointRasterizer>    pointRasterizer_;
    std::optional<render::GaussianRayTracer>  rayTracer_;   ///< made on first use
    std::optional<lod::CutSelector>           cutter_;      ///< made on first use
    std::optional<gpu::ComputeKernel>         aovConvert_;  ///< made on first use
    const render::RenderTargets*              lastTargets_ = nullptr;
    render::RenderTargets                     pointLayer_;

    std::mutex                                guard_;
    std::map<pxr::SdfPath, SplatEntry>        splats_;
    std::map<pxr::SdfPath, PointsEntry>       points_;
    std::map<pxr::SdfPath, MeshEntry>         meshes_;
    std::map<pxr::SdfPath, InstancerEntry>    instancers_;
    uint64_t                                  instancerVersion_ = 0;
    std::optional<world::Instancing>          instancing_;
    std::optional<geom::MeshBuilder>          meshBuilder_;   ///< made on first use
    std::optional<world::GpuScene>            scene_;
    std::optional<technique::VisibilityRaster> visibilityRaster_;   ///< each made on first use
    std::optional<world::RayTracingScene>      rayTracingScene_;
    std::optional<technique::VisibilityTrace>  visibilityTrace_;
    std::optional<world::BvhScene>             bvhScene_;
    std::optional<technique::VisibilityBvh>    visibilityBvh_;
    std::optional<technique::MaterialPrograms> materialPrograms_;
    std::optional<technique::MaterialShading> materialShading_;
    std::map<pxr::SdfPath, MaterialEntry>     materials_;
    bool                                      materialsChanged_ = true;
    bool                                      materialCutouts_ = false;   ///< a material in the frame cuts samples away
    std::unique_ptr<material::MaterialCompiler> compiler_;
    bool                                      compilerFailed_ = false;
    std::unique_ptr<material::TextureStore>   textures_;
    std::map<pxr::SdfPath, uint32_t>          materialRows_;     ///< into materialRecords_; absent: row 0, the fallback
    std::vector<std::string>                  materialSlotNames_;   ///< the extra primvar slots the blob was written for
    gpu::Buffer                               materialRecords_;
    gpu::Buffer                               materialBlob_;
    /// Row and blob for this frame's materials, primvar slots set on the scene.
    [[nodiscard]] Result<void> prepareMaterials(const std::vector<std::string>& aovPrimvars);
    std::optional<gpu::ComputeKernel>          nearest_;
    technique::VisibilityTargets              visibility_;
    std::optional<technique::AovShading>      aovShading_;
    technique::AovBuffers                     aovs_;
    bool                                      aovsValid_ = false;
    render::RenderTargets                     meshLayer_;
    render::RenderTargets                     opaqueLayer_;
};

}   // namespace lrt::usd
