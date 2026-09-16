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
#include <set>
#include <memory>
#include <mutex>
#include <atomic>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/path.h>

#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/usd/PrimData.h"
#include "lrt/geom/Curves.h"
#include "lrt/geom/Mesh.h"
#include "lrt/geom/Skinner.h"
#include "lrt/geom/Subdivision.h"
#include "lrt/lod/Lod.h"
#include "lrt/technique/Visibility.h"
#include "lrt/io/Vdb.h"
#include "lrt/world/VolumeSet.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/Instancing.h"
#include "lrt/lod/Lrtc.h"
#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"
#include "lrt/io/Ies.h"
#include "lrt/light/LightTable.h"
#include "lrt/technique/MaterialShading.h"
#include "lrt/technique/Denoiser.h"
#include "lrt/technique/PathTracer.h"
#include "lrt/technique/EmissiveTable.h"
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
    /// LrtSplatLightingAPI: relit by the scene's lights rather than shown as
    /// it was baked.
    bool                                relight = false;
    std::vector<pxr::TfToken>           categories;   ///< what a light's link is tested against
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
    std::optional<CurveArrays>             pendingCurves;   ///< a BasisCurves prim: built as a tube mesh
    std::vector<InstancerLink>             instancing;      ///< innermost first; empty: not instanced
    world::InstanceChain                   chain;           ///< composed from `instancing`
    /// Under a shutter, when an instancer in the chain moves: the chain at the
    /// samples bracketing it, and when those are (the first moving level's).
    world::InstanceChain                   chainStart;
    world::InstanceChain                   chainEnd;
    double                                 chainTimeStart = 0.0;
    double                                 chainTimeEnd = 0.0;
    std::vector<uint64_t>                  chainVersions;   ///< the instancer versions `chain` was made from
    bool                                   chainDirty = false;
    std::shared_ptr<const geom::GpuMesh>   gpu;
    uint64_t                               topologyKey = 0;   ///< the key its GpuMesh was built with
    render::Mat4                           objectToWorld = render::Mat4::identity();
    /// Motion blur: the prim at the shutter's open and close, where they
    /// differ from the frame -- transforms, and meshes built from the points
    /// there under the same topology key.
    MeshTransforms                         shutter;
    pxr::VtValue                           pendingStart, pendingEnd;   ///< points to build gpuStart/gpuEnd from
    double                                 pointsTimeStart = 0.0, pointsTimeEnd = 0.0;
    std::shared_ptr<const geom::GpuMesh>   gpuStart, gpuEnd;
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
enum class AovKind {
    Colour, Depth, PrimId, InstanceId, ElementId, EyeNormal, WorldNormal, Primvar, Albedo, ShadingNormal, LightGroup
};

struct AovSource {
    AovKind  kind = AovKind::Colour;
    uint32_t primvar = 0;   ///< AovKind::Primvar: its index in AovRequest::primvars; LightGroup: in lightGroups
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
    /// "lightGroup:NAME" outputs, by NAME: each light's direct contribution
    /// under its group (`lrt:lightGroup`). At most technique::kMaxLightGroups.
    std::vector<std::string> lightGroups;
};

/// How the engine draws a frame.
enum class Technique {
    Raster,     ///< tile rasteriser, points composited
    /// Rays. A frame of nothing but splats is GaussianRayTracer's, whole.
    /// With meshes in it the surfaces are path traced and the splats
    /// composed over them by the rasteriser, since the tracer takes no
    /// `under` layer: splats inside the rays is still to be written.
    RayTraced,
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
                   std::optional<StreamedAsset> asset = std::nullopt,
                   std::optional<bool> relight = std::nullopt,
                   std::optional<std::vector<pxr::TfToken>> categories = std::nullopt);
    void setPoints(const pxr::SdfPath& id, std::optional<PointsArrays> raw,
                   const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<render::PointStyle> style);
    /// A BasisCurves prim: a mesh entry whose GpuMesh is a tube over its spans.
    void setCurves(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                   std::optional<CurveArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<MeshLook> look);
    void setMesh(const pxr::SdfPath& id, int32_t primId, const pxr::TfToken& renderTag,
                 std::optional<MeshArrays> arrays, const render::Mat4* transform, std::optional<bool> visible,
                 std::optional<MeshLook> look,
                 std::optional<std::vector<InstancerLink>> instancing = std::nullopt,
                 std::optional<MeshTransforms> shutter = std::nullopt);
    /// A material's network as a MaterialX document (null where hdMtlx could not
    /// read it: its meshes show displayColor). Compiled at the next commit.
    void setMaterial(const pxr::SdfPath& id, std::shared_ptr<void> mtlxDocument);
    void removeMaterial(const pxr::SdfPath& id);
    /// A UsdLux light, as the delegate read it. Lights light the meshes; the
    /// splats carry their own radiance until LrtSplatLightingAPI (M5).
    void setLight(const pxr::SdfPath& id, const light::Light& lamp,
                  std::vector<InstancerLink> instancing = {});
    void removeLight(const pxr::SdfPath& id);
    /// Samples per light per pixel. One is what an interactive frame takes;
    /// a render that wants an area light without noise asks for more.
    void setLightSamples(uint32_t samples);
    /// One light per sample, chosen by power, instead of every light at every
    /// pixel: exact either way, and which is cheaper is a measurement.
    void setChooseLights(bool choose);
    /// Whether a relit splat casts a shadow ray against the cloud's own
    /// proxies (`lrt:splatShadows`). Off: it takes each light whole, which
    /// is what relighting did before there was a ray to ask with.
    void setSplatShadows(bool shadows);
    /// Paths a pixel a path traced frame gathers, and how many bounces each
    /// one takes after its first hit. One of each is what an interactive
    /// frame affords.
    void setPathSamples(uint32_t samples);
    void setPathBounces(uint32_t bounces);
    /// Motion blur's shutter slices (1 to 8) for a path traced frame whose
    /// prims move over the camera's shutter; one is no blur.
    void setMotionBuckets(uint32_t buckets);
    /// The camera's shutter, in frames about the frame: what the buckets
    /// span, and what the prims' samples are placed against.
    void setShutter(double open, double close);
    /// Paths a pixel at which a path traced frame is finished. One -- the
    /// default -- is a frame that never accumulates, which is what a viewport
    /// showing a moving camera wants.
    void setPathTotal(uint32_t total);
    /// Denoise a path traced frame once it has gathered `pathTotal` paths
    /// (every frame, when the total is one). Off by default.
    void setDenoise(bool denoise);
    /// Adaptive: a pixel stops taking paths once its relative standard error
    /// falls below `error`; the frame is gathered when every covered pixel
    /// has stopped or `pathTotal` is reached, whichever first.
    void setPathAdaptive(bool adaptive);
    void setPathMis(bool mis);
    void setPathError(float error);

    /// How many paths a pixel the path traced frame on screen has gathered,
    /// and whether that is all it is going to gather. A frame that is not a
    /// path traced one has nothing to gather and is always finished.
    [[nodiscard]] uint32_t pathAccumulated() const noexcept;
    [[nodiscard]] bool pathConverged() const noexcept;
    /// The mesh pools' generation (a repack each) and positions revision (a
    /// deformation in place each), so a host can tell which one a change was.
    [[nodiscard]] uint64_t meshGeneration() const noexcept;
    /// The coordinate systems bound to a mesh prim, as its last Sync read them.
    [[nodiscard]] std::vector<CoordSysBinding> coordSysOf(const pxr::SdfPath& id) const;
    [[nodiscard]] uint64_t meshPositionsRevision() const noexcept;
    /// Volumes and the field assets they read. The medium is path traced;
    /// the raster technique draws no volume, and says so once.
    void setVolume(const pxr::SdfPath& id, VolumeArrays arrays);
    void removeVolume(const pxr::SdfPath& id);
    void setVolumeField(const pxr::SdfPath& id, VolumeFieldAsset asset);
    void removeVolumeField(const pxr::SdfPath& id);
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

    mutable std::mutex                        guard_;
    std::map<pxr::SdfPath, SplatEntry>        splats_;
    std::map<pxr::SdfPath, PointsEntry>       points_;
    std::map<pxr::SdfPath, MeshEntry>         meshes_;
    std::map<pxr::SdfPath, InstancerEntry>    instancers_;
    uint64_t                                  instancerVersion_ = 0;
    uint64_t nextTopologyKey_ = 0;   ///< one per topology a mesh was given
    std::optional<world::Instancing>          instancing_;
    std::optional<geom::MeshBuilder>          meshBuilder_;
    std::optional<geom::Skinner>              skinner_;
    std::optional<geom::CurveBuilder>         curveBuilder_;
    std::optional<geom::Subdivider>           subdivider_;   ///< made on first use
    std::optional<world::GpuScene>            scene_;
    std::optional<technique::VisibilityRaster> visibilityRaster_;   ///< each made on first use
    std::optional<world::RayTracingScene>      rayTracingScene_;
    std::optional<technique::VisibilityTrace>  visibilityTrace_;
    std::optional<world::BvhScene>             bvhScene_;
    std::optional<technique::VisibilityBvh>    visibilityBvh_;
    std::optional<technique::MaterialPrograms> materialPrograms_;
    std::optional<technique::MaterialShading> materialShading_;
    std::optional<technique::PathTracer>      pathTracer_;   ///< made on first use
    technique::PathAux                        pathAux_;      ///< the last path traced frame's albedo and normal
    bool                                      pathAuxValid_ = false;
    /// The last frame's light groups: `lightGroupCount_` planes of float4,
    /// a pixel each, the means in `colour` and the path tracer's sums in
    /// `sum`; sized for `lightGroupPixels_`.
    gpu::Buffer                               lightGroupColour_;
    uint32_t                                  lightGroupCount_ = 0;
    uint64_t                                  lightGroupPixels_ = 0;
    std::optional<technique::Denoiser>        denoiser_;     ///< made on first use
    std::atomic<bool>                         denoise_{false};
    bool                                      denoiserFailed_ = false;   ///< said once
    std::map<pxr::SdfPath, MaterialEntry>     materials_;
    /// A light as the delegate read it, and the instancers above it, whose
    /// chain is composed on the device as a mesh's is.
    struct LightEntry {
        light::Light               lamp;
        std::vector<InstancerLink> instancing;
        world::InstanceChain       chain;
        world::InstanceChain       chainStart;       ///< as a mesh's: where an instancer in the chain moves
        world::InstanceChain       chainEnd;
        double                     chainTimeStart = 0.0;
        double                     chainTimeEnd = 0.0;
        std::vector<uint64_t>      chainVersions;
        bool                       chainDirty = false;
    };
    std::map<pxr::SdfPath, LightEntry>        lights_;
    float                                     lightSceneRadius_ = 1.0F;   ///< the scene's reach, for domes' and suns' power
    uint64_t                                  lightRadiusGeneration_ = ~uint64_t{0};
    uint64_t                                  lightRadiusRevision_ = ~uint64_t{0};
    std::map<pxr::SdfPath, VolumeArrays>      volumes_;
    std::map<pxr::SdfPath, VolumeFieldAsset>  volumeFields_;
    /// Grids read, by file and grid name; null for one that failed to read
    /// (reported once).
    std::map<std::pair<std::string, std::string>, std::shared_ptr<const io::NanoGrid>> nanoGrids_;
    std::optional<world::VolumeSet>           volumeSet_;
    uint64_t                                  volumesVersion_ = 1;   ///< raised by any volume or field change
    uint64_t                                  volumesBuilt_ = 0;     ///< the version volumeSet_ holds
    uint32_t                                  volumesDrawn_ = 0;     ///< how many volumes volumeSet_ holds
    bool                                      volumesUndrawnSaid_ = false;
    /// IES profiles by path, read once; a file that cannot be read is said
    /// once and the light goes unshaped.
    std::map<std::string, std::shared_ptr<const io::IesProfile>> iesProfiles_;
    std::set<std::string>                     iesFailed_;
    std::optional<light::LightTable>          lightTable_;
    std::atomic<uint32_t>                     lightSamples_{1};
    std::atomic<bool>                         chooseLights_{false};
    std::atomic<bool>                         splatShadows_{false};
    /// Built only for relit splats to shadow against: the Hardware route, so
    /// there is a structure an inline ray can trace (the frame's own tracer
    /// may be on the compute route, which has none).
    std::optional<render::GaussianRayTracer>  shadowTracer_;
    std::atomic<uint32_t>                     pathSamples_{1};
    std::atomic<uint32_t>                     pathBounces_{1};
    std::atomic<uint32_t>                     motionBuckets_{4};
    std::atomic<double>                       shutterOpen_{0.0};
    std::atomic<double>                       shutterClose_{0.0};
    std::atomic<uint32_t>                     pathTotal_{1};
    std::atomic<bool>                         pathAdaptive_{false};
    std::atomic<bool>                         pathMis_{true};
    std::atomic<float>                        pathError_{0.02F};
    technique::PathProgress                   pathProgress_;   ///< after the last adaptive pass
    uint32_t                                  pathSeed_ = 0;   ///< which samples a path traced frame takes
    /// What the last path traced frame was of. A frame that matches it in
    /// every particular is the same frame continued, and its paths are added
    /// to the mean; anything else starts the mean again. The revision is what
    /// says the scene itself moved: `commit` raises it whenever it uploads,
    /// and so does every setting that changes what a path would find.
    struct PathState {
        render::Mat4 worldToView{};
        double       focalX = 0.0;
        double       focalY = 0.0;
        double       centreX = 0.0;
        double       centreY = 0.0;
        double       nearZ = 0.0;
        double       farZ = 0.0;
        bool         orthographic = false;
        double       lensRadius = 0.0;
        double       focusDistance = 0.0;
        double       distortionK1 = 0.0;
        double       distortionK2 = 0.0;
        uint32_t     width = 0;
        uint32_t     height = 0;
        uint32_t     samples = 0;
        uint32_t     bounces = 0;
        bool         adaptive = false;
        bool         mis = true;
        bool         cameraMoves = false;
        render::Mat4 cameraStart = render::Mat4::identity();
        render::Mat4 cameraEnd = render::Mat4::identity();
        float        error = 0.0F;
        uint64_t     revision = 0;
        uint64_t     tags = 0;         ///< a hash of the render tags drawn: purposes that change start the mean again
        bool         traced = false;   ///< the last frame was path traced at all
        /// Mat4 has no comparison of its own, so the camera is compared
        /// element by element: identical bits are what "has not moved" means.
        [[nodiscard]] bool operator==(const PathState& o) const {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    if (worldToView.at(r, c) != o.worldToView.at(r, c) || cameraStart.at(r, c) != o.cameraStart.at(r, c) ||
                        cameraEnd.at(r, c) != o.cameraEnd.at(r, c)) {
                        return false;
                    }
                }
            }
            if (cameraMoves != o.cameraMoves) {
                return false;
            }
            return focalX == o.focalX && focalY == o.focalY && centreX == o.centreX && centreY == o.centreY &&
                   nearZ == o.nearZ && farZ == o.farZ && orthographic == o.orthographic &&
                   lensRadius == o.lensRadius && focusDistance == o.focusDistance &&
                   distortionK1 == o.distortionK1 && distortionK2 == o.distortionK2 && width == o.width &&
                   height == o.height && samples == o.samples && bounces == o.bounces && adaptive == o.adaptive && mis == o.mis &&
                   error == o.error && revision == o.revision && tags == o.tags && traced == o.traced;
        }
    };
    PathState                                 pathState_;
    std::atomic<uint64_t>                     revision_{1};   ///< raised by anything a path would see
    /// Frames still to draw before what is gathered is the shutter's: a
    /// shutter changed after the prims synced is answered by dirtying them
    /// (RenderPass), and they carry their new samples only from the next
    /// Sync -- so this frame and the one that resamples are not converged,
    /// however many paths they hold.
    std::atomic<int>                          shutterSettle_{0};
    /// A bit per category name, as they are first seen: a prim's mask and a
    /// light's link have to agree on the numbering, and this is the only
    /// place that sees both. Past 64 names a category cannot be represented
    /// and its link reaches nothing, which is said once.
    std::map<std::string, uint32_t>           categoryBits_;
    [[nodiscard]] uint32_t categoryBit(const std::string& name);
    [[nodiscard]] uint64_t categoryMask(const std::vector<pxr::TfToken>& names);
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
    std::optional<gpu::ComputeKernel>          domeBackground_;
    std::optional<gpu::ComputeKernel>          exposure_;   ///< made on first use
    /// The emitting triangles as a light, and what they were weighed from.
    struct EmissiveKey {
        uint64_t          generation = ~uint64_t{0};
        uint64_t          positions = 0;
        uint64_t          revision = 0;
        rhi::IBuffer*     records = nullptr;
        uint32_t          instances = 0;
        [[nodiscard]] bool operator==(const EmissiveKey& o) const {
            return generation == o.generation && positions == o.positions && revision == o.revision &&
                   records == o.records && instances == o.instances;
        }
    };
    std::optional<technique::EmissiveTable>    emissiveTable_;
    EmissiveKey                                emissiveKey_;
    std::optional<gpu::ComputeKernel>          domeGroups_;     ///< the domes' background in their groups' planes
    std::optional<gpu::ComputeKernel>          groupsScaled_;   ///< the path tracer's group means, copied out
    /// The camera's exposure over the composed frame, once, after everything.
    [[nodiscard]] Result<void> applyExposure(double stops, uint32_t width, uint32_t height,
                                             render::RenderTargets& targets);
    /// The frame's domes over what it drew nothing on, after everything else.
    /// The frame's light group planes into lightGroupColour_: the path
    /// tracer's means copied out of its accumulation (the raster writes
    /// there itself), so the domes and the exposure change a copy.
    [[nodiscard]] Result<void> gatherLightGroups(bool traced, uint32_t width, uint32_t height);
    [[nodiscard]] Result<void> paintDomes(const render::Projection& projection, uint32_t width, uint32_t height,
                                          render::RenderTargets& targets);
    technique::VisibilityTargets              visibility_;
    std::optional<technique::AovShading>      aovShading_;
    technique::AovBuffers                     aovs_;
    bool                                      aovsValid_ = false;
    render::RenderTargets                     meshLayer_;
    render::RenderTargets                     opaqueLayer_;
};

}   // namespace lrt::usd
