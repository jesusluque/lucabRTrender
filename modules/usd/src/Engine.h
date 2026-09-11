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
#include <string>
#include <vector>

#include <pxr/usd/sdf/path.h>

#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/io/RawSplats.h"
#include "lrt/lod/Lod.h"
#include "lrt/lod/Lrtc.h"
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
    std::optional<io::RawSplats>        pending;   ///< synced, not yet uploaded
    std::unique_ptr<scene::GpuSplats>   gpu;
    render::Mat4                        objectToWorld = render::Mat4::identity();
    bool                                visible = true;
    render::SplatEdit                   edit;
    std::optional<StreamedAsset>        assetPending;
    StreamedAsset                       asset;
    std::unique_ptr<lod::LodCloud>      lodCloud;   ///< the asset read whole
    std::unique_ptr<lod::StreamingPool> pool;       ///< or streamed
};

struct PointsEntry {
    std::optional<io::RawPoints>        pending;
    std::unique_ptr<scene::GpuPoints>   gpu;
    render::Mat4                        objectToWorld = render::Mat4::identity();
    render::PointStyle                  style;
    bool                                visible = true;
};

/// How the engine draws a frame.
enum class Technique {
    Raster,     ///< tile rasteriser, points composited
    RayTraced,  ///< GaussianRayTracer, the device's faster route; splats only
};

class Engine {
public:
    /// Null with a reason when there is no device; Hydra then gets nothing drawn.
    static std::unique_ptr<Engine> create(std::string& why);

    // --- from Sync (any thread) ---
    void setSplats(const pxr::SdfPath& id, std::optional<io::RawSplats> raw,
                   const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<render::SplatEdit> edit = std::nullopt,
                   std::optional<StreamedAsset> asset = std::nullopt);
    void setPoints(const pxr::SdfPath& id, std::optional<io::RawPoints> raw,
                   const render::Mat4* transform, std::optional<bool> visible,
                   std::optional<render::PointStyle> style);
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
                        bool settleStreams = false);

    [[nodiscard]] gpu::Device& device() noexcept { return *device_; }

private:
    Engine() = default;

    std::shared_ptr<gpu::Device>              device_;
    std::unique_ptr<gpu::ShaderLibrary>       library_;
    std::optional<scene::CloudLoader>         loader_;
    std::optional<render::TileRasterizer>     rasterizer_;
    std::optional<render::PointRasterizer>    pointRasterizer_;
    std::optional<render::GaussianRayTracer>  rayTracer_;   ///< made on first use
    std::optional<lod::CutSelector>           cutter_;      ///< made on first use
    render::RenderTargets                     pointLayer_;

    std::mutex                                guard_;
    std::map<pxr::SdfPath, SplatEntry>        splats_;
    std::map<pxr::SdfPath, PointsEntry>       points_;
};

}   // namespace lrt::usd
