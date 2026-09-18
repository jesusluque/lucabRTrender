// Copyright (c) 2026 lucabRTrender contributors.
//
// A USD stage through the engine's Hydra delegate: UsdImaging's scene indices
// feed a render index whose delegate is this engine, and a task controller
// asks for colour, depth and whatever outputs are requested. Images come back
// to the host (render), or stay on the device for a viewport (draw).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lrt/core/Result.h"
#include "lrt/render/Camera.h"
#include "lrt/usd/PrimData.h"
#include "lrt/usd/RenderSettings.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"
#include "lrt/technique/DisplayTransform.h"

namespace lrt::gpu {
class Device;
class ShaderLibrary;
}

namespace lrt::usd {

struct StageImage {
    uint32_t           width = 0;
    uint32_t           height = 0;
    std::vector<float> rgba;    ///< bottom row first, linear, premultiplied
    std::vector<float> depth;   ///< view z (distance along the view axis; 0 where nothing was drawn), bottom row first
};

/// A prim of the stage, for a tree view.
struct StagePrim {
    std::string path;
    std::string name;
    std::string type;
    bool        hasChildren = false;
    bool        instance = false;   ///< a native instance: its children are its prototype's
};

/// What a pixel saw.
struct StagePick {
    std::string rprim;          ///< Hydra's path (a prototype's, under instancing)
    std::string prim;           ///< the USD prim it came from
    int32_t     instance = -1;  ///< which instance of it
};

class StageRenderer {
public:
    [[nodiscard]] static Result<std::unique_ptr<StageRenderer>> open(const std::filesystem::path& stage);
    ~StageRenderer();

    /// `camera` is a UsdGeomCamera prim path; empty takes the first camera.
    /// `technique` is the delegate's `lrt:technique` setting: "raster" or "rt".
    /// A path traced image is drawn until it holds `setPathTotal` paths.
    [[nodiscard]] Result<StageImage> render(const std::string& camera, double time,
                                            uint32_t width, uint32_t height,
                                            const std::string& technique = "raster");

    /// The same, from a camera that is not on the stage (the engine's
    /// convention: looking down its own -Z, as USD's cameras do).
    [[nodiscard]] Result<StageImage> render(const render::Camera& camera, double time, uint32_t width,
                                            uint32_t height, const std::string& technique = "raster");

    /// The same frames for a viewport: drawn and left on the device, nothing
    /// read back, streamed assets filling in over the frames that follow.
    [[nodiscard]] Result<void> draw(const std::string& camera, double time, uint32_t width, uint32_t height,
                                    const std::string& technique = "raster");
    [[nodiscard]] Result<void> draw(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                    const std::string& technique = "raster");

    /// The device the engine draws on: a window's surface is made on it.
    [[nodiscard]] gpu::Device& device();
    /// The engine's shader library on that device, for kernels drawn beside it.
    [[nodiscard]] gpu::ShaderLibrary& library();

    /// The last frame's `aov` ("color", "depth", "primId", "instanceId",
    /// "elementId", "Neye", "normal") as DisplayTransform reads it. An AOV no
    /// mesh drew, or one not requested, has no buffer: it shows as the background.
    [[nodiscard]] Result<technique::DisplaySource> displaySource(const std::string& aov);

    /// A camera of the engine's own framing what the stage draws, as lrt view
    /// opens on a stage: a small frame commits the scene (raster, whatever
    /// `technique` is), the bounds of what it drew place an orbit camera of
    /// `focal` mm. For a stage without cameras.
    [[nodiscard]] Result<render::Camera> framingCamera(double time, double focal = 35.0,
                                                       const std::string& technique = "raster");

    /// Where what the last frame drew is, in world space.
    [[nodiscard]] Result<std::optional<scene::Bounds>> bounds();

    /// The prim under pixel (x, y) of the last frame, from the top left; nothing
    /// where no mesh drew. "primId" must be among the requested outputs.
    [[nodiscard]] Result<std::optional<StagePick>> pick(uint32_t x, uint32_t y);

    /// The children of the prim at `path` ("/" for the stage's root).
    [[nodiscard]] std::vector<StagePrim> children(const std::string& path) const;

    /// 'Y' or 'Z'.
    [[nodiscard]] char upAxis() const;
    [[nodiscard]] double endTimeCode() const;

    /// How renders find what meshes a pixel sees: "automatic" (the default:
    /// rays where the device has ray queries, else raster, else compute BVHs),
    /// "raster", "rays" or "bvh". The delegate's `lrt:visibility` setting.
    [[nodiscard]] Result<void> setMeshVisibility(const std::string& route);

    /// Whether the stage authors a UsdLux light of its own.
    [[nodiscard]] bool hasLights() const;
    /// A sky dome and a sun in the stage's session layer, or not. For a stage
    /// that authors no lights: without them the path tracer lights the first
    /// hit from the eye, as the raster does, and has nothing to bounce. The
    /// file is not touched; the lights live at `/lrtDefaultLights` and reach
    /// the renderer through Hydra as authored ones do.
    [[nodiscard]] Result<void> setDefaultLights(bool on);

    /// The light this stage's meshes carry, at points somebody names.
    ///
    /// `rays` holds two `float4` a point -- where its ray starts and how near
    /// it may hit, then which way it goes -- and what comes back, one `float4`
    /// a point, is the radiance leaving the surface that ray finds, path
    /// traced with this stage's own lights, shadows and bounces. The stage is
    /// synced first, by drawing one pixel of it, because the meshes and their
    /// materials have to be on the device before anything can be asked of
    /// them.
    ///
    /// What it is for: `lrt mesh2splat`, which turns a mesh into gaussians
    /// that carry the light the mesh had.
    [[nodiscard]] Result<std::vector<float>> bakePoints(const std::vector<float>& rays, uint32_t count,
                                                        double time, uint32_t samples = 64,
                                                        uint32_t bounces = 3);

    /// Samples per light per pixel: one for an interactive frame, more where
    /// an area light's noise would be read as error.
    void setLightSamples(uint32_t samples);

    /// One light per sample, chosen by power, rather than every light at every
    /// pixel.
    void setChooseLights(bool choose);
    /// `lrt:splatShadows`: a relit cloud shadows itself, one ray a splat.
    void setSplatShadows(bool shadows);

    /// The path traced technique ("rt" over meshes): paths a pixel each pass
    /// gathers, bounces after the first hit, and the paths a pixel at which
    /// the frame is finished. A total of one -- the default -- never
    /// accumulates, which is what a moving camera wants.
    void setPathSamples(uint32_t samples);
    void setPathBounces(uint32_t bounces);
    /// Motion blur's shutter slices for `rt`, 1 to 8; the shutter itself is
    /// the camera's.
    void setMotionBuckets(uint32_t buckets);
    /// Subdivision surfaces refined this many levels (0: the control mesh),
    /// as usdview's complexity sets it.
    void setRefineLevel(uint32_t level);
    void setPathTotal(uint32_t total);
    /// Denoise a path traced frame once it has gathered its total.
    void setDenoise(bool denoise);
    /// Adaptive sampling, and the relative error a pixel stops at.
    void setPathAdaptive(bool adaptive);
    /// Weigh light sampling and material sampling by the power heuristic
    /// (the default), or light a surface by light sampling alone.
    void setPathMis(bool mis);
    void setPathError(float error);

    /// How many paths a pixel the frame on the device holds, and whether it
    /// holds all it is going to. A frame that is not path traced has nothing
    /// to gather and reads as finished.
    [[nodiscard]] uint32_t pathAccumulated() const;
    [[nodiscard]] bool pathConverged() const;
    /// The mesh pools' generation and positions revision (Engine's): a
    /// deformation raises the second and not the first.
    [[nodiscard]] uint64_t meshGeneration() const;
    /// The coordinate systems bound to a mesh prim (UsdShadeCoordSysAPI):
    /// names and transforms, as the delegate resolved them.
    [[nodiscard]] std::vector<CoordSysBinding> coordSysBindings(const std::string& prim) const;
    [[nodiscard]] uint64_t meshPositionsRevision() const;

    /// The Hydra outputs renders produce, colour and depth always among them
    /// ("primId", "instanceId", "elementId", "Neye", "normal", "primvars:st"...).
    void requestOutputs(const std::vector<std::string>& aovs);

    /// The last render's Hydra render buffer for `aov` ("color", "depth"), as
    /// a host mapping it reads it: the buffer's own format, bottom row first
    /// (Hydra's layout, Storm's and hdEmbree's).
    [[nodiscard]] Result<std::vector<uint8_t>> mappedOutput(const std::string& aov);

    /// Every camera prim on the stage.
    [[nodiscard]] std::vector<std::string> cameras() const;

    /// A UsdRender settings prim, as Hydra's renderSettings bprim holds it
    /// once it is made the scene's active one and synced: its products and
    /// vars, purposes, colour space and `lrt:` settings.
    [[nodiscard]] Result<RenderSettingsInfo> renderSettings(const std::string& path);
    /// Renders every product of that settings prim -- each at its own
    /// resolution from its own camera, its vars as the layers of one OpenEXR
    /// written where `productName` says (relative to `directory`), 32-bit
    /// floats unless `lrt:exrHalf` is set -- with `includedPurposes` as the
    /// render tags and its `lrt:` settings applied. Returns the files written.
    [[nodiscard]] Result<std::vector<std::filesystem::path>> renderProducts(const std::string& path, double time,
                                                                            const std::filesystem::path& directory = {});
    /// The purposes the next renders draw ("default", "render", "proxy",
    /// "guide"): Hydra's render tags. Empty: default and render.
    void setIncludedPurposes(const std::vector<std::string>& purposes);
    /// The material binding purposes the next renders resolve, in order, as
    /// a settings prim's `materialBindingPurposes` lists them ("full",
    /// "preview", "" for the all-purpose binding). Empty: "full", then "".
    /// The first named purpose is the one looked for, the all-purpose
    /// binding the fallback: a second named purpose is not consulted.
    void setMaterialBindingPurposes(const std::vector<std::string>& purposes);

    /// The stage's timeCodesPerSecond and startTimeCode: how a frame on a
    /// clock maps to a USD time.
    [[nodiscard]] double timeCodesPerSecond() const;
    [[nodiscard]] double startTimeCode() const;

private:
    [[nodiscard]] Result<void> executeUntilGathered(uint32_t width, uint32_t height);
    StageRenderer();
    [[nodiscard]] Result<void> aim(const std::string& camera, double time, const std::string& technique);
    [[nodiscard]] Result<void> aim(const render::Camera& camera, double time, uint32_t width, uint32_t height,
                                   const std::string& technique);
    [[nodiscard]] Result<void> execute(uint32_t width, uint32_t height);
    [[nodiscard]] Result<StageImage> readImage(uint32_t width, uint32_t height);
    [[nodiscard]] const render::RenderTargets* lastTargets() const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::usd
