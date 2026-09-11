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
#include "lrt/render/TileRasterizer.h"
#include "lrt/scene/GpuClouds.h"
#include "lrt/technique/DisplayTransform.h"

namespace lrt::gpu {
class Device;
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

    /// The last frame's `aov` ("color", "depth", "primId", "instanceId",
    /// "elementId", "Neye", "normal") as DisplayTransform reads it. An AOV no
    /// mesh drew, or one not requested, has no buffer: it shows as the background.
    [[nodiscard]] Result<technique::DisplaySource> displaySource(const std::string& aov);

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

    /// The Hydra outputs renders produce, colour and depth always among them
    /// ("primId", "instanceId", "elementId", "Neye", "normal", "primvars:st"...).
    void requestOutputs(const std::vector<std::string>& aovs);

    /// The last render's Hydra render buffer for `aov` ("color", "depth"), as
    /// a host mapping it reads it: the buffer's own format, bottom row first
    /// (Hydra's layout, Storm's and hdEmbree's).
    [[nodiscard]] Result<std::vector<uint8_t>> mappedOutput(const std::string& aov);

    /// Every camera prim on the stage.
    [[nodiscard]] std::vector<std::string> cameras() const;

    /// The stage's timeCodesPerSecond and startTimeCode: how a frame on a
    /// clock maps to a USD time.
    [[nodiscard]] double timeCodesPerSecond() const;
    [[nodiscard]] double startTimeCode() const;

private:
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
