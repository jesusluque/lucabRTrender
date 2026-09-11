// Copyright (c) 2026 lucabRTrender contributors.
//
// Visibility: which triangle of which instance every pixel sees, as a texture
// of (instance + 1, triangle) -- 0 for nothing -- that shading, AOVs and
// picking read. Rasterised here; ray traced where the device has no raster.
// Whichever pass produced it, what a pixel shows is computed the same way
// from those two numbers.
#pragma once

#include <cstdint>
#include <span>

#include "lrt/core/Result.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/RasterKernel.h"
#include "lrt/gpu/Texture.h"
#include "lrt/render/Camera.h"
#include "lrt/render/TileRasterizer.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/BvhScene.h"
#include "lrt/world/RayTracingScene.h"

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::technique {

struct VisibilityTargets {
    uint32_t     width = 0;
    uint32_t     height = 0;
    gpu::Texture ids;     ///< RGBA32Uint: (instance + 1, triangle, 0, 0), row 0 on top
    gpu::Texture depth;   ///< D32Float, reversed: near / z
};

/// The camera's fields by name (shaders/lrt/technique/camera.slang).
void setCamera(rhi::ShaderCursor cursor, const render::Projection& projection, uint32_t width, uint32_t height);

class VisibilityRaster {
public:
    [[nodiscard]] static Result<VisibilityRaster> create(gpu::ShaderLibrary& library);

    /// Every instance of `scene` into `targets` (sized to width x height).
    [[nodiscard]] Result<void> render(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                      const render::Projection& projection, uint32_t width, uint32_t height,
                                      VisibilityTargets& targets);

private:
    gpu::Device*      device_ = nullptr;
    gpu::RasterKernel pass_;
};

/// Visibility by rays against a scene's hardware acceleration structures:
/// the same targets VisibilityRaster fills, for a device with no raster (and
/// as the rasteriser's check).
class VisibilityTrace {
public:
    [[nodiscard]] static Result<VisibilityTrace> create(gpu::ShaderLibrary& library);

    [[nodiscard]] Result<void> render(gpu::CommandBatch& batch, const world::RayTracingScene& scene,
                                      const render::Projection& projection, uint32_t width, uint32_t height,
                                      VisibilityTargets& targets);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel trace_;
};

/// Visibility by rays through a scene's compute BVHs: for a device with no ray
/// tracing hardware, and the second route that checks the first.
class VisibilityBvh {
public:
    [[nodiscard]] static Result<VisibilityBvh> create(gpu::ShaderLibrary& library);

    [[nodiscard]] Result<void> render(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                      const world::BvhScene& bvh, const render::Projection& projection,
                                      uint32_t width, uint32_t height, VisibilityTargets& targets);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel traverse_;
};

/// Lit from the eye: displayColor times the cosine to the eye, the smooth
/// normal where the mesh has one. The first shading, and the one tests use.
class HeadlightShading {
public:
    [[nodiscard]] static Result<HeadlightShading> create(gpu::ShaderLibrary& library);

    /// `targets` into an engine image (colour, view-z depth), bottom row first.
    [[nodiscard]] Result<void> shade(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                     const VisibilityTargets& targets, const render::Projection& projection,
                                     render::RenderTargets& out);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel shade_;
};

/// Arbitrary outputs from visibility, bottom row first: ids (prim, instance,
/// element; -1 for nothing), eye- and world-space normals, and primvars by
/// scene slot.
struct AovBuffers {
    uint32_t    width = 0;
    uint32_t    height = 0;
    uint32_t    primvarSlots = 0;
    gpu::Buffer ids;            ///< uint, 3 per pixel (int32 bits)
    gpu::Buffer eyeNormals;     ///< float4 per pixel (Hydra's eye space: -Z forward)
    gpu::Buffer worldNormals;   ///< float4 per pixel
    gpu::Buffer primvars;       ///< float4, primvarSlots per pixel
};

class AovShading {
public:
    [[nodiscard]] static Result<AovShading> create(gpu::ShaderLibrary& library);

    /// `slots`: the scene primvar slots to write, in output order.
    [[nodiscard]] Result<void> shade(gpu::CommandBatch& batch, const world::GpuScene& scene,
                                     const VisibilityTargets& targets, const render::Projection& projection,
                                     std::span<const uint32_t> slots, AovBuffers& out);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel aovs_;
};

}   // namespace lrt::technique
