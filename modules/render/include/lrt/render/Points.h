// Copyright (c) 2026 lucabRTrender contributors.
//
// Point clouds: how they are drawn and where they are.
//
// Two routes, chosen per frame:
//   - Discs: through the splat pipeline (TileRasterizer), sorted with the
//     splats, opaque hard discs. Every backend, CUDA included, and exactly
//     composited with splats.
//   - Raster: a rasterisation pipeline with a z-buffer (PointRasterizer) --
//     no sort at all, the fast route for large clouds -- on the backends that
//     rasterise (Metal, Vulkan, D3D12), with eye-dome lighting and surface
//     splatting. Splats then composite over it through the blend's
//     under-layer.
#pragma once

#include "lrt/render/Camera.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::render {

struct PointStyle {
    enum class Size { World, Pixels };
    Size   sizeMode = Size::World;
    float  size = 0.01F;              ///< diameter, world units or pixels
    bool   constantColour = false;
    float  colour[3] = {1.0F, 1.0F, 1.0F};
    /// Eye-dome lighting (raster route). Zero strength is off.
    float  edlStrength = 0.0F;
    float  edlRadius = 1.0F;
    /// Surface splatting (raster route): blend points within `depthOffset` of
    /// the nearest surface with Gaussian weights. Zero is plain opaque discs.
    float  surfaceDepthOffset = 0.0F;
};

struct PointInstance {
    const scene::GpuPoints* points = nullptr;
    Mat4                    objectToWorld = Mat4::identity();
    PointStyle              style;
};

}   // namespace lrt::render
