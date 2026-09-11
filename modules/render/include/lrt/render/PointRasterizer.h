// Copyright (c) 2026 lucabRTrender contributors.
//
// The raster route for point clouds: a z-buffered pipeline, no sort, then
// optional surface splatting and eye-dome lighting. See Points.h for when to
// use it and when to use discs in the splat pipeline instead.
#pragma once

#include <span>

#include <slang-rhi.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/render/Points.h"
#include "lrt/render/TileRasterizer.h"

namespace lrt::render {

class PointRasterizer {
public:
    /// Unsupported on a device that cannot rasterise (CUDA): use discs.
    [[nodiscard]] static Result<PointRasterizer> create(gpu::ShaderLibrary& library);

    /// Draws `instances` into `targets` (colour linear premultiplied, depth in
    /// view z, zero where nothing was drawn; bottom row first).
    [[nodiscard]] Result<void> render(const Camera& camera,
                                      std::span<const PointInstance> instances,
                                      const RenderSettings& settings, RenderTargets& targets);
    [[nodiscard]] Result<void> render(const Projection& projection,
                                      std::span<const PointInstance> instances,
                                      const RenderSettings& settings, RenderTargets& targets);

    /// The colour render target of the last frame, top row first. For tests.
    [[nodiscard]] rhi::ITexture* colourTarget() const noexcept { return colourTarget_.get(); }

private:
    [[nodiscard]] Result<void> reserve(uint32_t width, uint32_t height);

    gpu::Device*                       device_ = nullptr;
    rhi::ComPtr<rhi::IRenderPipeline>  opaque_;
    rhi::ComPtr<rhi::IRenderPipeline>  surfaceDepth_;
    rhi::ComPtr<rhi::IRenderPipeline>  surfaceAccumulate_;
    gpu::ComputeKernel                 resolve_;
    gpu::ComputeKernel                 edl_;
    uint32_t                           width_ = 0;
    uint32_t                           height_ = 0;
    rhi::ComPtr<rhi::ITexture>         colourTarget_;
    rhi::ComPtr<rhi::ITexture>         depthTarget_;
    rhi::ComPtr<rhi::ITexture>         zBuffer_;
    rhi::ComPtr<rhi::ITextureView>     colourView_;
    rhi::ComPtr<rhi::ITextureView>     depthView_;
    rhi::ComPtr<rhi::ITextureView>     zView_;
    gpu::Buffer                        shaded_;
};

}   // namespace lrt::render
