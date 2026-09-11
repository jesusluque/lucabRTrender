// Copyright (c) 2026 lucabRTrender contributors.
//
// Gaussians ray traced through a hardware BVH of per-particle proxies, after
// 3DGRT (Moenne-Loccoz et al. 2024). The kernel, what it costs and what it
// does differently from the rasteriser are in shaders/lrt/rt/rt_integrate.slang.
//
//   per cloud, when it changes    particle frames (GPU), and
//                                   Hardware: proxies (GPU) -> one BLAS per chunk
//                                   ComputeBvh: boxes, Morton sort, hierarchy, refit (GPU)
//   per frame                     colours per instance (GPU), and
//                                   Hardware: TLAS over instances x chunks
//                                   ComputeBvh: the instance list
//
// Two routes to the same integrator (shaders/lrt/rt/rt_integrate.slang):
//   Hardware     proxies in a device BVH, inline RayQuery: Metal, Vulkan.
//   ComputeBvh   an LBVH of particle boxes built and traversed in compute:
//                every device, CUDA included.
// The same image either way. On Metal the compute route is the faster one
// (see create()); Auto picks it there.
// There is no CPU route in this engine.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <slang-rhi.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/RadixSort.h"
#include "lrt/render/TileRasterizer.h"

namespace lrt::render {

enum class RayTracingRoute { Auto, Hardware, ComputeBvh };

struct RayTracerSettings {
    /// Auto: ComputeBvh on Metal and wherever there is no ray tracing,
    /// Hardware elsewhere.
    RayTracingRoute route = RayTracingRoute::Auto;
    /// Traversals a ray may make (each records up to 256 entries) before what
    /// it has found is all it draws.
    uint32_t maxSegments = 16;
    /// Splats per bottom-level structure. Bounds a single build's memory.
    uint32_t chunkSplats = uint32_t{1} << 20;
};

struct RayTracerStats {
    uint32_t splats = 0;
    uint32_t chunks = 0;
    uint32_t instances = 0;
    RayTracingRoute route = RayTracingRoute::Hardware;
    bool     rebuilt = false;   ///< per-cloud structures were built this frame
    double   buildMs = 0;       ///< structures (and BLAS when rebuilt) and colours
    double   renderMs = 0;      ///< the traced pass
    double   totalMs = 0;
};

class GaussianRayTracer {
public:
    /// Whether `device` can take the Hardware route. ComputeBvh runs anywhere.
    [[nodiscard]] static bool hardwareSupported(const gpu::Device& device) noexcept;

    [[nodiscard]] static Result<GaussianRayTracer> create(gpu::ShaderLibrary& library,
                                                          RayTracerSettings settings = {});

    [[nodiscard]] Result<RayTracerStats> render(const Camera& camera,
                                                std::span<const SplatInstance> instances,
                                                const RenderSettings& settings,
                                                RenderTargets& targets);
    [[nodiscard]] Result<RayTracerStats> render(const Projection& projection,
                                                std::span<const SplatInstance> instances,
                                                const RenderSettings& settings,
                                                RenderTargets& targets);

private:
    struct CloudKey {
        const scene::GpuSplats* cloud = nullptr;
        rhi::IBuffer*           positions = nullptr;
        uint32_t                count = 0;
        uint32_t                restPerColour = 0;
        bool operator==(const CloudKey&) const = default;
    };
    struct Cloud {
        CloudKey key;
        uint32_t base = 0;          ///< first particle in `frames_` (and leaf in `bvhLeaves_`)
        uint32_t firstChunk = 0;    ///< Hardware
        uint32_t chunks = 0;        ///< Hardware
        uint32_t nodeBase = 0;      ///< ComputeBvh: first internal node
    };

    [[nodiscard]] Result<void> rebuild(std::span<const SplatInstance> instances);
    [[nodiscard]] Result<void> buildHardware(const Cloud& cloud,
                                             std::vector<rhi::ComPtr<rhi::IAccelerationStructure>>& blas);
    [[nodiscard]] Result<void> buildBvh(const Cloud& cloud, gpu::Buffer& boxes, gpu::Buffer& children,
                                        gpu::Buffer& leaves);
    [[nodiscard]] Result<void> prepareFrame(std::span<const SplatInstance> instances,
                                            const Vec3& eyeWorld, uint32_t shLimit);
    [[nodiscard]] const Cloud* find(const scene::GpuSplats& splats) const;

    gpu::Device*       device_ = nullptr;
    RayTracerSettings  settings_;   ///< route resolved: never Auto
    gpu::ComputeKernel proxy_;
    gpu::ComputeKernel frames_kernel_;
    gpu::ComputeKernel shade_;
    gpu::ComputeKernel render_;
    // ComputeBvh
    gpu::RadixSort     sort_;
    gpu::ComputeKernel bvhLeaves_;
    gpu::ComputeKernel bvhHierarchy_;
    gpu::ComputeKernel bvhRefit_;
    gpu::ComputeKernel count_;
    gpu::ComputeKernel bvhRender_;

    std::vector<Cloud> clouds_;
    std::vector<rhi::ComPtr<rhi::IAccelerationStructure>> blas_;
    rhi::ComPtr<rhi::IAccelerationStructure> tlas_;
    uint32_t    splats_ = 0;
    gpu::Buffer frames_;            ///< float4 * 4 per splat
    gpu::Buffer colours_;           ///< float4 per instance particle, this frame
    uint64_t    colourCapacity_ = 0;
    gpu::Buffer instanceDescs_;
    gpu::Buffer instanceData_;      ///< per TLAS instance: world->cloud rows (float4 * 3)
    gpu::Buffer instanceIndices_;   ///< per instance: first particle (of the chunk), colour offset
    uint32_t    instanceCount_ = 0; ///< ComputeBvh: instances this frame
    gpu::Buffer bvhBoxes_;          ///< float4 * 2 per internal node, all clouds
    gpu::Buffer bvhChildren_;       ///< uint * 2 per internal node
    gpu::Buffer bvhLeaves_buffer_;  ///< uint per particle: sorted leaf -> particle in cloud
    gpu::Buffer bvhInstances_;      ///< per instance: node base, leaf base, particles
};

}   // namespace lrt::render
