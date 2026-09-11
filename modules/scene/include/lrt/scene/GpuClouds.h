// Copyright (c) 2026 lucabRTrender contributors.
//
// Splat and point clouds as they live on the device, and how they get there.
//
// The layout is in shaders/lrt/common/packing.slang. Getting there is:
//
//   CPU  read the file, arrange float records (io::RawSplats / RawPoints)
//   GPU  validate -> prefix sum -> decode + compact -> bounds
//
// in slices of at most kSliceBytes of raw records, so a cloud larger than a
// single device buffer may be still loads. The only numbers read back are one
// count per slice and the six numbers of the bounding box.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/PrefixSum.h"
#include "lrt/io/RawSplats.h"

namespace lrt::gpu {
class ShaderLibrary;
}

namespace lrt::scene {

struct Bounds {
    std::array<float, 3> min{0, 0, 0};
    std::array<float, 3> max{0, 0, 0};
};

struct GpuSplats {
    std::string source;
    uint32_t    count = 0;          ///< splats kept
    uint32_t    declared = 0;       ///< records in the file
    uint32_t    restPerColour = 0;  ///< 0, 3, 8, 15 -> degree 0..3
    uint32_t    shWords = 0;
    gpu::Buffer positions;          ///< float4
    gpu::Buffer shape;              ///< uint * 4
    gpu::Buffer sh;                 ///< uint * shWords (one dummy word at degree 0)
    Bounds      bounds;

    [[nodiscard]] uint32_t degree() const noexcept {
        return restPerColour == 15 ? 3 : restPerColour == 8 ? 2 : restPerColour == 3 ? 1 : 0;
    }
};

struct GpuPoints {
    std::string source;
    uint32_t    count = 0;
    uint32_t    declared = 0;
    gpu::Buffer positions;   ///< float4 xyz, 1
    gpu::Buffer colours;     ///< uint * 2: f16 r|g, b|a (linear)
    Bounds      bounds;
};

class CloudLoader {
public:
    static constexpr uint64_t kSliceBytes = uint64_t{256} << 20;

    [[nodiscard]] static Result<CloudLoader> create(gpu::ShaderLibrary& library);

    /// `maxDegree` caps the harmonics kept (0..3).
    [[nodiscard]] Result<GpuSplats> upload(const io::RawSplats& raw, uint32_t maxDegree = 3);
    /// `detail` keeps that fraction of the points, the same ones every time.
    [[nodiscard]] Result<GpuPoints> upload(const io::RawPoints& raw, float detail = 1.0F);

    /// The extent of `count` float4 positions, computed on the device.
    [[nodiscard]] Result<Bounds> boundsOf(const gpu::Buffer& positions, uint32_t count);

private:
    gpu::Device*       device_ = nullptr;
    gpu::PrefixSum     prefix_;
    gpu::ComputeKernel splatValidate_;
    gpu::ComputeKernel splatDecode_;
    gpu::ComputeKernel pointsValidate_;
    gpu::ComputeKernel pointsDecode_;
    gpu::ComputeKernel boundsChunks_;
    gpu::ComputeKernel boundsReduce_;
};

}   // namespace lrt::scene
