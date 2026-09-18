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
#include <cstddef>
#include <span>
#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/algo/PrefixSum.h"
#include "lrt/io/RawSplats.h"
#include "lrt/io/Sog.h"

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
    /// What each splat reflects with, where the cloud carries it: one uint a
    /// splat, metallic in the low byte and roughness in the next. Empty for a
    /// capture, which has neither -- `hasPbr` is how a kernel asks.
    gpu::Buffer pbr;
    Bounds      bounds;

    [[nodiscard]] bool hasPbr() const noexcept { return pbr.valid(); }

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

/// A float array as it sits in memory -- float32 values, or float16 when
/// `half` -- uploaded as bytes and read on the device.
struct FloatStream {
    std::span<const std::byte> bytes;
    bool                       half = false;
    bool                       isDouble = false;   ///< float64 (matrix4d, point3d); `half` then false

    [[nodiscard]] bool   empty() const noexcept { return bytes.empty(); }
    [[nodiscard]] size_t values() const noexcept { return bytes.size() / (isDouble ? 8 : half ? 2 : 4); }
    /// The kind kernels read: 0 none, 1 float, 2 half, 3 double.
    [[nodiscard]] uint32_t kind() const noexcept { return empty() ? 0 : isDouble ? 3 : half ? 2 : 1; }
};

/// A splat cloud as separate arrays, the way USD's ParticleField stores one.
/// Empty streams take defaults: identity rotation, unit scale, opacity 1, DC 0.
struct SplatStreams {
    std::string source;
    uint32_t    count = 0;
    FloatStream positions;      ///< xyz
    FloatStream rotations;      ///< xyzw (GfQuat's layout)
    FloatStream scales;         ///< xyz, linear
    FloatStream opacities;      ///< linear
    uint32_t    coefficients = 0;   ///< SH coefficients per splat, DC first: (degree + 1)^2
    FloatStream sh;             ///< rgb per coefficient
    /// What the gaussian reflects with, one per splat, where the stage says
    /// so (`primvars:lrt:splat:metallic` and `:roughness`). Empty otherwise.
    FloatStream metallic;
    FloatStream roughness;
    FloatStream transmission;
};

/// A point cloud as separate arrays, the way UsdGeomPoints stores one.
struct PointStreams {
    std::string source;
    uint32_t    count = 0;
    FloatStream positions;   ///< xyz
    FloatStream colours;     ///< rgb, linear: one for every point, one per point, or empty (white)
};

class CloudLoader {
public:
    static constexpr uint64_t kSliceBytes = uint64_t{256} << 20;

    [[nodiscard]] static Result<CloudLoader> create(gpu::ShaderLibrary& library);

    /// `maxDegree` caps the harmonics kept (0..3).
    [[nodiscard]] Result<GpuSplats> upload(const io::RawSplats& raw, uint32_t maxDegree = 3);
    /// A SOG's images, decoded on the device into records that then take the
    /// same validate and decode as any other format -- nothing crosses back.
    [[nodiscard]] Result<GpuSplats> upload(const io::RawSog& sog, uint32_t maxDegree = 3);
    /// The same decode, read back as float records, for what consumes records
    /// on the host side (the USD export).
    [[nodiscard]] Result<io::RawSplats> records(const io::RawSog& sog, uint32_t maxDegree = 3);
    /// `detail` keeps that fraction of the points, the same ones every time.
    [[nodiscard]] Result<GpuPoints> upload(const io::RawPoints& raw, float detail = 1.0F);
    /// Arrays uploaded as they are and interleaved into records on the device.
    [[nodiscard]] Result<GpuSplats> upload(const SplatStreams& streams, uint32_t maxDegree = 3);
    [[nodiscard]] Result<GpuPoints> upload(const PointStreams& streams, float detail = 1.0F);

    /// The extent of `count` float4 positions, computed on the device.
    [[nodiscard]] Result<Bounds> boundsOf(const gpu::Buffer& positions, uint32_t count);

private:
    struct SogOnDevice;
    [[nodiscard]] Result<SogOnDevice> sogOnDevice(const io::RawSog& sog, uint32_t maxDegree);
    [[nodiscard]] Result<void> sogSlice(const SogOnDevice& on, uint32_t first, uint32_t n, const gpu::Buffer& into);
    [[nodiscard]] Result<GpuSplats> startSplats(const std::string& source, uint32_t declared, uint32_t keep,
                                                bool withPbr = false);
    /// Validates and decodes `n` records in `raw` into `splats` after `written`.
    [[nodiscard]] Result<uint32_t> decodeSlice(const gpu::Buffer& raw, const io::SplatEncoding& e, uint32_t n,
                                               uint32_t written, uint32_t keep, GpuSplats& splats);
    [[nodiscard]] Result<void> finishSplats(GpuSplats& splats, uint32_t written);
    [[nodiscard]] Result<uint32_t> decodePoints(const gpu::Buffer& raw, uint32_t n, uint32_t first,
                                                uint32_t colourKind, float detail, uint32_t written,
                                                GpuPoints& points);
    [[nodiscard]] Result<gpu::Buffer> streamBuffer(const FloatStream& stream, const char* label);

    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel sogDecode_;
    gpu::PrefixSum     prefix_;
    gpu::ComputeKernel splatValidate_;
    gpu::ComputeKernel splatDecode_;
    gpu::ComputeKernel pointsValidate_;
    gpu::ComputeKernel pointsDecode_;
    gpu::ComputeKernel splatStreams_;
    gpu::ComputeKernel pointStreams_;
    gpu::ComputeKernel boundsChunks_;
    gpu::ComputeKernel boundsReduce_;
};

/// Whether `path` names a SOG: a .sog bundle or an unbundled meta.json.
[[nodiscard]] bool isSog(const std::filesystem::path& path);

/// Any splat file onto the device: .ply, .splat, .spz through io::readSplats,
/// SOG through io::readSog and the device decode.
[[nodiscard]] Result<GpuSplats> loadSplatFile(CloudLoader& loader, const std::filesystem::path& path,
                                              uint32_t maxDegree = 3);

/// Any splat file as host records in the engine's float encoding, for what
/// consumes records (the USD export). SOG is decoded on the device and read back.
[[nodiscard]] Result<io::RawSplats> readSplatRecords(CloudLoader& loader, const std::filesystem::path& path,
                                                     uint32_t maxDegree = 3);

}   // namespace lrt::scene
