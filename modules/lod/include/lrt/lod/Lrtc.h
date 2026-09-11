// Copyright (c) 2026 lucabRTrender contributors.
//
// .lrtc: a cloud with its levels of detail, as a file a stream reads chunk by
// chunk. Everything in it was computed on the device; the CPU copies bytes.
//
//   page 0        header
//   page 1..      level table, chunk table
//   aligned       the finest level's group starts
//   aligned       each level: positions, shape, SH, cells (per group)
//   aligned       each chunk: positions, shape, SH, finest group (per splat)
//
// Every block starts on a 4096-byte page, so a chunk maps and faults in alone.
// Little-endian, the device's own packing (lrt/common/packing.slang).
//
//   StreamingPool  the merged levels on the device from the start; the
//                  splats' chunks loaded by worker threads when a cut wants
//                  them, placed in a store of fixed size, most wanted first,
//                  and dropped when it is full and they are wanted least
#pragma once

#include <filesystem>
#include <memory>
#include <span>

#include "lrt/lod/Lod.h"

namespace lrt::lod {

/// Writes a cloud whose chunks are all on the device.
[[nodiscard]] Result<void> writeLrtc(gpu::Device& device, const LodCloud& cloud, const std::filesystem::path& path);

/// Reads the whole file onto the device: every chunk there, chunk c in slot c.
[[nodiscard]] Result<LodCloud> readLrtc(gpu::Device& device, const std::filesystem::path& path);

[[nodiscard]] bool isLrtc(const std::filesystem::path& path);

struct StreamingSettings {
    /// The store's size in splats, rounded down to whole chunks (at least one).
    uint64_t budgetSplats = uint64_t{4} << 20;
    uint32_t loaders = 2;
};

class StreamingPool {
public:
    [[nodiscard]] static Result<std::unique_ptr<StreamingPool>> open(gpu::Device& device,
                                                                     const std::filesystem::path& path,
                                                                     const StreamingSettings& settings = {});
    ~StreamingPool();

    /// Cut this, as any cloud; it changes only in update().
    [[nodiscard]] const LodCloud& cloud() const noexcept;

    /// What a cut of this frame wants (CutStats::needs), once per instance.
    void want(std::span<const uint32_t> needs);

    /// Ends the frame. Wanted chunks not on the device are queued, most wanted
    /// first, as many as there is a place for: a free slot, a chunk not wanted
    /// now (least recently wanted first), or a chunk wanted less than half as
    /// much -- the half keeps two chunks from trading places frame after
    /// frame. Chunks loaded since the last update are placed the same way.
    /// `wait` blocks until the queued ones have loaded and places them too: an
    /// offline render's way to a complete frame. Returns the chunks placed.
    [[nodiscard]] Result<uint32_t> update(bool wait = false);

    struct Status {
        uint32_t slots = 0;      ///< chunks the store holds
        uint32_t resident = 0;   ///< chunks on the device
        uint32_t missing = 0;    ///< wanted at the last update and not on the device
        uint32_t inFlight = 0;   ///< queued or loading
        uint64_t loads = 0;      ///< chunks placed since open
        uint64_t evictions = 0;
    };
    [[nodiscard]] Status status() const;

private:
    StreamingPool();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace lrt::lod
