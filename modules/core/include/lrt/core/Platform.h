// Copyright (c) 2026 lucabRTrender contributors.
//
// Every operating-system call the engine makes outside its dependencies lives
// behind this header. Linux and macOS are implemented; Windows is a later
// port, and this is the file it starts from.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lrt/core/Result.h"

namespace lrt::platform {

/// A read-only mapping of a whole file.
///
/// Mapped rather than read: a trained splat cloud is hundreds of megabytes and
/// the loader touches it in blocks from several threads, each of which should
/// fault in only the pages it reads.
class MappedFile {
public:
    [[nodiscard]] static Result<MappedFile> open(const std::filesystem::path& path);

    MappedFile() = default;
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {static_cast<const std::byte*>(data_), size_};
    }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    /// Modification time in nanoseconds, for cache keys (path@mtime:size).
    [[nodiscard]] int64_t modifiedNs() const noexcept { return mtimeNs_; }

private:
    void*   data_ = nullptr;
    size_t  size_ = 0;
    int64_t mtimeNs_ = 0;
};

/// Sleeps for `duration` and wakes on time, to tens of microseconds.
///
/// A plain sleep may overrun by the timer slack the OS allows itself: on macOS
/// that measured 7 ms on average and 10 at worst, a quarter of a 25 fps frame.
/// On macOS the thread holds a real-time (time constraint) policy only while
/// it sleeps -- held while it renders, a thread that overruns its computation
/// budget is demoted -- and wakes 36 µs late at worst. On Linux the thread's
/// timer slack goes to 1 ns.
void sleepPrecisely(std::chrono::nanoseconds duration);

/// Every shared library the process has loaded, by path.
[[nodiscard]] std::vector<std::string> loadedLibraries();

/// Where a user's caches go: ~/Library/Caches on macOS, $XDG_CACHE_HOME or
/// ~/.cache on Linux.
[[nodiscard]] std::filesystem::path cacheDirectory();

/// The directory holding the running executable.
[[nodiscard]] std::filesystem::path executableDir();

/// `name` from the environment, or empty.
[[nodiscard]] std::string env(const char* name);

/// A window's Metal layer drawn at the window's backing scale (2 on a Retina
/// display), so a drawable of the framebuffer's pixels maps one to one.
/// `nsWindow` is an NSWindow*; a no-op elsewhere.
void matchLayerToBacking(void* nsWindow);

/// The extended dynamic range a window's screen offers: its peak over its
/// reference white (1.0 on a standard display; 2 to 16 on an HDR one, as
/// its brightness stands). `nsWindow` is an NSWindow*; 1.0 elsewhere.
[[nodiscard]] double extendedRangeHeadroom(void* nsWindow);

/// A window's Metal layer asked for extended range content in linear
/// Display P3 (1.0 the reference white, values above it the headroom):
/// what a float surface shows as HDR. Returns whether it was set;
/// false elsewhere.
bool enableExtendedRange(void* nsWindow);

/// A Metal buffer with private storage and hazard tracking, on `mtlDevice`:
/// what OIDN will share, and what slang-rhi -- which tracks nothing and
/// orders its own work -- will not make. Null off macOS or on failure. The
/// caller releases it with releaseMetalBuffer.
[[nodiscard]] void* newTrackedMetalBuffer(void* mtlDevice, uint64_t bytes);
void releaseMetalBuffer(void* mtlBuffer);

}   // namespace lrt::platform
