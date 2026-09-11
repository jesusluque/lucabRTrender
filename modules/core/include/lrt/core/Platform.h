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

/// The directory holding the running executable.
[[nodiscard]] std::filesystem::path executableDir();

/// `name` from the environment, or empty.
[[nodiscard]] std::string env(const char* name);

}   // namespace lrt::platform
