// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/core/Platform.h"

#include <cstdlib>
#include <utility>

#if defined(_WIN32)
#error "Windows is a later port: implement MappedFile with CreateFileMapping here."
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#elif defined(__linux__)
#include <link.h>
#include <sys/prctl.h>
#endif

#include <thread>

namespace lrt::platform {

Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    struct stat info {};
    if (::fstat(fd, &info) != 0) {
        ::close(fd);
        return Error::make(ErrorCode::IoFailure, "cannot stat '{}'", path.string());
    }
    MappedFile file;
    file.size_ = static_cast<size_t>(info.st_size);
#if defined(__APPLE__)
    file.mtimeNs_ = int64_t{info.st_mtimespec.tv_sec} * 1'000'000'000 +
                    info.st_mtimespec.tv_nsec;
#else
    file.mtimeNs_ = int64_t{info.st_mtim.tv_sec} * 1'000'000'000 + info.st_mtim.tv_nsec;
#endif
    if (file.size_ > 0) {
        void* mapped = ::mmap(nullptr, file.size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            return Error::make(ErrorCode::IoFailure, "cannot map '{}'", path.string());
        }
        file.data_ = mapped;
    }
    ::close(fd);   // the mapping keeps the file alive
    return file;
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      mtimeNs_(other.mtimeNs_) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            ::munmap(data_, size_);
        }
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        mtimeNs_ = other.mtimeNs_;
    }
    return *this;
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
    }
}

void sleepPrecisely(std::chrono::nanoseconds duration) {
    if (duration.count() <= 0) {
        return;
    }
#if defined(__APPLE__)
    const mach_port_t self = pthread_mach_thread_np(pthread_self());
    static const double absPerNs = [] {
        mach_timebase_info_data_t base{};
        mach_timebase_info(&base);
        return static_cast<double>(base.denom) / static_cast<double>(base.numer);
    }();
    thread_time_constraint_policy_data_t realtime{};
    realtime.period = 0;
    realtime.computation = static_cast<uint32_t>(1e6 * absPerNs);
    realtime.constraint = static_cast<uint32_t>(2e6 * absPerNs);
    realtime.preemptible = 1;
    thread_policy_set(self, THREAD_TIME_CONSTRAINT_POLICY, reinterpret_cast<thread_policy_t>(&realtime),
                      THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    std::this_thread::sleep_for(duration);
    thread_standard_policy_data_t standard{};
    thread_policy_set(self, THREAD_STANDARD_POLICY, reinterpret_cast<thread_policy_t>(&standard),
                      THREAD_STANDARD_POLICY_COUNT);
#elif defined(__linux__)
    thread_local const bool slack = prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL) == 0;
    (void)slack;
    std::this_thread::sleep_for(duration);
#else
    std::this_thread::sleep_for(duration);
#endif
}

std::vector<std::string> loadedLibraries() {
    std::vector<std::string> out;
#if defined(__APPLE__)
    const uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        if (const char* name = _dyld_get_image_name(i); name != nullptr) {
            out.emplace_back(name);
        }
    }
#elif defined(__linux__)
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) {
            if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0') {
                static_cast<std::vector<std::string>*>(data)->emplace_back(info->dlpi_name);
            }
            return 0;
        },
        &out);
#endif
    return out;
}

std::filesystem::path cacheDirectory() {
#if defined(__APPLE__)
    return std::filesystem::path(env("HOME")) / "Library" / "Caches";
#elif defined(_WIN32)
    return std::filesystem::path(env("LOCALAPPDATA"));
#else
    if (std::string xdg = env("XDG_CACHE_HOME"); !xdg.empty()) {
        return xdg;
    }
    return std::filesystem::path(env("HOME")) / ".cache";
#endif
}

std::filesystem::path executableDir() {
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
        return std::filesystem::weakly_canonical(buffer.c_str()).parent_path();
    }
    return {};
#else
    std::error_code failed;
    auto self = std::filesystem::read_symlink("/proc/self/exe", failed);
    return failed ? std::filesystem::path{} : self.parent_path();
#endif
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
}

}   // namespace lrt::platform
