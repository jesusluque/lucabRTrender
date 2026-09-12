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
#include <objc/message.h>
#include <objc/runtime.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
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

void matchLayerToBacking(void* nsWindow) {
#if defined(__APPLE__)
    if (nsWindow == nullptr) {
        return;
    }
    // Through the Objective-C runtime, so this file stays C++.
    const auto send = [](void* receiver, const char* selector) {
        return reinterpret_cast<void* (*)(void*, SEL)>(objc_msgSend)(receiver, sel_registerName(selector));
    };
    const double scale =
        reinterpret_cast<double (*)(void*, SEL)>(objc_msgSend)(nsWindow, sel_registerName("backingScaleFactor"));
    void* view = send(nsWindow, "contentView");
    void* layer = view != nullptr ? send(view, "layer") : nullptr;
    if (layer != nullptr) {
        reinterpret_cast<void (*)(void*, SEL, double)>(objc_msgSend)(layer, sel_registerName("setContentsScale:"),
                                                                     scale);
    }
#else
    (void)nsWindow;
#endif
}

double extendedRangeHeadroom(void* nsWindow) {
#if defined(__APPLE__)
    if (nsWindow == nullptr) {
        return 1.0;
    }
    const auto send = [](void* receiver, const char* selector) {
        return reinterpret_cast<void* (*)(void*, SEL)>(objc_msgSend)(receiver, sel_registerName(selector));
    };
    void* screen = send(nsWindow, "screen");
    if (screen == nullptr) {
        return 1.0;
    }
    const double headroom = reinterpret_cast<double (*)(void*, SEL)>(objc_msgSend)(
        screen, sel_registerName("maximumExtendedDynamicRangeColorComponentValue"));
    return headroom > 1.0 ? headroom : 1.0;
#else
    (void)nsWindow;
    return 1.0;
#endif
}

bool enableExtendedRange(void* nsWindow) {
#if defined(__APPLE__)
    if (nsWindow == nullptr) {
        return false;
    }
    const auto send = [](void* receiver, const char* selector) {
        return reinterpret_cast<void* (*)(void*, SEL)>(objc_msgSend)(receiver, sel_registerName(selector));
    };
    void* view = send(nsWindow, "contentView");
    void* layer = view != nullptr ? send(view, "layer") : nullptr;
    if (layer == nullptr) {
        return false;
    }
    // CAMetalLayer: wantsExtendedDynamicRangeContent, and a colour space
    // whose 1.0 is the reference white with the range beyond it kept. The
    // colour space's name is the constant's own string.
    reinterpret_cast<void (*)(void*, SEL, BOOL)>(objc_msgSend)(
        layer, sel_registerName("setWantsExtendedDynamicRangeContent:"), YES);
    CFStringRef name = CFStringCreateWithCString(nullptr, "kCGColorSpaceExtendedLinearDisplayP3",
                                                 kCFStringEncodingUTF8);
    CGColorSpaceRef space = CGColorSpaceCreateWithName(name);
    CFRelease(name);
    if (space == nullptr) {
        return false;
    }
    reinterpret_cast<void (*)(void*, SEL, CGColorSpaceRef)>(objc_msgSend)(layer, sel_registerName("setColorspace:"),
                                                                          space);
    CGColorSpaceRelease(space);
    return true;
#else
    (void)nsWindow;
    return false;
#endif
}

void* newTrackedMetalBuffer(void* mtlDevice, uint64_t bytes) {
#if defined(__APPLE__)
    if (mtlDevice == nullptr || bytes == 0) {
        return nullptr;
    }
    // MTLResourceStorageModePrivate (2 << 4) | MTLResourceHazardTrackingModeTracked (2 << 8).
    const unsigned long options = (2UL << 4) | (2UL << 8);
    return reinterpret_cast<void* (*)(void*, SEL, unsigned long, unsigned long)>(objc_msgSend)(
        mtlDevice, sel_registerName("newBufferWithLength:options:"), static_cast<unsigned long>(bytes), options);
#else
    (void)mtlDevice;
    (void)bytes;
    return nullptr;
#endif
}

void releaseMetalBuffer(void* mtlBuffer) {
#if defined(__APPLE__)
    if (mtlBuffer != nullptr) {
        reinterpret_cast<void (*)(void*, SEL)>(objc_msgSend)(mtlBuffer, sel_registerName("release"));
    }
#else
    (void)mtlBuffer;
#endif
}

}   // namespace lrt::platform
