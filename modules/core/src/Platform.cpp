// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/core/Platform.h"

#if defined(LRT_HAVE_EGL)
#define EGL_EGLEXT_PROTOTYPES 1   // the device extensions' declarations
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

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

int duplicateDescriptor(int descriptor) {
#if defined(_WIN32)
    (void)descriptor;
    return -1;
#else
    return ::dup(descriptor);
#endif
}

bool makeHeadlessGlContextCurrent() {
#if defined(LRT_HAVE_EGL)
    static EGLDisplay display = EGL_NO_DISPLAY;
    static EGLContext context = EGL_NO_CONTEXT;
    static EGLSurface surface = EGL_NO_SURFACE;
    static bool tried = false;
    if (!tried) {
        tried = true;
        // A display on the first GPU device, needing no X server.
        // glvnd's libEGL dispatches the device extension without exporting
        // it: found by name.
        using QueryDevices = EGLBoolean (*)(EGLint, EGLDeviceEXT*, EGLint*);
        const auto queryDevices = reinterpret_cast<QueryDevices>(eglGetProcAddress("eglQueryDevicesEXT"));
        EGLDeviceEXT devices[8];
        EGLint count = 0;
        if (queryDevices != nullptr && queryDevices(8, devices, &count) && count > 0) {
            for (EGLint k = 0; k < count && display == EGL_NO_DISPLAY; ++k) {
                EGLDisplay candidate = eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[k], nullptr);
                EGLint major = 0;
                EGLint minor = 0;
                if (candidate != EGL_NO_DISPLAY && eglInitialize(candidate, &major, &minor)) {
                    display = candidate;
                }
            }
        }
        if (display == EGL_NO_DISPLAY) {
            return false;
        }
        const EGLint attributes[] = {EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                     EGL_RED_SIZE,        8,               EGL_GREEN_SIZE,      8,
                                     EGL_BLUE_SIZE,       8,               EGL_DEPTH_SIZE,      24,
                                     EGL_NONE};
        EGLConfig config = nullptr;
        EGLint configs = 0;
        if (!eglChooseConfig(display, attributes, &config, 1, &configs) || configs == 0 ||
            !eglBindAPI(EGL_OPENGL_API)) {
            return false;
        }
        const EGLint pbuffer[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        surface = eglCreatePbufferSurface(display, config, pbuffer);
        // Compatibility, not core: Storm's state holder and draws use enums a
        // core profile refuses (measured: GL "invalid enum" and "invalid
        // operation" at every draw, and Kitchen_set drawn as nothing).
        const EGLint version[] = {EGL_CONTEXT_MAJOR_VERSION,       4,
                                  EGL_CONTEXT_MINOR_VERSION,       5,
                                  EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                                  EGL_NONE};
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, version);
        if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT) {
            return false;
        }
    }
    return context != EGL_NO_CONTEXT && eglMakeCurrent(display, surface, surface, context) == EGL_TRUE;
#else
    return false;
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

uint64_t pageSize() {
    return static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
}

void* mapPages(uint64_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    const uint64_t page = pageSize();
    const uint64_t whole = (bytes + page - 1) / page * page;
    void* pages = ::mmap(nullptr, static_cast<size_t>(whole), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    return pages == MAP_FAILED ? nullptr : pages;
}

void unmapPages(void* pages, uint64_t bytes) {
    if (pages == nullptr || bytes == 0) {
        return;
    }
    const uint64_t page = pageSize();
    ::munmap(pages, static_cast<size_t>((bytes + page - 1) / page * page));
}

void* newMetalBufferOverPages(void* mtlDevice, const void* pages, uint64_t bytes) {
#if defined(__APPLE__)
    if (mtlDevice == nullptr || pages == nullptr || bytes == 0) {
        return nullptr;
    }
    const uint64_t page = pageSize();
    if (reinterpret_cast<uintptr_t>(pages) % page != 0 || bytes % page != 0) {
        return nullptr;
    }
    // A discrete GPU would read these pages over the bus: not what borrowing
    // promises, so it is refused and the effect keeps a copy instead.
    const BOOL unified = reinterpret_cast<BOOL (*)(void*, SEL)>(objc_msgSend)(mtlDevice, sel_registerName("hasUnifiedMemory"));
    if (!unified) {
        return nullptr;
    }
    // MTLResourceStorageModeShared (0 << 4), no deallocator: the pages stay
    // the caller's.
    const unsigned long options = 0UL;
    return reinterpret_cast<void* (*)(void*, SEL, void*, unsigned long, unsigned long, void*)>(objc_msgSend)(
        mtlDevice, sel_registerName("newBufferWithBytesNoCopy:length:options:deallocator:"),
        const_cast<void*>(pages), static_cast<unsigned long>(bytes), options, nullptr);
#else
    (void)mtlDevice;
    (void)pages;
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
