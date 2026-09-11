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
#endif

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
