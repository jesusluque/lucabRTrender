// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/io/Exr.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TINYEXR_USE_MINIZ 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace lrt::io {

Result<void> writeExr(const std::filesystem::path& path, uint32_t width, uint32_t height,
                      std::span<const float> rgba, std::span<const float> depth, bool half) {
    const size_t pixels = size_t{width} * height;
    if (rgba.size() < pixels * 4) {
        return Error(ErrorCode::InvalidArgument, "not enough pixels to write");
    }
    const bool withDepth = depth.size() >= pixels;
    // EXR stores the top row first; the engine renders the bottom row first.
    // Channels are stored in alphabetical order: A B G R (Z).
    const int channels = withDepth ? 5 : 4;
    std::vector<std::vector<float>> planes(static_cast<size_t>(channels),
                                           std::vector<float>(pixels));
    for (uint32_t y = 0; y < height; ++y) {
        const size_t from = size_t{height - 1 - y} * width;
        const size_t to = size_t{y} * width;
        for (uint32_t x = 0; x < width; ++x) {
            const float* const p = rgba.data() + (from + x) * 4;
            planes[0][to + x] = p[3];
            planes[1][to + x] = p[2];
            planes[2][to + x] = p[1];
            planes[3][to + x] = p[0];
            if (withDepth) {
                planes[4][to + x] = depth[from + x];
            }
        }
    }

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);
    std::vector<float*> pointers;
    for (auto& plane : planes) {
        pointers.push_back(plane.data());
    }
    image.num_channels = channels;
    image.images = reinterpret_cast<unsigned char**>(pointers.data());
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);

    std::array<EXRChannelInfo, 5> info{};
    static constexpr const char* kNames[5] = {"A", "B", "G", "R", "Z"};
    std::array<int, 5> pixelTypes{};
    std::array<int, 5> requested{};
    for (int c = 0; c < channels; ++c) {
        std::strncpy(info[static_cast<size_t>(c)].name, kNames[c], 255);
        pixelTypes[static_cast<size_t>(c)] = TINYEXR_PIXELTYPE_FLOAT;
        // Depth stays float: half would step at a few metres.
        requested[static_cast<size_t>(c)] =
            (half && c < 4) ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
    }
    header.num_channels = channels;
    header.channels = info.data();
    header.pixel_types = pixelTypes.data();
    header.requested_pixel_types = requested.data();
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;

    const char* message = nullptr;
    const int status = SaveEXRImageToFile(&image, &header, path.string().c_str(), &message);
    if (status != TINYEXR_SUCCESS) {
        std::string why = message != nullptr ? message : "unknown";
        FreeEXRErrorMessage(message);
        return Error::make(ErrorCode::IoFailure, "cannot write '{}': {}", path.string(), why);
    }
    return ok();
}

Result<ExrPixels> readExr(const std::filesystem::path& path) {
    float* rgba = nullptr;
    int width = 0;
    int height = 0;
    const char* message = nullptr;
    if (LoadEXR(&rgba, &width, &height, path.string().c_str(), &message) != TINYEXR_SUCCESS) {
        std::string why = message != nullptr ? message : "unknown";
        FreeEXRErrorMessage(message);
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), why);
    }
    ExrPixels out;
    out.width = static_cast<uint32_t>(width);
    out.height = static_cast<uint32_t>(height);
    out.rgba.resize(size_t{out.width} * out.height * 4);
    for (uint32_t y = 0; y < out.height; ++y) {
        std::memcpy(out.rgba.data() + size_t{y} * out.width * 4,
                    rgba + size_t{out.height - 1 - y} * out.width * 4, size_t{out.width} * 4 * sizeof(float));
    }
    std::free(rgba);
    return out;
}

}   // namespace lrt::io
