// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/io/Exr.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TINYEXR_USE_MINIZ 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace lrt::io {

namespace {

void put(std::vector<uint8_t>& into, const void* data, size_t bytes) {
    const auto* b = static_cast<const uint8_t*>(data);
    into.insert(into.end(), b, b + bytes);
}

}   // namespace

ExrAttribute ExrAttribute::timecode(std::string name, uint32_t timeAndFlags, uint32_t userData) {
    ExrAttribute a{std::move(name), "timecode", {}};
    put(a.value, &timeAndFlags, 4);
    put(a.value, &userData, 4);
    return a;
}

ExrAttribute ExrAttribute::rational(std::string name, int32_t numerator, uint32_t denominator) {
    ExrAttribute a{std::move(name), "rational", {}};
    put(a.value, &numerator, 4);
    put(a.value, &denominator, 4);
    return a;
}

ExrAttribute ExrAttribute::text(std::string name, const std::string& value) {
    ExrAttribute a{std::move(name), "string", {}};
    put(a.value, value.data(), value.size());
    return a;
}

ExrAttribute ExrAttribute::float64(std::string name, double value) {
    ExrAttribute a{std::move(name), "double", {}};
    put(a.value, &value, 8);
    return a;
}

Result<void> writeExrChannels(const std::filesystem::path& path, uint32_t width, uint32_t height,
                              std::span<const ExrChannel> channels, std::span<const ExrAttribute> attributes) {
    const size_t pixels = size_t{width} * height;
    if (channels.empty()) {
        return Error(ErrorCode::InvalidArgument, "no channels to write");
    }
    for (const ExrChannel& channel : channels) {
        if (channel.words.size() < pixels) {
            return Error::make(ErrorCode::InvalidArgument, "channel '{}': not enough pixels to write", channel.name);
        }
        if (channel.name.empty() || channel.name.size() > 255) {
            return Error(ErrorCode::InvalidArgument, "a channel's name is empty or longer than 255");
        }
    }
    // The format stores channels by name; the file's rows run top first and
    // the engine's bottom first.
    std::vector<size_t> order(channels.size());
    for (size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return channels[a].name < channels[b].name; });
    std::vector<std::vector<uint32_t>> planes(channels.size(), std::vector<uint32_t>(pixels));
    for (size_t k = 0; k < order.size(); ++k) {
        const ExrChannel& channel = channels[order[k]];
        for (uint32_t y = 0; y < height; ++y) {
            std::memcpy(planes[k].data() + size_t{y} * width, channel.words.data() + size_t{height - 1 - y} * width,
                        size_t{width} * 4);
        }
    }

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);
    std::vector<unsigned char*> pointers;
    for (auto& plane : planes) {
        pointers.push_back(reinterpret_cast<unsigned char*>(plane.data()));
    }
    image.num_channels = static_cast<int>(channels.size());
    image.images = pointers.data();
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);

    std::vector<EXRChannelInfo> info(channels.size());
    std::vector<int> pixelTypes(channels.size());
    std::vector<int> requested(channels.size());
    for (size_t k = 0; k < order.size(); ++k) {
        const ExrChannel& channel = channels[order[k]];
        std::memset(&info[k], 0, sizeof(EXRChannelInfo));
        std::strncpy(info[k].name, channel.name.c_str(), 255);
        pixelTypes[k] = channel.type == ExrChannelType::Uint ? TINYEXR_PIXELTYPE_UINT : TINYEXR_PIXELTYPE_FLOAT;
        requested[k] = channel.type == ExrChannelType::Uint   ? TINYEXR_PIXELTYPE_UINT
                       : channel.type == ExrChannelType::Half ? TINYEXR_PIXELTYPE_HALF
                                                              : TINYEXR_PIXELTYPE_FLOAT;
    }
    header.num_channels = static_cast<int>(channels.size());
    header.channels = info.data();
    header.pixel_types = pixelTypes.data();
    header.requested_pixel_types = requested.data();
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
    std::vector<EXRAttribute> custom(attributes.size());
    for (size_t k = 0; k < attributes.size(); ++k) {
        std::strncpy(custom[k].name, attributes[k].name.c_str(), 255);
        std::strncpy(custom[k].type, attributes[k].type.c_str(), 255);
        // tinyexr only reads through the pointer while writing.
        custom[k].value = const_cast<unsigned char*>(attributes[k].value.data());
        custom[k].size = static_cast<int>(attributes[k].value.size());
    }
    header.num_custom_attributes = static_cast<int>(custom.size());
    header.custom_attributes = custom.empty() ? nullptr : custom.data();

    const char* message = nullptr;
    const int status = SaveEXRImageToFile(&image, &header, path.string().c_str(), &message);
    if (status != TINYEXR_SUCCESS) {
        std::string why = message != nullptr ? message : "unknown";
        FreeEXRErrorMessage(message);
        return Error::make(ErrorCode::IoFailure, "cannot write '{}': {}", path.string(), why);
    }
    return ok();
}

Result<void> writeExr(const std::filesystem::path& path, uint32_t width, uint32_t height,
                      std::span<const float> rgba, std::span<const float> depth, bool half,
                      std::span<const ExrAttribute> attributes) {
    const size_t pixels = size_t{width} * height;
    if (rgba.size() < pixels * 4) {
        return Error(ErrorCode::InvalidArgument, "not enough pixels to write");
    }
    const bool withDepth = depth.size() >= pixels;
    // R, G, B, A as planes of words, and Z. Depth stays float: half would
    // step at a few metres.
    std::array<std::vector<uint32_t>, 4> planes;
    for (auto& plane : planes) plane.resize(pixels);
    for (size_t p = 0; p < pixels; ++p) {
        for (size_t c = 0; c < 4; ++c) {
            std::memcpy(&planes[c][p], rgba.data() + p * 4 + c, 4);
        }
    }
    const ExrChannelType colourType = half ? ExrChannelType::Half : ExrChannelType::Float;
    std::vector<ExrChannel> channels{{"R", colourType, planes[0]},
                                     {"G", colourType, planes[1]},
                                     {"B", colourType, planes[2]},
                                     {"A", colourType, planes[3]}};
    if (withDepth) {
        channels.push_back({"Z", ExrChannelType::Float,
                            std::span<const uint32_t>(reinterpret_cast<const uint32_t*>(depth.data()), pixels)});
    }
    return writeExrChannels(path, width, height, channels, attributes);
}

Result<ExrChannels> readExrChannels(const std::filesystem::path& path) {
    EXRVersion version;
    const char* message = nullptr;
    if (ParseEXRVersionFromFile(&version, path.string().c_str()) != TINYEXR_SUCCESS) {
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': not an OpenEXR file", path.string());
    }
    EXRHeader header;
    InitEXRHeader(&header);
    if (ParseEXRHeaderFromFile(&header, &version, path.string().c_str(), &message) != TINYEXR_SUCCESS) {
        std::string why = message != nullptr ? message : "unknown";
        FreeEXRErrorMessage(message);
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), why);
    }
    // Half channels are asked for as float; the others as they are stored.
    for (int c = 0; c < header.num_channels; ++c) {
        if (header.pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
            header.requested_pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
        }
    }
    EXRImage image;
    InitEXRImage(&image);
    if (LoadEXRImageFromFile(&image, &header, path.string().c_str(), &message) != TINYEXR_SUCCESS) {
        std::string why = message != nullptr ? message : "unknown";
        FreeEXRErrorMessage(message);
        FreeEXRHeader(&header);
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), why);
    }
    ExrChannels out;
    out.width = static_cast<uint32_t>(image.width);
    out.height = static_cast<uint32_t>(image.height);
    const size_t pixels = size_t{out.width} * out.height;
    if (image.images == nullptr) {
        FreeEXRImage(&image);
        FreeEXRHeader(&header);
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': tiled images are not read", path.string());
    }
    for (int c = 0; c < image.num_channels; ++c) {
        ExrChannelData channel;
        channel.name = header.channels[c].name;
        channel.type = header.pixel_types[c] == TINYEXR_PIXELTYPE_UINT   ? ExrChannelType::Uint
                       : header.pixel_types[c] == TINYEXR_PIXELTYPE_HALF ? ExrChannelType::Half
                                                                          : ExrChannelType::Float;
        channel.words.resize(pixels);
        for (uint32_t y = 0; y < out.height; ++y) {
            std::memcpy(channel.words.data() + size_t{y} * out.width,
                        image.images[c] + size_t{out.height - 1 - y} * out.width * 4, size_t{out.width} * 4);
        }
        out.channels.push_back(std::move(channel));
    }
    FreeEXRImage(&image);
    FreeEXRHeader(&header);
    return out;
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

    EXRVersion version;
    EXRHeader header;
    InitEXRHeader(&header);
    if (ParseEXRVersionFromFile(&version, path.string().c_str()) == TINYEXR_SUCCESS &&
        ParseEXRHeaderFromFile(&header, &version, path.string().c_str(), &message) == TINYEXR_SUCCESS) {
        for (int k = 0; k < header.num_custom_attributes; ++k) {
            const EXRAttribute& a = header.custom_attributes[k];
            ExrAttribute attribute{a.name, a.type, {}};
            put(attribute.value, a.value, static_cast<size_t>(std::max(a.size, 0)));
            out.attributes.push_back(std::move(attribute));
        }
        FreeEXRHeader(&header);
    } else if (message != nullptr) {
        FreeEXRErrorMessage(message);
    }
    return out;
}

}   // namespace lrt::io
