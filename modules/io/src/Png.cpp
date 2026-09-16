// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/io/Png.h"

#include <cstring>
#include <fstream>
#include <vector>

#include <zlib.h>

namespace lrt::io {
namespace {

void putBig(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value >> 24));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
}

/// One PNG chunk: length, type, data, CRC of type and data.
void chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
    putBig(out, static_cast<uint32_t>(data.size()));
    const size_t at = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uLong crc = crc32(crc32(0L, nullptr, 0), out.data() + at, static_cast<uInt>(out.size() - at));
    putBig(out, static_cast<uint32_t>(crc));
}

}   // namespace

Result<std::vector<uint8_t>> encodePng(uint32_t width, uint32_t height, std::span<const uint8_t> rgba) {
    const size_t wanted = size_t{width} * height * 4;
    if (width == 0 || height == 0 || rgba.size() < wanted) {
        return Error::make(ErrorCode::InvalidArgument, "a {}x{} PNG wants {} bytes, not {}", width, height, wanted,
                           rgba.size());
    }
    // Each row is preceded by its filter byte; none of them is filtered, which
    // costs size and keeps this a container and not a codec.
    std::vector<uint8_t> raw;
    raw.reserve(size_t{height} * (size_t{width} * 4 + 1));
    for (uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);
        const uint8_t* row = rgba.data() + size_t{y} * width * 4;
        raw.insert(raw.end(), row, row + size_t{width} * 4);
    }
    uLongf bound = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> deflated(bound);
    if (compress2(deflated.data(), &bound, raw.data(), static_cast<uLong>(raw.size()), Z_BEST_SPEED) != Z_OK) {
        return Error(ErrorCode::IoFailure, "zlib would not deflate the image");
    }
    deflated.resize(bound);

    std::vector<uint8_t> png{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> header;
    putBig(header, width);
    putBig(header, height);
    header.push_back(8);   // bits a channel
    header.push_back(6);   // RGBA
    header.push_back(0);   // deflate
    header.push_back(0);   // no filtering beyond the per-row byte
    header.push_back(0);   // no interlacing
    chunk(png, "IHDR", header);
    chunk(png, "IDAT", deflated);
    chunk(png, "IEND", {});
    return png;
}

Result<void> writePng(const std::filesystem::path& path, uint32_t width, uint32_t height,
                      std::span<const uint8_t> rgba) {
    auto encoded = encodePng(width, height, rgba);
    if (!encoded) return std::move(encoded).error();
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}'", path.string());
    }
    out.write(reinterpret_cast<const char*>(encoded->data()), static_cast<std::streamsize>(encoded->size()));
    if (!out) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}'", path.string());
    }
    return ok();
}

}   // namespace lrt::io
