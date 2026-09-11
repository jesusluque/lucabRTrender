// Copyright (c) 2026 lucabRTrender contributors.
#include <algorithm>
#include <cstring>
#include <string>

#include <tbb/parallel_for.h>

#include "lrt/core/Log.h"
#include "lrt/core/Platform.h"
#include "lrt/io/PlyHeader.h"
#include "lrt/io/Readers.h"

#ifdef LRT_HAVE_SPZ
#include "load-spz.h"
#endif

namespace lrt::io {
namespace {

constexpr size_t kHeaderProbe = 1 << 16;
constexpr size_t kGrain = 1 << 16;   // records per parallel block

std::string lowerExtension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

}   // namespace

Result<RawSplats> readSplats(const std::filesystem::path& path) {
    const std::string ext = lowerExtension(path);
    if (ext == ".splat") {
        return readDotSplat(path);
    }
    if (ext == ".ply") {
        return readSplatPly(path);
    }
    if (ext == ".spz") {
        return readSpz(path);
    }
    return Error::make(ErrorCode::Unsupported, "'{}': no splat reader for '{}'", path.string(), ext);
}

Result<RawSplats> readSplatPly(const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) {
        return std::move(file).error();
    }
    const auto bytes = file->bytes();
    const std::string_view probe(reinterpret_cast<const char*>(bytes.data()),
                                 std::min(bytes.size(), kHeaderProbe));
    auto header = parsePlyHeader(probe);
    if (!header) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {}", path.string(),
                           header.error().message());
    }
    if (!header->binaryLittleEndian) {
        return Error::make(ErrorCode::Unsupported,
                           "'{}': PLY format '{}'; splat clouds are binary_little_endian",
                           path.string(), header->format);
    }
    const PlyElement* vertex = header->element("vertex");
    if (vertex == nullptr || vertex->count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no vertices", path.string());
    }
    if (vertex->hasList) {
        return Error::make(ErrorCode::Unsupported, "'{}': a list in the vertex element",
                           path.string());
    }
    if (vertex->count > 0xFFFFFFFFull) {
        return Error::make(ErrorCode::Unsupported, "'{}': more than 2^32 splats", path.string());
    }
    auto start = header->binaryOffsetOf("vertex");
    if (!start) {
        return std::move(start).error();
    }
    if (*start + vertex->stride * vertex->count > bytes.size()) {
        return Error::make(ErrorCode::IoFailure, "'{}': the file ends before its vertices do",
                           path.string());
    }

    static constexpr const char* kRequired[] = {"x",       "y",       "z",       "opacity",
                                                "scale_0", "scale_1", "scale_2", "rot_0",
                                                "rot_1",   "rot_2",   "rot_3",   "f_dc_0",
                                                "f_dc_1",  "f_dc_2"};
    for (const char* name : kRequired) {
        if (vertex->find(name) == nullptr) {
            return Error::make(ErrorCode::InvalidArgument,
                               "'{}': no '{}' -- this does not look like a trained splat cloud",
                               path.string(), name);
        }
    }

    // Every property becomes one float in the record, in the file's order.
    // When the file is all float32 that is the file's own bytes, copied; when
    // it is not (a double here, a uchar there) each value is converted to a
    // float -- a type conversion, not a decode.
    const bool allFloat32 = std::all_of(
        vertex->properties.begin(), vertex->properties.end(),
        [](const PlyProperty& p) { return p.type == "float" || p.type == "float32"; });
    const uint32_t floats = static_cast<uint32_t>(vertex->properties.size());
    const auto indexOf = [&](const char* name) -> uint32_t {
        const PlyProperty* p = vertex->find(name);
        return static_cast<uint32_t>(p - vertex->properties.data());
    };

    RawSplats raw;
    raw.source = path.string();
    raw.count = static_cast<uint32_t>(vertex->count);
    SplatEncoding& enc = raw.encoding;
    enc.floatsPerRecord = floats;
    enc.x = indexOf("x");
    enc.y = indexOf("y");
    enc.z = indexOf("z");
    enc.opacity = indexOf("opacity");
    enc.scale0 = indexOf("scale_0");
    enc.scale1 = indexOf("scale_1");
    enc.scale2 = indexOf("scale_2");
    enc.rotW = indexOf("rot_0");
    enc.rotX = indexOf("rot_1");
    enc.rotY = indexOf("rot_2");
    enc.rotZ = indexOf("rot_3");
    enc.dc0 = indexOf("f_dc_0");
    enc.dc1 = indexOf("f_dc_1");
    enc.dc2 = indexOf("f_dc_2");

    // The harmonics: counted, contiguous, colour-outermost. Degree is the
    // largest whole one the count covers.
    uint32_t restCount = 0;
    for (const PlyProperty& p : vertex->properties) {
        if (p.name.rfind("f_rest_", 0) == 0) {
            ++restCount;
        }
    }
    uint32_t perColour = 0;
    for (uint32_t basis : {15u, 8u, 3u}) {
        if (restCount >= basis * 3) {
            perColour = basis;
            break;
        }
    }
    if (perColour > 0) {
        enc.restBase = indexOf("f_rest_0");
        for (uint32_t k = 0; k < perColour * 3; ++k) {
            // The file's own per-colour count decides where green starts.
            const uint32_t filePerColour = restCount / 3;
            const uint32_t colour = k / perColour;
            const uint32_t basis = k % perColour;
            const std::string name = "f_rest_" + std::to_string(colour * filePerColour + basis);
            const PlyProperty* p = vertex->find(name);
            if (p == nullptr || indexOf(name.c_str()) != enc.restBase + colour * filePerColour + basis) {
                return Error::make(ErrorCode::Unsupported,
                                   "'{}': harmonics are not contiguous in the vertex element",
                                   path.string());
            }
        }
        // restPerColour is the kept count; the kernel indexes green as
        // restBase + filePerColour + basis, so it needs the file's count too.
        enc.restPerColour = perColour;
        if (restCount / 3 != perColour) {
            return Error::make(ErrorCode::Unsupported,
                               "'{}': {} harmonic coefficients is not a whole degree",
                               path.string(), restCount);
        }
    }
    enc.restColourOuter = 1;
    enc.opacity_ = SplatEncoding::Opacity::Logit;
    enc.scale_ = SplatEncoding::Scale::Log;
    enc.colour = SplatEncoding::Colour::ShDc;
    enc.rotation = SplatEncoding::Rotation::Float;

    raw.records.resize(size_t{raw.count} * floats);
    const char* const base = reinterpret_cast<const char*>(bytes.data()) + *start;
    const size_t stride = vertex->stride;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, raw.count, kGrain),
                      [&](const tbb::blocked_range<size_t>& range) {
        if (allFloat32) {
            std::memcpy(raw.records.data() + range.begin() * floats, base + range.begin() * stride,
                        (range.end() - range.begin()) * stride);
            return;
        }
        for (size_t i = range.begin(); i < range.end(); ++i) {
            const char* record = base + i * stride;
            float* out = raw.records.data() + i * floats;
            for (uint32_t k = 0; k < floats; ++k) {
                out[k] = static_cast<float>(readPlyValue(record, vertex->properties[k]));
            }
        }
    });
    return raw;
}

Result<RawSplats> readDotSplat(const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) {
        return std::move(file).error();
    }
    const auto bytes = file->bytes();
    constexpr size_t kRecord = 32;
    if (bytes.size() % kRecord != 0 || bytes.empty()) {
        return Error::make(ErrorCode::InvalidArgument,
                           "'{}': {} bytes is not a whole number of 32-byte .splat records",
                           path.string(), bytes.size());
    }
    // pos f32*3 | scale f32*3 | rgba u8*4 | rot u8*4 (w x y z, (v-128)/128)
    RawSplats raw;
    raw.source = path.string();
    raw.count = static_cast<uint32_t>(bytes.size() / kRecord);
    SplatEncoding& enc = raw.encoding;
    enc.floatsPerRecord = 14;
    enc.x = 0; enc.y = 1; enc.z = 2;
    enc.scale0 = 3; enc.scale1 = 4; enc.scale2 = 5;
    enc.dc0 = 6; enc.dc1 = 7; enc.dc2 = 8; enc.opacity = 9;
    enc.rotW = 10; enc.rotX = 11; enc.rotY = 12; enc.rotZ = 13;
    enc.opacity_ = SplatEncoding::Opacity::Byte;
    enc.scale_ = SplatEncoding::Scale::Linear;
    enc.colour = SplatEncoding::Colour::Byte;
    enc.rotation = SplatEncoding::Rotation::Byte;

    raw.records.resize(size_t{raw.count} * enc.floatsPerRecord);
    const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, raw.count, kGrain),
                      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
            const unsigned char* record = data + i * kRecord;
            float* out = raw.records.data() + i * enc.floatsPerRecord;
            std::memcpy(out, record, 24);   // six float32, as stored
            for (int k = 0; k < 8; ++k) {
                out[6 + k] = static_cast<float>(record[24 + k]);   // bytes as values
            }
        }
    });
    return raw;
}

Result<RawSplats> readSpz(const std::filesystem::path& path) {
#ifndef LRT_HAVE_SPZ
    return Error::make(ErrorCode::Unsupported, "'{}': this build reads no .spz (zstd was not found)",
                       path.string());
#else
    const spz::PackedGaussians packed = spz::loadSpzPacked(path.string());
    if (packed.numPoints <= 0) {
        return Error::make(ErrorCode::IoFailure, "'{}': not a readable SPZ file", path.string());
    }
    if (packed.usesFloat16()) {
        return Error::make(ErrorCode::Unsupported, "'{}': SPZ float16 positions (a format never released)",
                           path.string());
    }
    if (packed.hadSkippedExtensions) {
        log::warn("'{}': SPZ extensions ignored; its coordinates are read as right-up-back", path.string());
    }
    const auto n = static_cast<size_t>(packed.numPoints);
    const uint32_t shDim = packed.shDegree >= 4 ? 24 : packed.shDegree == 3 ? 15 : packed.shDegree == 2 ? 8
                                                   : packed.shDegree == 1 ? 3 : 0;
    const bool smallestThree = packed.usesQuaternionSmallestThree;
    if (packed.positions.size() < n * 9 || packed.alphas.size() < n || packed.colors.size() < n * 3 ||
        packed.scales.size() < n * 3 || packed.rotations.size() < n * (smallestThree ? 4 : 3) ||
        packed.sh.size() < n * shDim * 3) {
        return Error::make(ErrorCode::IoFailure, "'{}': SPZ attribute streams shorter than {} points",
                           path.string(), n);
    }

    RawSplats raw;
    raw.source = path.string();
    raw.count = static_cast<uint32_t>(n);
    SplatEncoding& enc = raw.encoding;
    // Up to degree 3 kept: the engine evaluates no further, and a degree-4
    // file's fourth band is 27 floats a splat for nothing.
    enc.floatsPerRecord = 14 + std::min(shDim, uint32_t{15}) * 3;
    enc.x = 0; enc.y = 1; enc.z = 2;
    enc.opacity = 3;
    enc.scale0 = 4; enc.scale1 = 5; enc.scale2 = 6;
    enc.dc0 = 7; enc.dc1 = 8; enc.dc2 = 9;
    enc.rotX = 10; enc.rotY = 11; enc.rotZ = 12; enc.rotW = 13;
    enc.restBase = 14;
    enc.restPerColour = std::min(shDim, uint32_t{15});   // the engine keeps up to degree 3
    enc.restColourOuter = 0;                               // rgb per basis
    enc.opacity_ = SplatEncoding::Opacity::Byte;
    enc.scale_ = SplatEncoding::Scale::SpzByte;
    enc.colour = SplatEncoding::Colour::SpzByte;
    enc.rotation = smallestThree ? SplatEncoding::Rotation::SmallestThree : SplatEncoding::Rotation::FirstThree;
    enc.rest = SplatEncoding::Rest::Byte;
    enc.positionScale = 1.0F / static_cast<float>(1 << packed.fractionalBits);
    enc.flipYZ = true;

    raw.records.assign(n * enc.floatsPerRecord, 0.0F);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n, kGrain), [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
            float* out = raw.records.data() + i * enc.floatsPerRecord;
            const uint8_t* p = packed.positions.data() + i * 9;
            for (int k = 0; k < 3; ++k) {
                // A little-endian 24-bit signed integer: parsed, not scaled.
                int32_t fixed = p[k * 3] | (p[k * 3 + 1] << 8) | (p[k * 3 + 2] << 16);
                if ((fixed & 0x800000) != 0) {
                    fixed |= static_cast<int32_t>(0xFF000000u);
                }
                out[k] = static_cast<float>(fixed);
            }
            out[3] = packed.alphas[i];
            for (int k = 0; k < 3; ++k) {
                out[4 + k] = packed.scales[i * 3 + static_cast<size_t>(k)];
                out[7 + k] = packed.colors[i * 3 + static_cast<size_t>(k)];
            }
            if (smallestThree) {
                const uint8_t* r = packed.rotations.data() + i * 4;
                out[10] = static_cast<float>(r[0] | (r[1] << 8));
                out[11] = static_cast<float>(r[2] | (r[3] << 8));
            } else {
                const uint8_t* r = packed.rotations.data() + i * 3;
                out[10] = r[0];
                out[11] = r[1];
                out[12] = r[2];
            }
            // Only the first 15 bases (degree 3): the file's own count says
            // where the next point starts.
            for (uint32_t k = 0; k < enc.restPerColour * 3; ++k) {
                out[14 + k] = packed.sh[i * shDim * 3 + k];
            }
        }
    });
    return raw;
#endif
}

}   // namespace lrt::io
