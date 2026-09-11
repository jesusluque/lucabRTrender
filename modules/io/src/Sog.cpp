// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/io/Sog.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>

#include <nlohmann/json.hpp>
#include <zlib.h>

#ifdef LRT_HAVE_WEBP
#include <webp/decode.h>
#endif

namespace lrt::io {
namespace {

using Bytes = std::vector<uint8_t>;

uint16_t u16(const Bytes& b, size_t at) {
    return static_cast<uint16_t>(b[at] | (b[at + 1] << 8));
}
uint32_t u32(const Bytes& b, size_t at) {
    return static_cast<uint32_t>(b[at]) | (static_cast<uint32_t>(b[at + 1]) << 8) |
           (static_cast<uint32_t>(b[at + 2]) << 16) | (static_cast<uint32_t>(b[at + 3]) << 24);
}

Result<Bytes> readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    Bytes bytes(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

/// The entries of a zip archive by name: stored or deflated, no zip64 -- what
/// SOG bundles are.
Result<std::map<std::string, Bytes>> readZip(const std::filesystem::path& path) {
    auto file = readFile(path);
    if (!file) return std::move(file).error();
    const Bytes& z = *file;
    if (z.size() < 22) {
        return Error::make(ErrorCode::InvalidArgument, "'{}' is not a zip bundle", path.string());
    }
    // The end-of-central-directory record, allowing for an archive comment.
    size_t eocd = z.size() - 22;
    const size_t floor = z.size() > 22 + 65535 ? z.size() - 22 - 65535 : 0;
    while (eocd > floor && u32(z, eocd) != 0x06054b50u) {
        --eocd;
    }
    if (u32(z, eocd) != 0x06054b50u) {
        return Error::make(ErrorCode::InvalidArgument, "'{}' is not a zip bundle", path.string());
    }
    const uint16_t entries = u16(z, eocd + 10);
    size_t at = u32(z, eocd + 16);
    std::map<std::string, Bytes> out;
    for (uint16_t e = 0; e < entries; ++e) {
        if (at + 46 > z.size() || u32(z, at) != 0x02014b50u) {
            return Error::make(ErrorCode::InvalidArgument, "'{}': a broken zip directory", path.string());
        }
        const uint16_t method = u16(z, at + 10);
        const uint32_t packed = u32(z, at + 20);
        const uint32_t unpacked = u32(z, at + 24);
        const uint16_t nameLength = u16(z, at + 28);
        const uint16_t extraLength = u16(z, at + 30);
        const uint16_t commentLength = u16(z, at + 32);
        const uint32_t local = u32(z, at + 42);
        if (packed == 0xFFFFFFFFu || unpacked == 0xFFFFFFFFu || local == 0xFFFFFFFFu) {
            return Error::make(ErrorCode::Unsupported, "'{}': zip64 bundles are not read", path.string());
        }
        std::string name(reinterpret_cast<const char*>(z.data()) + at + 46, nameLength);
        if (local + 30 > z.size() || u32(z, local) != 0x04034b50u) {
            return Error::make(ErrorCode::InvalidArgument, "'{}': a broken zip entry", path.string());
        }
        const size_t payload = local + 30u + u16(z, local + 26) + u16(z, local + 28);
        if (payload + packed > z.size()) {
            return Error::make(ErrorCode::InvalidArgument, "'{}': a truncated zip entry", path.string());
        }
        Bytes data(unpacked);
        if (method == 0) {
            std::memcpy(data.data(), z.data() + payload, unpacked);
        } else if (method == 8) {
            z_stream stream{};
            // A zip entry is a raw deflate stream: negative window bits, no header.
            if (inflateInit2(&stream, -15) != Z_OK) {
                return Error(ErrorCode::InternalError, "zlib would not start");
            }
            stream.next_in = const_cast<Bytef*>(z.data() + payload);
            stream.avail_in = packed;
            stream.next_out = data.data();
            stream.avail_out = unpacked;
            const int status = inflate(&stream, Z_FINISH);
            inflateEnd(&stream);
            if (status != Z_STREAM_END || stream.total_out != unpacked) {
                return Error::make(ErrorCode::InvalidArgument, "'{}': '{}' does not inflate", path.string(), name);
            }
        } else {
            return Error::make(ErrorCode::Unsupported, "'{}': zip method {} for '{}'", path.string(), method, name);
        }
        out[std::move(name)] = std::move(data);
        at += 46u + nameLength + extraLength + commentLength;
    }
    return out;
}

Result<SogImage> decodeWebp(const std::map<std::string, Bytes>& files, const std::string& name,
                            bool required, const std::string& source) {
    const auto found = files.find(name);
    if (found == files.end()) {
        if (required) {
            return Error::make(ErrorCode::InvalidArgument, "'{}': no {}", source, name);
        }
        return SogImage{};
    }
#ifndef LRT_HAVE_WEBP
    return Error::make(ErrorCode::Unsupported, "'{}': this build decodes no WebP (libwebp was not found)", source);
#else
    int width = 0;
    int height = 0;
    // Straight RGBA off the decoder: an index premultiplied by alpha and
    // divided back is a different index.
    uint8_t* rgba = WebPDecodeRGBA(found->second.data(), found->second.size(), &width, &height);
    if (rgba == nullptr) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {} is not a WebP image", source, name);
    }
    SogImage image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.texels.resize(size_t{image.width} * image.height);
    for (size_t i = 0; i < image.texels.size(); ++i) {
        const uint8_t* p = rgba + i * 4;
        image.texels[i] = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                          (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    }
    WebPFree(rgba);
    return image;
#endif
}

void rangeOf(const nlohmann::json& section, float* mins, float* maxs, size_t n) {
    const auto read = [&](const char* key, float* into) {
        const auto found = section.find(key);
        if (found == section.end()) {
            return;
        }
        for (size_t k = 0; k < n; ++k) {
            if (found->is_array() && k < found->size() && (*found)[k].is_number()) {
                into[k] = (*found)[k].get<float>();
            } else if (found->is_number()) {
                into[k] = found->get<float>();
            }
        }
    };
    read("mins", mins);
    read("maxs", maxs);
}

std::vector<float> codebookOf(const nlohmann::json& section) {
    std::vector<float> book;
    const auto found = section.find("codebook");
    if (found != section.end() && found->is_array()) {
        for (const auto& v : *found) {
            book.push_back(v.is_number() ? v.get<float>() : 0.0F);
        }
    }
    return book;
}

}   // namespace

bool sogSupported() noexcept {
#ifdef LRT_HAVE_WEBP
    return true;
#else
    return false;
#endif
}

Result<RawSog> readSog(const std::filesystem::path& path) {
    std::map<std::string, Bytes> files;
    if (path.extension() == ".json") {
        std::error_code ignored;
        for (const auto& entry : std::filesystem::directory_iterator(path.parent_path(), ignored)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name != path.filename().string() && entry.path().extension() != ".webp") {
                continue;
            }
            auto bytes = readFile(entry.path());
            if (!bytes) return std::move(bytes).error();
            files[name == path.filename().string() ? "meta.json" : name] = std::move(*bytes);
        }
    } else {
        auto zip = readZip(path);
        if (!zip) return std::move(zip).error();
        files = std::move(*zip);
    }
    const std::string source = path.string();
    const auto meta = files.find("meta.json");
    if (meta == files.end()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no meta.json", source);
    }
    const nlohmann::json root = nlohmann::json::parse(meta->second.begin(), meta->second.end(), nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': meta.json does not parse", source);
    }

    RawSog sog;
    sog.source = source;
    sog.version = root.value("version", 1u);
    sog.count = root.value("count", 0u);
    if (sog.version != 1 && sog.version != 2) {
        return Error::make(ErrorCode::Unsupported, "'{}': SOG version {}", source, sog.version);
    }
    const nlohmann::json empty = nlohmann::json::object();
    const auto section = [&](const char* key) -> const nlohmann::json& {
        const auto found = root.find(key);
        return found != root.end() && found->is_object() ? *found : empty;
    };
    rangeOf(section("means"), sog.meansMin.data(), sog.meansMax.data(), 3);
    rangeOf(section("scales"), sog.scalesMin.data(), sog.scalesMax.data(), 3);
    rangeOf(section("sh0"), sog.sh0Min.data(), sog.sh0Max.data(), 4);
    rangeOf(section("shN"), &sog.shNMin, &sog.shNMax, 1);
    sog.scalesBook = codebookOf(section("scales"));
    sog.sh0Book = codebookOf(section("sh0"));
    sog.shNBook = codebookOf(section("shN"));
    if (sog.version == 2 && (sog.scalesBook.size() != 256 || sog.sh0Book.size() != 256)) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': a version 2 SOG without its 256-entry codebooks",
                           source);
    }

    const auto image = [&](const char* name, bool required, SogImage& into) -> Result<void> {
        auto decoded = decodeWebp(files, name, required, source);
        if (!decoded) return std::move(decoded).error();
        into = std::move(*decoded);
        return ok();
    };
    LRT_TRY(image("means_l.webp", true, sog.meansL));
    LRT_TRY(image("means_u.webp", true, sog.meansU));
    LRT_TRY(image("quats.webp", true, sog.quats));
    LRT_TRY(image("scales.webp", true, sog.scales));
    LRT_TRY(image("sh0.webp", true, sog.sh0));
    LRT_TRY(image("shN_centroids.webp", false, sog.shCentroids));
    LRT_TRY(image("shN_labels.webp", false, sog.shLabels));

    for (const SogImage* attribute : {&sog.meansL, &sog.meansU, &sog.quats, &sog.scales, &sog.sh0}) {
        if (attribute->texels.size() < sog.count) {
            return Error::make(ErrorCode::InvalidArgument, "'{}': an attribute image smaller than {} splats",
                               source, sog.count);
        }
    }
    if (!sog.shCentroids.empty() && !sog.shLabels.empty() && sog.shLabels.texels.size() >= sog.count) {
        const uint32_t perEntry = sog.shCentroids.width / 64;
        sog.shCoefficients = perEntry >= 15 ? 15 : perEntry >= 8 ? 8 : perEntry >= 3 ? 3 : 0;
        if (sog.version == 2 && sog.shNBook.size() != 256) {
            sog.shCoefficients = 0;
        }
    }
    if (sog.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': meta.json names no splats", source);
    }
    return sog;
}

}   // namespace lrt::io
