// Copyright (c) 2026 lucabRTrender contributors.
//
// Point files: every supported format ends as x y z r g b float records and a
// word on what the colour numbers mean. Parsing text into numbers happens
// here (there is no other place it can); what those numbers become --
// linearised, packed, kept or thinned -- is decided on the GPU.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include <tbb/parallel_for.h>

#include "lrt/core/Platform.h"
#include "lrt/io/PlyHeader.h"
#include "lrt/io/Readers.h"

namespace lrt::io {
namespace {

constexpr size_t kTextBlock = 1 << 22;   // bytes of text per parallel block

/// Splits `text` at newlines into roughly kTextBlock pieces.
std::vector<std::string_view> splitBlocks(std::string_view text) {
    std::vector<std::string_view> blocks;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = std::min(text.size(), start + kTextBlock);
        if (end < text.size()) {
            const size_t newline = text.find('\n', end);
            end = newline == std::string_view::npos ? text.size() : newline + 1;
        }
        blocks.push_back(text.substr(start, end - start));
        start = end;
    }
    return blocks;
}

/// Parses up to `max` numbers from a line; separators are spaces, tabs, commas.
size_t parseNumbers(std::string_view line, double* out, size_t max) {
    size_t found = 0;
    std::string buffer;
    size_t at = 0;
    while (at < line.size() && found < max) {
        while (at < line.size() && (line[at] == ' ' || line[at] == '\t' || line[at] == ',' ||
                                    line[at] == '\r')) {
            ++at;
        }
        size_t end = at;
        while (end < line.size() && line[end] != ' ' && line[end] != '\t' && line[end] != ',' &&
               line[end] != '\r') {
            ++end;
        }
        if (end == at) {
            break;
        }
        buffer.assign(line.substr(at, end - at));
        char* stop = nullptr;
        const double value = std::strtod(buffer.c_str(), &stop);
        if (stop == buffer.c_str()) {
            break;   // a word, not a number: the line ends here for us
        }
        out[found++] = value;
        at = end;
    }
    return found;
}

/// Text lines of numbers into records, in parallel, in file order.
/// `skip` leading numbers per line are ignored (COLMAP's point id).
RawPoints parseTextPoints(std::string_view text, size_t skip, size_t firstLineToParse,
                          size_t linesToParse) {
    std::vector<std::string_view> blocks = splitBlocks(text);
    std::vector<std::vector<float>> parsed(blocks.size());
    std::vector<uint8_t> anyByte(blocks.size(), 0);
    std::vector<uint8_t> anyColour(blocks.size(), 0);
    // Line limits (for an ASCII PLY's vertex element) need global line numbers,
    // so a limited parse runs as one block.
    const bool limited = linesToParse != static_cast<size_t>(-1) || firstLineToParse != 0;
    if (limited && blocks.size() > 1) {
        blocks = {text};
        parsed.resize(1);
        anyByte.resize(1);
        anyColour.resize(1);
    }
    tbb::parallel_for(size_t{0}, blocks.size(), [&](size_t b) {
        std::vector<float>& out = parsed[b];
        double values[16];
        size_t line = 0;
        size_t at = 0;
        const std::string_view block = blocks[b];
        while (at < block.size()) {
            size_t end = block.find('\n', at);
            if (end == std::string_view::npos) {
                end = block.size();
            }
            const std::string_view current = block.substr(at, end - at);
            at = end + 1;
            const size_t index = line++;
            if (limited && (index < firstLineToParse || index >= firstLineToParse + linesToParse)) {
                continue;
            }
            if (current.empty() || current[0] == '#') {
                continue;
            }
            const size_t n = parseNumbers(current, values, skip + 6);
            if (n < skip + 3) {
                continue;
            }
            out.push_back(static_cast<float>(values[skip]));
            out.push_back(static_cast<float>(values[skip + 1]));
            out.push_back(static_cast<float>(values[skip + 2]));
            if (n >= skip + 6) {
                anyColour[b] = 1;
                for (int c = 0; c < 3; ++c) {
                    const double v = values[skip + 3 + static_cast<size_t>(c)];
                    anyByte[b] |= v > 1.0 ? 1 : 0;
                    out.push_back(static_cast<float>(v));
                }
            } else {
                out.push_back(1.0F);
                out.push_back(1.0F);
                out.push_back(1.0F);
            }
        }
    });
    RawPoints points;
    size_t total = 0;
    for (const auto& block : parsed) {
        total += block.size();
    }
    points.records.reserve(total);
    for (const auto& block : parsed) {
        points.records.insert(points.records.end(), block.begin(), block.end());
    }
    points.count = static_cast<uint32_t>(points.records.size() / 6);
    const bool colour = std::any_of(anyColour.begin(), anyColour.end(), [](uint8_t v) { return v; });
    const bool bytes = std::any_of(anyByte.begin(), anyByte.end(), [](uint8_t v) { return v; });
    points.colourKind = !colour ? 0u : bytes ? 1u : 2u;
    return points;
}

Result<RawPoints> readPlyPoints(const std::filesystem::path& path, const platform::MappedFile& file) {
    const auto bytes = file.bytes();
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    auto header = parsePlyHeader(text.substr(0, std::min<size_t>(text.size(), 1 << 16)));
    if (!header) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {}", path.string(),
                           header.error().message());
    }
    const PlyElement* vertex = header->element("vertex");
    if (vertex == nullptr || vertex->count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no vertices", path.string());
    }
    const PlyProperty* px = vertex->find("x");
    const PlyProperty* py = vertex->find("y");
    const PlyProperty* pz = vertex->find("z");
    if (px == nullptr || py == nullptr || pz == nullptr) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no x y z", path.string());
    }
    const auto first = [&](std::initializer_list<const char*> names) -> const PlyProperty* {
        for (const char* name : names) {
            if (const PlyProperty* p = vertex->find(name)) {
                return p;
            }
        }
        return nullptr;
    };
    const PlyProperty* pr = first({"red", "r", "diffuse_red"});
    const PlyProperty* pg = first({"green", "g", "diffuse_green"});
    const PlyProperty* pb = first({"blue", "b", "diffuse_blue"});
    const bool colour = pr != nullptr && pg != nullptr && pb != nullptr;

    if (header->ascii) {
        // Lines before the vertex element belong to earlier elements.
        size_t skipLines = 0;
        for (const PlyElement& e : header->elements) {
            if (e.name == "vertex") {
                break;
            }
            skipLines += e.count;
        }
        if (vertex->hasList || px != &vertex->properties[0] || py != &vertex->properties[1] ||
            pz != &vertex->properties[2]) {
            return Error::make(ErrorCode::Unsupported,
                               "'{}': ASCII PLY points must start with x y z", path.string());
        }
        RawPoints points = parseTextPoints(text.substr(header->bytes), 0, skipLines, vertex->count);
        points.source = path.string();
        if (!colour) {
            points.colourKind = 0;
        }
        return points;
    }
    if (!header->binaryLittleEndian || vertex->hasList) {
        return Error::make(ErrorCode::Unsupported, "'{}': PLY format '{}'", path.string(),
                           header->format);
    }
    auto start = header->binaryOffsetOf("vertex");
    if (!start) {
        return std::move(start).error();
    }
    if (*start + vertex->stride * vertex->count > bytes.size()) {
        return Error::make(ErrorCode::IoFailure, "'{}': the file ends early", path.string());
    }
    RawPoints points;
    points.source = path.string();
    points.count = static_cast<uint32_t>(vertex->count);
    points.colourKind = !colour ? 0u : pr->isFloat ? 2u : 1u;
    points.records.resize(size_t{points.count} * 6);
    const char* const base = text.data() + *start;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, points.count, 1 << 16),
                      [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
            const char* record = base + i * vertex->stride;
            float* out = points.records.data() + i * 6;
            out[0] = static_cast<float>(readPlyValue(record, *px));
            out[1] = static_cast<float>(readPlyValue(record, *py));
            out[2] = static_cast<float>(readPlyValue(record, *pz));
            out[3] = colour ? static_cast<float>(readPlyValue(record, *pr)) : 1.0F;
            out[4] = colour ? static_cast<float>(readPlyValue(record, *pg)) : 1.0F;
            out[5] = colour ? static_cast<float>(readPlyValue(record, *pb)) : 1.0F;
        }
    });
    return points;
}

Result<RawPoints> readColmapBinary(const std::filesystem::path& path,
                                   const platform::MappedFile& file) {
    const auto bytes = file.bytes();
    const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
    size_t at = 0;
    const auto take = [&](void* into, size_t n) {
        if (at + n > bytes.size()) {
            return false;
        }
        std::memcpy(into, data + at, n);
        at += n;
        return true;
    };
    uint64_t count = 0;
    if (!take(&count, sizeof count)) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': not a COLMAP points3D.bin",
                           path.string());
    }
    RawPoints points;
    points.source = path.string();
    points.colourKind = 1;
    points.records.reserve(count * 6);
    // Sequential: each record's track has its own length, so where the next
    // one starts is known only by reading this one.
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t id = 0, trackLength = 0;
        double xyz[3] = {0, 0, 0};
        uint8_t rgb[3] = {0, 0, 0};
        double err = 0;
        if (!take(&id, 8) || !take(xyz, 24) || !take(rgb, 3) || !take(&err, 8) ||
            !take(&trackLength, 8)) {
            break;
        }
        at += trackLength * 8;
        points.records.insert(points.records.end(),
                              {static_cast<float>(xyz[0]), static_cast<float>(xyz[1]),
                               static_cast<float>(xyz[2]), static_cast<float>(rgb[0]),
                               static_cast<float>(rgb[1]), static_cast<float>(rgb[2])});
    }
    points.count = static_cast<uint32_t>(points.records.size() / 6);
    return points;
}

}   // namespace

Result<RawPoints> readPoints(const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) {
        return std::move(file).error();
    }
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const std::string name = path.filename().string();

    Result<RawPoints> points = Error(ErrorCode::Unsupported, "");
    if (ext == ".ply") {
        points = readPlyPoints(path, *file);
    } else if (name == "points3D.bin") {
        points = readColmapBinary(path, *file);
    } else if (ext == ".xyz" || ext == ".txt" || ext == ".pts" || ext == ".csv") {
        const auto bytes = file->bytes();
        const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        RawPoints parsed = parseTextPoints(text, name == "points3D.txt" ? 1 : 0, 0,
                                           static_cast<size_t>(-1));
        if (name == "points3D.txt" && parsed.colourKind != 0) {
            parsed.colourKind = 1;   // COLMAP colours are always 8-bit
        }
        parsed.source = path.string();
        points = std::move(parsed);
    } else {
        return Error::make(ErrorCode::Unsupported, "'{}': no point reader for '{}'", path.string(),
                           ext);
    }
    if (points && points->count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no points", path.string());
    }
    return points;
}

}   // namespace lrt::io
