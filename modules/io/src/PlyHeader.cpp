// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/io/PlyHeader.h"

#include <cstdint>
#include <cstring>
#include <sstream>

namespace lrt::io {
namespace {

size_t sizeOfType(const std::string& type) {
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
    if (type == "double" || type == "float64") return 8;
    return 4;
}

bool isFloatType(const std::string& type) {
    return type == "float" || type == "float32" || type == "double" || type == "float64";
}

}   // namespace

const PlyProperty* PlyElement::find(std::string_view property) const {
    for (const PlyProperty& p : properties) {
        if (p.name == property) {
            return &p;
        }
    }
    return nullptr;
}

const PlyElement* PlyHeader::element(std::string_view name) const {
    for (const PlyElement& e : elements) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

Result<size_t> PlyHeader::binaryOffsetOf(std::string_view name) const {
    size_t offset = bytes;
    for (const PlyElement& e : elements) {
        if (e.name == name) {
            return offset;
        }
        if (e.hasList) {
            return Error::make(ErrorCode::Unsupported,
                               "a list element ('{}') before '{}' makes its offset unknowable",
                               e.name, name);
        }
        offset += e.stride * e.count;
    }
    return Error::make(ErrorCode::NotFound, "no '{}' element", name);
}

Result<PlyHeader> parsePlyHeader(std::string_view text) {
    PlyHeader out;
    size_t at = 0;
    const auto nextLine = [&](std::string& line) {
        if (at >= text.size()) {
            return false;
        }
        size_t end = text.find('\n', at);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        line.assign(text.substr(at, end - at));
        at = end + 1;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        return true;
    };
    std::string line;
    if (!nextLine(line) || line.rfind("ply", 0) != 0) {
        return Error(ErrorCode::InvalidArgument, "not a PLY file");
    }
    bool ended = false;
    while (nextLine(line)) {
        std::istringstream words(line);
        std::string word;
        words >> word;
        if (word == "format") {
            words >> out.format;
            out.ascii = out.format == "ascii";
            out.binaryLittleEndian = out.format == "binary_little_endian";
        } else if (word == "element") {
            PlyElement element;
            words >> element.name >> element.count;
            out.elements.push_back(std::move(element));
        } else if (word == "property") {
            if (out.elements.empty()) {
                return Error(ErrorCode::InvalidArgument, "a PLY property before any element");
            }
            PlyElement& element = out.elements.back();
            std::string type;
            words >> type;
            PlyProperty property;
            if (type == "list") {
                element.hasList = true;
                std::string countType, itemType;
                words >> countType >> itemType >> property.name;
                property.type = "list";
                element.properties.push_back(std::move(property));
                continue;
            }
            words >> property.name;
            property.type = type;
            property.offset = element.stride;
            property.size = sizeOfType(type);
            property.isFloat = isFloatType(type);
            element.stride += property.size;
            element.properties.push_back(std::move(property));
        } else if (word == "end_header") {
            ended = true;
            break;
        }
    }
    if (!ended) {
        return Error(ErrorCode::InvalidArgument, "PLY header has no end_header");
    }
    if (out.format.empty()) {
        return Error(ErrorCode::InvalidArgument, "PLY header has no format line");
    }
    out.bytes = at;
    return out;
}

double readPlyValue(const char* record, const PlyProperty& p) noexcept {
    const char* at = record + p.offset;
    if (p.type == "float" || p.type == "float32") {
        float v = 0.0F;
        std::memcpy(&v, at, sizeof v);
        return static_cast<double>(v);
    }
    if (p.type == "double" || p.type == "float64") {
        double v = 0.0;
        std::memcpy(&v, at, sizeof v);
        return v;
    }
    if (p.type == "uchar" || p.type == "uint8") return static_cast<uint8_t>(*at);
    if (p.type == "char" || p.type == "int8") return static_cast<int8_t>(*at);
    if (p.type == "ushort" || p.type == "uint16") {
        uint16_t v = 0;
        std::memcpy(&v, at, sizeof v);
        return v;
    }
    if (p.type == "short" || p.type == "int16") {
        int16_t v = 0;
        std::memcpy(&v, at, sizeof v);
        return v;
    }
    if (p.type == "uint" || p.type == "uint32") {
        uint32_t v = 0;
        std::memcpy(&v, at, sizeof v);
        return v;
    }
    int32_t v = 0;
    std::memcpy(&v, at, sizeof v);
    return v;
}

}   // namespace lrt::io
