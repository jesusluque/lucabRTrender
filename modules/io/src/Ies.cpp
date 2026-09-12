// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/io/Ies.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace lrt::io {

namespace {

/// The numbers after the TILT line, whatever separates them.
struct Numbers {
    std::vector<double> values;
    size_t              at = 0;
    bool take(size_t count, std::vector<float>& out) {
        if (at + count > values.size()) {
            return false;
        }
        out.reserve(out.size() + count);
        for (size_t k = 0; k < count; ++k) {
            out.push_back(static_cast<float>(values[at++]));
        }
        return true;
    }
    bool one(double& out) {
        if (at >= values.size()) {
            return false;
        }
        out = values[at++];
        return true;
    }
};

}   // namespace

Result<IesProfile> parseIes(std::string_view text) {
    // Keyword lines until TILT=; the profile follows it.
    const size_t tilt = text.find("TILT=");
    if (tilt == std::string_view::npos) {
        return Error(ErrorCode::InvalidArgument, "IES: no TILT line");
    }
    const size_t eol = text.find('\n', tilt);
    const std::string_view tiltValue = text.substr(tilt + 5, eol == std::string_view::npos ? std::string_view::npos : eol - tilt - 5);
    std::string_view rest = eol == std::string_view::npos ? std::string_view() : text.substr(eol + 1);
    Numbers numbers;
    {
        std::string body(rest);
        for (char& ch : body) {
            if (ch == ',') {
                ch = ' ';
            }
        }
        std::istringstream in(body);
        double v = 0.0;
        while (in >> v) {
            numbers.values.push_back(v);
        }
    }
    const bool include = tiltValue.find("INCLUDE") != std::string_view::npos;
    if (include) {
        // <lamp to luminaire geometry> <pairs> <angles> <factors>: read past.
        double geometry = 0.0;
        double pairs = 0.0;
        if (!numbers.one(geometry) || !numbers.one(pairs) || pairs < 0.0) {
            return Error(ErrorCode::InvalidArgument, "IES: TILT=INCLUDE without its pairs");
        }
        std::vector<float> skipped;
        if (!numbers.take(static_cast<size_t>(pairs) * 2, skipped)) {
            return Error(ErrorCode::InvalidArgument, "IES: TILT=INCLUDE pairs short");
        }
    }
    // <lamps> <lumens per lamp> <multiplier> <vertical> <horizontal> <type> <units> <width> <length> <height>
    double lamps = 0.0, lumens = 0.0, multiplier = 1.0, nVertical = 0.0, nHorizontal = 0.0, type = 1.0;
    double units = 0.0, width = 0.0, length = 0.0, height = 0.0;
    if (!numbers.one(lamps) || !numbers.one(lumens) || !numbers.one(multiplier) || !numbers.one(nVertical) ||
        !numbers.one(nHorizontal) || !numbers.one(type) || !numbers.one(units) || !numbers.one(width) ||
        !numbers.one(length) || !numbers.one(height)) {
        return Error(ErrorCode::InvalidArgument, "IES: the ten-number line is short");
    }
    // <ballast factor> <future use> <input watts>
    double ballast = 0.0, future = 0.0, watts = 0.0;
    if (!numbers.one(ballast) || !numbers.one(future) || !numbers.one(watts)) {
        return Error(ErrorCode::InvalidArgument, "IES: the three-number line is short");
    }
    if (nVertical < 1.0 || nHorizontal < 1.0 || nVertical > 4096.0 || nHorizontal > 4096.0) {
        return Error(ErrorCode::InvalidArgument, "IES: implausible angle counts");
    }
    IesProfile profile;
    profile.multiplier = static_cast<float>(multiplier);
    profile.photometricType = static_cast<uint32_t>(type);
    const size_t nv = static_cast<size_t>(nVertical);
    const size_t nh = static_cast<size_t>(nHorizontal);
    if (!numbers.take(nv, profile.vertical) || !numbers.take(nh, profile.horizontal) ||
        !numbers.take(nv * nh, profile.candela)) {
        return Error(ErrorCode::InvalidArgument, "IES: angle lists or candela table short");
    }
    for (size_t k = 1; k < nv; ++k) {
        if (!(profile.vertical[k] > profile.vertical[k - 1])) {
            return Error(ErrorCode::InvalidArgument, "IES: vertical angles not ascending");
        }
    }
    return profile;
}

Result<IesProfile> readIes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Error::make(ErrorCode::NotFound, "IES: cannot open '{}'", path.string());
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parseIes(text);
}

}   // namespace lrt::io
