// Copyright (c) 2026 lucabRTrender contributors.
//
// Synthetic splat clouds for the render tests: records written in the 3DGS
// PLY encoding, so they go through the same GPU decode a file does.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "lrt/io/RawSplats.h"

namespace lrt::test {

/// Degree-1 records in the 3DGS encoding (logit opacity, log scale, SH DC).
struct CloudBuilder {
    io::RawSplats raw;

    CloudBuilder() {
        io::SplatEncoding& e = raw.encoding;
        e.floatsPerRecord = 23;
        e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
        e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
        e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
        e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
        e.restBase = 14; e.restPerColour = 3; e.restColourOuter = 1;
        raw.source = "synthetic";
    }

    /// `colour` is the base colour wanted (0.5 + SH0 * dc).
    void add(float x, float y, float z, float opacity, float sx, float sy, float sz,
             std::array<float, 4> wxyz, std::array<float, 3> colour,
             std::array<float, 9> rest = {}) {
        const auto logit = [](float a) { return std::log(a / (1.0F - a)); };
        const float sh0 = 0.28209479F;
        std::array<float, 23> r{};
        r = {x, y, z, logit(opacity), std::log(sx), std::log(sy), std::log(sz),
             wxyz[0], wxyz[1], wxyz[2], wxyz[3],
             (colour[0] - 0.5F) / sh0, (colour[1] - 0.5F) / sh0, (colour[2] - 0.5F) / sh0};
        for (size_t k = 0; k < 9; ++k) {
            r[14 + k] = rest[k];
        }
        raw.records.insert(raw.records.end(), r.begin(), r.end());
        raw.count += 1;
    }
};

struct Lcg {
    uint64_t state = 0x853c49e6748fea9bULL;
    float next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0F;
    }
    float range(float lo, float hi) { return lo + (hi - lo) * next(); }
};

inline CloudBuilder randomCloud(uint32_t count, uint64_t seed, float minSize = 0.004F,
                                float maxSize = 0.6F, float minOpacity = 0.02F,
                                float maxOpacity = 0.99F) {
    CloudBuilder b;
    Lcg rng;
    rng.state ^= seed;
    for (uint32_t i = 0; i < count; ++i) {
        const float size = std::exp(rng.range(std::log(minSize), std::log(maxSize)));
        const float stretch = rng.range(0.05F, 1.0F);
        std::array<float, 9> rest{};
        for (float& v : rest) {
            v = rng.range(-0.3F, 0.3F);
        }
        // Drawn into names first: the order a call evaluates its arguments
        // in is unspecified, and GCC takes them right to left where Clang
        // takes them left to right -- the same seed made another cloud on
        // Linux, which the EWA comparison measured as a different scene.
        const float x = rng.range(-2.0F, 2.0F);
        const float y = rng.range(-1.5F, 1.5F);
        const float z = rng.range(-2.0F, 2.0F);
        const float opacity = rng.range(minOpacity, maxOpacity);
        const float thin = rng.range(0.1F, 1.0F);
        const std::array<float, 4> turn{rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)};
        const std::array<float, 3> dc{rng.range(0.0F, 1.2F), rng.range(0.0F, 1.2F), rng.range(0.0F, 1.2F)};
        b.add(x, y, z, opacity, size, size * stretch, size * thin, {turn[0], turn[1], turn[2], turn[3]},
              {dc[0], dc[1], dc[2]}, rest);
    }
    return b;
}

}   // namespace lrt::test
