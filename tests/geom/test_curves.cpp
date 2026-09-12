// Copyright (c) 2026 lucabRTrender contributors.
//
// Curve tubes against the curves: every tube vertex sits at half the width
// from the point it rings, that point being each basis's polynomial form
// written a second time in the check kernel -- so the builder's span
// bookkeeping (which control points a span takes, wrapped for a periodic
// curve) and the kernel's evaluation are both under test.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "lrt/geom/Curves.h"

using namespace lrt;

namespace {

struct SpanRecord {
    uint32_t first, curve, count, varyingFirst;
    uint32_t wrap[4];
};

/// The spans the builder must have laid out, for the check: the same rule.
std::vector<SpanRecord> spansOf(std::span<const int32_t> counts, geom::CurveBasis basis, geom::CurveWrap wrap) {
    std::vector<SpanRecord> out;
    const bool cubic = basis != geom::CurveBasis::Linear;
    const uint32_t vstep = basis == geom::CurveBasis::Bezier ? 3u : 1u;
    uint32_t at = 0;
    for (size_t c = 0; c < counts.size(); ++c) {
        const uint32_t n = static_cast<uint32_t>(counts[c]);
        uint32_t spans = 0;
        if (cubic) {
            spans = wrap == geom::CurveWrap::Periodic ? (basis == geom::CurveBasis::Bezier ? n / 3 : n)
                                                      : (n >= 4 ? (n - 4) / vstep + 1 : 0);
        } else {
            spans = wrap == geom::CurveWrap::Periodic ? n : n - 1;
        }
        for (uint32_t s = 0; s < spans; ++s) {
            SpanRecord span{};
            span.curve = static_cast<uint32_t>(c);
            span.count = cubic ? 4u : 2u;
            for (uint32_t k = 0; k < 4; ++k) {
                uint32_t local = s * vstep + k;
                if (wrap == geom::CurveWrap::Periodic) local %= n;
                span.wrap[k] = at + std::min(local, n - 1);
            }
            out.push_back(span);
        }
        at += n;
    }
    return out;
}

void checkTube(test::Gpu& gpu, const char* what, geom::CurveBasis basis, geom::CurveWrap wrap,
               const std::vector<float>& points, const std::vector<int32_t>& counts, float width, uint32_t sides,
               uint32_t segments) {
    auto builder = geom::CurveBuilder::create(*gpu.library);
    if (!builder) FAIL(builder.error().toString());
    geom::CurveInput in;
    in.source = what;
    in.points = {std::as_bytes(std::span<const float>(points)), false};
    in.curveVertexCounts = counts;
    in.basis = basis;
    in.wrap = wrap;
    in.width = width;
    in.sides = sides;
    in.segments = segments;
    auto built = builder->build(in);
    if (!built) FAIL(built.error().toString());
    const std::vector<SpanRecord> spans = spansOf(counts, basis, wrap);
    REQUIRE(built->spans == spans.size());
    const uint32_t vertices = built->mesh.points;
    REQUIRE(vertices == spans.size() * (segments + 1) * sides);
    std::vector<float> points4;
    for (size_t i = 0; i < points.size() / 3; ++i) {
        points4.insert(points4.end(), {points[i * 3], points[i * 3 + 1], points[i * 3 + 2], 1.0F});
    }
    auto pointsBuffer = gpu::Buffer::fromSpan<float>(*gpu.device, points4, "check.points");
    auto spanBuffer = gpu::Buffer::fromSpan<SpanRecord>(*gpu.device, spans, "check.spans");
    REQUIRE(pointsBuffer);
    REQUIRE(spanBuffer);
    auto made = gpu::ComputeKernel::create(*gpu.library, "lrt/test/curve_check", "curveCheck");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counters = test::uintBuffer(*gpu.device, 3, "check.counts");
    const geom::GpuPrimvar* tangentPrimvar = built->mesh.primvar("tangent");
    REQUIRE(tangentPrimvar != nullptr);
    REQUIRE(tangentPrimvar->count == vertices);
    gpu::Buffer worst = test::uintBuffer(*gpu.device, 1, "check.worst");
    {
        gpu::CommandBatch batch(*gpu.device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["points"].setBinding(pointsBuffer->rhi());
            cursor["spans"].setBinding(spanBuffer->rhi());
            cursor["tube"].setBinding(built->mesh.positions.rhi());
            cursor["tangents"].setBinding(tangentPrimvar->values.rhi());
            cursor["counts"].setBinding(counters.rhi());
            cursor["worst"].setBinding(worst.rhi());
            rhi::ShaderCursor c = cursor["check"];
            c["vertices"].setData(vertices);
            c["sides"].setData(sides);
            c["segments"].setData(segments);
            c["basis"].setData(static_cast<uint32_t>(basis));
            c["width"].setData(width);
            c["tolerance"].setData(1.0e-5F);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t n[3] = {0, 0, 0};
    float e = 0.0F;
    REQUIRE(counters.read(*gpu.device, 0, sizeof(n), n));
    REQUIRE(worst.read(*gpu.device, 0, sizeof(e), &e));
    std::printf("  %s: %u spans, %u tube vertices, %u farther than 1e-5 from half the width off the curve (worst "
                "%.2e), %u tangents off the curve's direction; %u triangles\n",
                what, built->spans, n[0], n[1], static_cast<double>(e), n[2], built->mesh.triangles);
    CHECK(n[0] == vertices);
    CHECK(n[1] == 0);
    CHECK(n[2] == 0);
    CHECK(built->mesh.triangles == spans.size() * segments * sides * 2);
}

}   // namespace

TEST_CASE("a curve's tube rings the curve at half its width, for every basis and wrap", "[geom][curves]") {
    LRT_REQUIRE_GPU(gpu);
    // Two curves of control points on a bent path, and a third short one.
    std::vector<float> points;
    for (int i = 0; i < 7; ++i) {
        points.insert(points.end(), {i * 0.5F, std::sin(i * 0.9F), 0.3F * std::cos(i * 0.7F)});
    }
    for (int i = 0; i < 5; ++i) {
        points.insert(points.end(), {-1.0F + i * 0.4F, 2.0F + 0.2F * i * i, -1.0F});
    }
    for (int i = 0; i < 4; ++i) {
        points.insert(points.end(), {3.0F, i * 0.25F, 1.0F + 0.1F * i});
    }
    const std::vector<int32_t> counts{7, 5, 4};
    checkTube(*gpu, "linear", geom::CurveBasis::Linear, geom::CurveWrap::Nonperiodic, points, counts, 0.1F, 6, 1);
    checkTube(*gpu, "bezier", geom::CurveBasis::Bezier, geom::CurveWrap::Nonperiodic, points, counts, 0.08F, 8, 6);
    checkTube(*gpu, "bspline", geom::CurveBasis::BSpline, geom::CurveWrap::Nonperiodic, points, counts, 0.05F, 5, 4);
    checkTube(*gpu, "catmullRom", geom::CurveBasis::CatmullRom, geom::CurveWrap::Nonperiodic, points, counts, 0.12F,
              8, 8);
    checkTube(*gpu, "linear periodic", geom::CurveBasis::Linear, geom::CurveWrap::Periodic, points, counts, 0.1F, 4,
              1);
    checkTube(*gpu, "bspline periodic", geom::CurveBasis::BSpline, geom::CurveWrap::Periodic, points, counts, 0.06F,
              6, 3);
}
