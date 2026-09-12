// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/geom/Curves.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::geom {
namespace {

struct SpanRecord {
    uint32_t first, curve, count, varyingFirst;
    uint32_t wrap[4];
};
static_assert(sizeof(SpanRecord) == 32);

}   // namespace

Result<CurveBuilder> CurveBuilder::create(gpu::ShaderLibrary& library) {
    auto tube = gpu::ComputeKernel::create(library, "lrt/geom/curve_tube", "curveTube");
    if (!tube) return std::move(tube).error();
    auto points = gpu::ComputeKernel::create(library, "lrt/geom/mesh_topology", "meshPoints");
    if (!points) return std::move(points).error();
    auto meshes = MeshBuilder::create(library);
    if (!meshes) return std::move(meshes).error();
    CurveBuilder b;
    b.device_ = &library.device();
    b.tube_ = std::move(*tube);
    b.points_ = std::move(*points);
    b.meshes_ = std::move(*meshes);
    return b;
}

Result<CurveBuilder::Built> CurveBuilder::build(const CurveInput& in) {
    gpu::Device& device = *device_;
    const uint32_t controlPoints = static_cast<uint32_t>(in.points.values() / 3);
    if (controlPoints == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': curves without points", in.source);
    }
    const bool cubic = in.basis != CurveBasis::Linear;
    const uint32_t segments = in.segments != 0 ? in.segments : (cubic ? 8u : 1u);
    const uint32_t sides = std::max(in.sides, 3u);
    const uint32_t vstep = in.basis == CurveBasis::Bezier ? 3u : 1u;

    // The spans, curve by curve: which control points each takes, wrapped
    // round for a periodic curve. Pure bookkeeping of indices.
    std::vector<SpanRecord> spans;
    uint32_t at = 0;
    uint32_t varyingAt = 0;
    const auto pointAt = [&](uint32_t k) {
        const uint32_t index = k < in.curveIndices.size() && !in.curveIndices.empty()
                                   ? static_cast<uint32_t>(std::max(in.curveIndices[k], 0))
                                   : k;
        return std::min(index, controlPoints - 1);
    };
    for (size_t c = 0; c < in.curveVertexCounts.size(); ++c) {
        const uint32_t n = static_cast<uint32_t>(std::max(in.curveVertexCounts[c], 0));
        uint32_t count = 0;
        if (cubic) {
            if (in.wrap == CurveWrap::Periodic) {
                count = in.basis == CurveBasis::Bezier ? n / 3 : n;
            } else {
                count = n >= 4 ? (n - 4) / vstep + 1 : 0;
            }
        } else {
            count = in.wrap == CurveWrap::Periodic ? n : (n >= 2 ? n - 1 : 0);
        }
        for (uint32_t s = 0; s < count; ++s) {
            SpanRecord span{};
            span.first = at + s * vstep;
            span.curve = static_cast<uint32_t>(c);
            span.count = cubic ? 4u : 2u;
            span.varyingFirst = varyingAt + s;
            for (uint32_t k = 0; k < span.count; ++k) {
                uint32_t local = s * vstep + k;
                if (in.wrap == CurveWrap::Periodic && n > 0) {
                    local %= n;
                }
                span.wrap[k] = pointAt(at + std::min(local, n > 0 ? n - 1 : 0u));
            }
            spans.push_back(span);
        }
        at += n;
        varyingAt += in.wrap == CurveWrap::Periodic ? count : count + 1;
    }
    if (spans.empty()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no curve has a span", in.source);
    }
    const uint32_t rings = segments + 1;
    const uint32_t vertices = static_cast<uint32_t>(spans.size()) * rings * sides;

    // The control points as float4, by the mesh builder's decode.
    gpu::BufferDesc wordsDesc;
    wordsDesc.bytes = std::max<uint64_t>(in.points.bytes.size(), 4);
    wordsDesc.elementBytes = 4;
    wordsDesc.label = "curves.points.words";
    auto words = gpu::Buffer::create(device, wordsDesc);
    if (!words) return std::move(words).error();
    LRT_TRY(words->write(device, 0, in.points.bytes.size(), in.points.bytes.data()));
    gpu::BufferDesc pointsDesc;
    pointsDesc.bytes = uint64_t{controlPoints} * 16;
    pointsDesc.elementBytes = 16;
    pointsDesc.label = "curves.points";
    auto control = gpu::Buffer::create(device, pointsDesc);
    if (!control) return std::move(control).error();
    auto spanBuffer = gpu::Buffer::fromSpan<SpanRecord>(device, spans, "curves.spans");
    if (!spanBuffer) return std::move(spanBuffer).error();
    std::vector<float> widthValues(in.widths.begin(), in.widths.end());
    if (widthValues.empty()) {
        widthValues.push_back(in.width);
    }
    auto widths = gpu::Buffer::fromSpan<float>(device, widthValues, "curves.widths");
    if (!widths) return std::move(widths).error();
    gpu::BufferDesc tubeDesc;
    tubeDesc.bytes = uint64_t{vertices} * 16;
    tubeDesc.elementBytes = 16;
    tubeDesc.label = "curves.tube";
    auto tube = gpu::Buffer::create(device, tubeDesc);
    if (!tube) return std::move(tube).error();
    tubeDesc.label = "curves.tangents";
    auto tangents = gpu::Buffer::create(device, tubeDesc);
    if (!tangents) return std::move(tangents).error();
    {
        gpu::CommandBatch batch(device);
        points_.dispatch(batch, {controlPoints, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(words->rhi());
            cursor["positions"].setBinding(control->rhi());
            cursor["params"]["count"].setData(controlPoints);
            cursor["params"]["half"].setData(uint32_t{in.points.half ? 1u : 0u});
        });
        tube_.dispatch(batch, {vertices, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["points"].setBinding(control->rhi());
            cursor["spans"].setBinding(spanBuffer->rhi());
            cursor["widths"].setBinding(widths->rhi());
            cursor["positions"].setBinding(tube->rhi());
            cursor["tangents"].setBinding(tangents->rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["vertices"].setData(vertices);
            p["sides"].setData(sides);
            p["segments"].setData(segments);
            p["basis"].setData(static_cast<uint32_t>(in.basis));
            p["widthMode"].setData(in.widths.empty() ? 0u : static_cast<uint32_t>(in.widthInterpolation));
            p["width"].setData(in.width);
        });
        LRT_TRY(batch.submit(true));
    }

    // The rings' topology: a quad between neighbouring vertices of
    // neighbouring rings, each span its own strip of rings.
    std::vector<int32_t> counts;
    std::vector<int32_t> indices;
    std::vector<int32_t> faceCurve;   // the face's curve, for uniform primvars
    counts.reserve(static_cast<size_t>(spans.size()) * segments * sides);
    for (uint32_t s = 0; s < spans.size(); ++s) {
        const uint32_t base = s * rings * sides;
        for (uint32_t k = 0; k < segments; ++k) {
            for (uint32_t j = 0; j < sides; ++j) {
                const uint32_t j1 = (j + 1) % sides;
                const int32_t a = static_cast<int32_t>(base + k * sides + j);
                const int32_t b = static_cast<int32_t>(base + k * sides + j1);
                const int32_t c = static_cast<int32_t>(base + (k + 1) * sides + j1);
                const int32_t d = static_cast<int32_t>(base + (k + 1) * sides + j);
                counts.push_back(4);
                indices.insert(indices.end(), {a, b, c, d});
                faceCurve.push_back(static_cast<int32_t>(spans[s].curve));
            }
        }
    }
    MeshInput mesh;
    mesh.source = in.source;
    mesh.devicePositions = &*tube;
    mesh.devicePoints = vertices;
    mesh.faceVertexCounts = counts;
    mesh.faceVertexIndices = indices;
    mesh.smoothNormals = true;
    mesh.topology = in.topology;
    // A curve's uniform primvars become the tube's faces': one value a face,
    // each face taking its curve's. A constant primvar stays constant.
    std::vector<PrimvarInput> primvars;
    std::vector<std::vector<float>> expanded;
    for (const PrimvarInput& p : in.primvars) {
        PrimvarInput out = p;
        if (p.interpolation == Interpolation::Uniform && !p.values.isDouble && !p.values.half) {
            const size_t have = p.values.bytes.size() / sizeof(float);
            std::vector<float>& values = expanded.emplace_back();
            values.reserve(faceCurve.size() * p.components);
            for (const int32_t curve : faceCurve) {
                for (uint32_t c = 0; c < p.components; ++c) {
                    const size_t k = static_cast<size_t>(curve) * p.components + c;
                    float v = 0.0F;
                    if (k < have) {
                        std::memcpy(&v, p.values.bytes.data() + k * sizeof(float), sizeof(float));
                    }
                    values.push_back(v);
                }
            }
            out.values = {std::as_bytes(std::span<const float>(values)), false};
            out.indices = {};
        } else if (p.interpolation != Interpolation::Constant) {
            continue;   // vertex and varying primvars are not carried to the tube yet
        }
        primvars.push_back(out);
    }
    // The fibre's direction at every vertex, for a hair material's frame.
    PrimvarInput tangent;
    tangent.name = "tangent";
    tangent.interpolation = Interpolation::Vertex;
    tangent.components = 3;
    tangent.deviceValues = &*tangents;
    tangent.deviceCount = vertices;
    primvars.push_back(tangent);
    mesh.primvars = primvars;
    auto built = meshes_.build(mesh);
    if (!built) return std::move(built).error();
    Built out;
    out.mesh = std::move(*built);
    out.curves = static_cast<uint32_t>(in.curveVertexCounts.size());
    out.spans = static_cast<uint32_t>(spans.size());
    return out;
}

}   // namespace lrt::geom
