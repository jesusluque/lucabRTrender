// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/geom/Subdivision.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::geom {
namespace {

constexpr uint32_t kNone = 0xFFFFFFFFu;

/// One level's topology, as the kernels read it: faces as corner lists over
/// `vertices` vertices, the edges those corners make, what each vertex
/// touches, and the sharpness of edges and vertices.
struct Level {
    uint32_t              vertices = 0;
    std::vector<uint32_t> faceStarts, faceCounts, corners;
    std::vector<uint32_t> cornerEdge;       ///< per corner: the edge from it to the next corner
    std::vector<uint32_t> edgeEnds;         ///< 4 an edge: v0, v1, f0, f1
    std::vector<uint32_t> edgeOpposite;     ///< 2 an edge: the third vertex of f0, f1 (triangles)
    std::vector<float>    edgeSharp;
    std::vector<uint32_t> vertexOffsets, vertexEdges, vertexFaces;
    std::vector<float>    vertexSharp;
    std::vector<uint32_t> faceParent;       ///< per face: the coarse face it came from
    std::vector<uint8_t>  faceHole;
    bool                  triangles = true; ///< every face a triangle (Loop applies)
    [[nodiscard]] uint32_t edges() const noexcept { return static_cast<uint32_t>(edgeSharp.size()); }
    [[nodiscard]] uint32_t faces() const noexcept { return static_cast<uint32_t>(faceCounts.size()); }
};

uint64_t edgeKey(uint32_t a, uint32_t b) {
    return (uint64_t{std::min(a, b)} << 32) | std::max(a, b);
}

/// The edges and the vertex adjacency of a level whose faces and corners
/// are set; `sharpOf` gives an edge's sharpness by its ends.
void connect(Level& level, const std::unordered_map<uint64_t, float>& sharpOf) {
    std::unordered_map<uint64_t, uint32_t> edgeOf;
    level.cornerEdge.assign(level.corners.size(), kNone);
    level.edgeEnds.clear();
    level.edgeOpposite.clear();
    level.edgeSharp.clear();
    level.triangles = true;
    for (uint32_t f = 0; f < level.faces(); ++f) {
        const uint32_t start = level.faceStarts[f];
        const uint32_t n = level.faceCounts[f];
        level.triangles = level.triangles && n == 3;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t a = level.corners[start + j];
            const uint32_t b = level.corners[start + (j + 1) % n];
            const uint32_t opposite = n == 3 ? level.corners[start + (j + 2) % 3] : kNone;
            const uint64_t key = edgeKey(a, b);
            auto found = edgeOf.find(key);
            uint32_t e = 0;
            if (found == edgeOf.end()) {
                e = level.edges();
                edgeOf.emplace(key, e);
                level.edgeEnds.insert(level.edgeEnds.end(), {std::min(a, b), std::max(a, b), f, kNone});
                level.edgeOpposite.insert(level.edgeOpposite.end(), {opposite, kNone});
                const auto sharp = sharpOf.find(key);
                level.edgeSharp.push_back(sharp != sharpOf.end() ? sharp->second : 0.0F);
            } else {
                e = found->second;
                if (level.edgeEnds[e * 4 + 3] == kNone) {
                    level.edgeEnds[e * 4 + 3] = f;
                    level.edgeOpposite[e * 2 + 1] = opposite;
                }
            }
            level.cornerEdge[start + j] = e;
        }
    }
    // Per vertex: the edges at it (each once) and the faces at it.
    std::vector<std::vector<uint32_t>> edgesAt(level.vertices);
    std::vector<std::vector<uint32_t>> facesAt(level.vertices);
    for (uint32_t e = 0; e < level.edges(); ++e) {
        edgesAt[level.edgeEnds[e * 4]].push_back(e);
        edgesAt[level.edgeEnds[e * 4 + 1]].push_back(e);
    }
    for (uint32_t f = 0; f < level.faces(); ++f) {
        const uint32_t start = level.faceStarts[f];
        for (uint32_t j = 0; j < level.faceCounts[f]; ++j) {
            facesAt[level.corners[start + j]].push_back(f);
        }
    }
    level.vertexOffsets.assign(level.vertices + 1, 0);
    level.vertexEdges.clear();
    level.vertexFaces.clear();
    for (uint32_t v = 0; v < level.vertices; ++v) {
        auto& es = edgesAt[v];
        std::sort(es.begin(), es.end());
        es.erase(std::unique(es.begin(), es.end()), es.end());
        auto& fs = facesAt[v];
        std::sort(fs.begin(), fs.end());
        fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
        const size_t n = std::max(es.size(), fs.size());
        for (size_t k = 0; k < n; ++k) {
            level.vertexEdges.push_back(k < es.size() ? es[k] : kNone);
            level.vertexFaces.push_back(k < fs.size() ? fs[k] : kNone);
        }
        level.vertexOffsets[v + 1] = static_cast<uint32_t>(level.vertexEdges.size());
    }
    if (level.vertexSharp.size() != level.vertices) {
        level.vertexSharp.assign(level.vertices, 0.0F);
    }
}

/// The next level's faces over the vertex numbering [vertices][edges][faces],
/// with the sharpness the children inherit.
Level split(const Level& level, SubdivisionScheme scheme) {
    Level next;
    const bool loop = scheme == SubdivisionScheme::Loop && level.triangles;
    const uint32_t v0 = 0;
    const uint32_t e0 = level.vertices;
    const uint32_t f0 = level.vertices + level.edges();
    next.vertices = loop ? level.vertices + level.edges() : level.vertices + level.edges() + level.faces();
    std::unordered_map<uint64_t, float> sharpOf;
    const auto child = [&](std::initializer_list<uint32_t> ids, uint32_t parent, bool hole) {
        next.faceStarts.push_back(static_cast<uint32_t>(next.corners.size()));
        next.faceCounts.push_back(static_cast<uint32_t>(ids.size()));
        next.corners.insert(next.corners.end(), ids.begin(), ids.end());
        next.faceParent.push_back(parent);
        next.faceHole.push_back(hole ? 1 : 0);
    };
    for (uint32_t f = 0; f < level.faces(); ++f) {
        const uint32_t start = level.faceStarts[f];
        const uint32_t n = level.faceCounts[f];
        const uint32_t parent = level.faceParent.empty() ? f : level.faceParent[f];
        const bool hole = !level.faceHole.empty() && level.faceHole[f] != 0;
        if (loop) {
            const uint32_t a = level.corners[start], b = level.corners[start + 1], c = level.corners[start + 2];
            const uint32_t ab = e0 + level.cornerEdge[start], bc = e0 + level.cornerEdge[start + 1],
                           ca = e0 + level.cornerEdge[start + 2];
            child({v0 + a, ab, ca}, parent, hole);
            child({v0 + b, bc, ab}, parent, hole);
            child({v0 + c, ca, bc}, parent, hole);
            child({ab, bc, ca}, parent, hole);
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t v = level.corners[start + j];
                const uint32_t e = e0 + level.cornerEdge[start + j];
                const uint32_t previous = e0 + level.cornerEdge[start + (j + n - 1) % n];
                child({v0 + v, e, f0 + f, previous}, parent, hole);
            }
        }
    }
    // A split edge's halves keep its sharpness less one; everything else is smooth.
    for (uint32_t e = 0; e < level.edges(); ++e) {
        const float s = std::max(level.edgeSharp[e] - 1.0F, 0.0F);
        if (s > 0.0F) {
            sharpOf[edgeKey(v0 + level.edgeEnds[e * 4], e0 + e)] = s;
            sharpOf[edgeKey(v0 + level.edgeEnds[e * 4 + 1], e0 + e)] = s;
        }
    }
    next.vertexSharp.assign(next.vertices, 0.0F);
    for (uint32_t v = 0; v < level.vertices; ++v) {
        next.vertexSharp[v] = std::max(level.vertexSharp[v] - 1.0F, 0.0F);
    }
    connect(next, sharpOf);
    return next;
}

}   // namespace

Result<Subdivider> Subdivider::create(gpu::ShaderLibrary& library) {
    Subdivider s;
    s.device_ = &library.device();
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    LRT_TRY(make(s.facePoints_, "lrt/geom/subdivision", "subdivFacePoints"));
    LRT_TRY(make(s.edgePoints_, "lrt/geom/subdivision", "subdivEdgePoints"));
    LRT_TRY(make(s.vertexPoints_, "lrt/geom/subdivision", "subdivVertexPoints"));
    LRT_TRY(make(s.limit_, "lrt/geom/subdivision", "subdivLimit"));
    LRT_TRY(make(s.gather_, "lrt/geom/subdivision", "subdivGather"));
    LRT_TRY(make(s.points_, "lrt/geom/mesh_topology", "meshPoints"));
    LRT_TRY(make(s.expand_, "lrt/geom/primvar_expand", "primvarExpand"));
    return s;
}

Result<Refined> Subdivider::refine(const SubdivisionInput& in) {
    gpu::Device& device = *device_;
    const auto buffer = [&](uint64_t count, uint32_t element, const char* label) -> Result<gpu::Buffer> {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(count, 1) * element;
        desc.elementBytes = element;
        desc.label = label;
        return gpu::Buffer::create(device, desc);
    };
    const auto upload = [&](const auto& values, uint32_t element, const char* label) -> Result<gpu::Buffer> {
        using T = typename std::remove_reference_t<decltype(values)>::value_type;
        std::vector<T> padded(values.begin(), values.end());
        if (padded.empty()) {
            padded.push_back(T{});
        }
        return gpu::Buffer::fromSpan<T>(device, padded, label);
    };

    // The coarse points as float4 on the device.
    const uint32_t coarsePoints =
        in.devicePositions != nullptr ? in.devicePoints : static_cast<uint32_t>(in.points.values() / 3);
    if (coarsePoints == 0 || in.faceVertexCounts.empty()) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': nothing to subdivide", in.source);
    }
    auto positions = buffer(coarsePoints, 16, "subdiv.points");
    if (!positions) return std::move(positions).error();
    if (in.devicePositions != nullptr) {
        gpu::CommandBatch batch(device);
        batch.encoder()->copyBuffer(positions->rhi(), 0, in.devicePositions->rhi(), 0, uint64_t{coarsePoints} * 16);
        batch.markDirty();
        LRT_TRY(batch.submit(true));
    } else {
        auto words = buffer((in.points.bytes.size() + 3) / 4, 4, "subdiv.points.words");
        if (!words) return std::move(words).error();
        LRT_TRY(words->write(device, 0, in.points.bytes.size(), in.points.bytes.data()));
        gpu::CommandBatch batch(device);
        points_.dispatch(batch, {coarsePoints, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(words->rhi());
            cursor["positions"].setBinding(positions->rhi());
            cursor["params"]["count"].setData(coarsePoints);
            cursor["params"]["half"].setData(uint32_t{in.points.half ? 1u : 0u});
        });
        LRT_TRY(batch.submit(true));
    }

    // The coarse level: faces as corner lists, holes, creases and corners.
    Level level;
    level.vertices = coarsePoints;
    uint32_t at = 0;
    for (size_t f = 0; f < in.faceVertexCounts.size(); ++f) {
        const uint32_t n = static_cast<uint32_t>(std::max(in.faceVertexCounts[f], 0));
        level.faceStarts.push_back(static_cast<uint32_t>(level.corners.size()));
        level.faceCounts.push_back(n);
        for (uint32_t j = 0; j < n; ++j) {
            const int32_t index = at + j < in.faceVertexIndices.size() ? in.faceVertexIndices[at + j] : 0;
            level.corners.push_back(static_cast<uint32_t>(std::clamp<int32_t>(index, 0, int32_t(coarsePoints) - 1)));
        }
        level.faceParent.push_back(static_cast<uint32_t>(f));
        level.faceHole.push_back(0);
        at += n;
    }
    for (const int32_t hole : in.holeIndices) {
        if (hole >= 0 && static_cast<size_t>(hole) < level.faceHole.size()) {
            level.faceHole[static_cast<size_t>(hole)] = 1;
        }
    }
    std::unordered_map<uint64_t, float> sharpOf;
    {
        size_t k = 0;
        size_t edge = 0;
        const bool perEdge = in.creaseSharpnesses.size() != in.creaseLengths.size();
        for (size_t c = 0; c < in.creaseLengths.size(); ++c) {
            const size_t n = static_cast<size_t>(std::max(in.creaseLengths[c], 0));
            for (size_t j = 0; j + 1 < n && k + j + 1 < in.creaseIndices.size(); ++j) {
                const size_t which = perEdge ? edge : c;
                const float s = which < in.creaseSharpnesses.size() ? in.creaseSharpnesses[which] : 0.0F;
                sharpOf[edgeKey(static_cast<uint32_t>(in.creaseIndices[k + j]),
                                static_cast<uint32_t>(in.creaseIndices[k + j + 1]))] = s;
                ++edge;
            }
            k += n;
        }
    }
    level.vertexSharp.assign(coarsePoints, 0.0F);
    for (size_t k = 0; k < in.cornerIndices.size() && k < in.cornerSharpnesses.size(); ++k) {
        const int32_t v = in.cornerIndices[k];
        if (v >= 0 && static_cast<uint32_t>(v) < coarsePoints) {
            level.vertexSharp[static_cast<size_t>(v)] = in.cornerSharpnesses[k];
        }
    }
    connect(level, sharpOf);
    const SubdivisionScheme scheme = in.scheme == SubdivisionScheme::Loop && !level.triangles
                                         ? SubdivisionScheme::CatmullClark
                                         : in.scheme;

    // A channel refined over a level: values a vertex in, values a vertex of
    // the next level out, or the limit positions when `limit`.
    const auto refineValues = [&](const Level& lv, const gpu::Buffer& values, SubdivisionScheme s, bool limit,
                                  gpu::Buffer& out) -> Result<void> {
        auto starts = upload(lv.faceStarts, 4, "subdiv.faceStarts");
        auto counts = upload(lv.faceCounts, 4, "subdiv.faceCounts");
        auto corners = upload(lv.corners, 4, "subdiv.corners");
        auto ends = upload(lv.edgeEnds, 4, "subdiv.edgeEnds");
        auto opposite = upload(lv.edgeOpposite, 4, "subdiv.edgeOpposite");
        auto sharp = upload(lv.edgeSharp, 4, "subdiv.edgeSharp");
        auto offsets = upload(lv.vertexOffsets, 4, "subdiv.vertexOffsets");
        auto vertexEdges = upload(lv.vertexEdges, 4, "subdiv.vertexEdges");
        auto vertexFaces = upload(lv.vertexFaces, 4, "subdiv.vertexFaces");
        auto vertexSharp = upload(lv.vertexSharp, 4, "subdiv.vertexSharp");
        for (const auto* r : {&starts, &counts, &corners, &ends, &opposite, &sharp, &offsets, &vertexEdges,
                              &vertexFaces, &vertexSharp}) {
            if (!*r) return r->error();
        }
        const uint32_t outCount = limit ? lv.vertices : lv.vertices + lv.edges() + lv.faces();
        auto made = buffer(outCount, 16, "subdiv.refined");
        if (!made) return std::move(made).error();
        out = std::move(*made);
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["values"].setBinding(values.rhi());
            cursor["faceStarts"].setBinding(starts->rhi());
            cursor["faceCounts"].setBinding(counts->rhi());
            cursor["corners"].setBinding(corners->rhi());
            cursor["edgeEnds"].setBinding(ends->rhi());
            cursor["edgeOpposite"].setBinding(opposite->rhi());
            cursor["edgeSharp"].setBinding(sharp->rhi());
            cursor["vertexOffsets"].setBinding(offsets->rhi());
            cursor["vertexEdges"].setBinding(vertexEdges->rhi());
            cursor["vertexFaces"].setBinding(vertexFaces->rhi());
            cursor["vertexSharp"].setBinding(vertexSharp->rhi());
            cursor["refined"].setBinding(out.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["scheme"].setData(static_cast<uint32_t>(s));
            p["vertices"].setData(lv.vertices);
            p["edges"].setData(lv.edges());
            p["faces"].setData(lv.faces());
            p["edgeBase"].setData(lv.vertices);
            p["faceBase"].setData(lv.vertices + lv.edges());
        };
        gpu::CommandBatch batch(device);
        if (limit) {
            limit_.dispatch(batch, {lv.vertices, 1, 1}, bind);
        } else {
            facePoints_.dispatch(batch, {lv.faces(), 1, 1}, bind);
            edgePoints_.dispatch(batch, {lv.edges(), 1, 1}, bind);
            vertexPoints_.dispatch(batch, {lv.vertices, 1, 1}, bind);
        }
        return batch.submit(true);
    };

    // Face-varying channels each carry a topology of their own: their
    // vertices are the channel's unique values (its indices, or its corners
    // when it has none), their edges the pairs of consecutive corners, and a
    // seam -- an edge one face makes -- is a boundary.
    struct Channel {
        const PrimvarInput* source;
        Level               level;      ///< face-varying: its own; else unused
        gpu::Buffer         values;     ///< a float4 per channel vertex
        uint32_t            count = 0;
    };
    std::vector<Channel> channels;
    for (const PrimvarInput& p : in.primvars) {
        if (p.name == "normals" || p.interpolation == Interpolation::Constant ||
            p.interpolation == Interpolation::Uniform) {
            continue;
        }
        Channel channel{&p};
        const uint32_t components = std::clamp<uint32_t>(p.components, 1, 4);
        const uint32_t elements = static_cast<uint32_t>(p.values.values() / components);
        if (p.interpolation == Interpolation::FaceVarying) {
            Level& lv = channel.level;
            lv.faceStarts = level.faceStarts;
            lv.faceCounts = level.faceCounts;
            lv.faceParent = level.faceParent;
            lv.faceHole = level.faceHole;
            const bool indexed = !p.indices.empty();
            lv.vertices = indexed ? elements : static_cast<uint32_t>(level.corners.size());
            for (size_t c = 0; c < level.corners.size(); ++c) {
                const int32_t index = indexed && c < p.indices.size() ? p.indices[c] : static_cast<int32_t>(c);
                lv.corners.push_back(static_cast<uint32_t>(std::clamp<int32_t>(index, 0, int32_t(lv.vertices) - 1)));
            }
            lv.vertexSharp.assign(lv.vertices, 0.0F);
            connect(lv, {});
            channel.count = lv.vertices;
        } else {
            channel.count = std::min(elements, coarsePoints);
            if (!p.indices.empty()) {
                continue;   // an indexed vertex primvar: rare, and not refined here
            }
        }
        // The channel's values as float4, by the builder's expand.
        auto words = buffer((p.values.bytes.size() + 3) / 4, 4, "subdiv.channel.words");
        if (!words) return std::move(words).error();
        LRT_TRY(words->write(device, 0, p.values.bytes.size(), p.values.bytes.data()));
        auto none = upload(std::vector<int32_t>{}, 4, "subdiv.channel.noIndices");
        if (!none) return std::move(none).error();
        auto expanded = buffer(channel.count, 16, "subdiv.channel");
        if (!expanded) return std::move(expanded).error();
        channel.values = std::move(*expanded);
        gpu::CommandBatch batch(device);
        expand_.dispatch(batch, {channel.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(words->rhi());
            cursor["indices"].setBinding(none->rhi());
            cursor["expanded"].setBinding(channel.values.rhi());
            rhi::ShaderCursor q = cursor["params"];
            q["count"].setData(channel.count);
            q["components"].setData(components);
            q["kind"].setData(p.values.kind());
            q["indexed"].setData(uint32_t{0});
            q["values"].setData(elements);
        });
        LRT_TRY(batch.submit(true));
        channels.push_back(std::move(channel));
    }

    // The levels.
    for (uint32_t l = 0; l < std::max(in.levels, 1u); ++l) {
        gpu::Buffer next;
        LRT_TRY(refineValues(level, *positions, scheme, false, next));
        for (Channel& channel : channels) {
            gpu::Buffer refined;
            if (channel.source->interpolation == Interpolation::FaceVarying) {
                LRT_TRY(refineValues(channel.level, channel.values, scheme, false, refined));
                channel.level = split(channel.level, scheme);
                channel.count = channel.level.vertices;
            } else {
                const SubdivisionScheme rule =
                    channel.source->interpolation == Interpolation::Varying ? SubdivisionScheme::Bilinear : scheme;
                LRT_TRY(refineValues(level, channel.values, rule, false, refined));
                channel.count = level.vertices + level.edges() + (scheme == SubdivisionScheme::Loop && level.triangles ? 0 : level.faces());
            }
            channel.values = std::move(refined);
        }
        level = split(level, scheme);
        *positions = std::move(next);
    }
    if (in.limit && scheme != SubdivisionScheme::Bilinear) {
        gpu::Buffer limited;
        LRT_TRY(refineValues(level, *positions, scheme, true, limited));
        *positions = std::move(limited);
        for (Channel& channel : channels) {
            if (channel.source->interpolation == Interpolation::Vertex) {
                gpu::Buffer refined;
                LRT_TRY(refineValues(level, channel.values, scheme, true, refined));
                channel.values = std::move(refined);
            }
        }
    }

    Refined out;
    out.positions = std::move(*positions);
    out.points = level.vertices;
    out.edges = level.edges();
    for (uint32_t f = 0; f < level.faces(); ++f) {
        out.faceVertexCounts.push_back(static_cast<int32_t>(level.faceCounts[f]));
        if (level.faceHole[f] != 0) {
            out.holeIndices.push_back(static_cast<int32_t>(f));
        }
    }
    out.faceVertexIndices.reserve(level.corners.size());
    for (const uint32_t c : level.corners) {
        out.faceVertexIndices.push_back(static_cast<int32_t>(c));
    }
    for (const PrimvarInput& p : in.primvars) {
        if (p.name == "normals") {
            continue;   // the builder makes the refined surface's own
        }
        if (p.interpolation == Interpolation::Constant || p.interpolation == Interpolation::Uniform) {
            RefinedPrimvar r;
            r.name = p.name;
            r.interpolation = p.interpolation;
            r.components = p.components;
            r.hostValues = p.values;
            if (p.interpolation == Interpolation::Uniform) {
                for (const uint32_t parent : level.faceParent) {
                    const int32_t coarse = static_cast<int32_t>(parent);
                    r.indices.push_back(coarse < static_cast<int32_t>(p.indices.size()) && !p.indices.empty()
                                            ? p.indices[static_cast<size_t>(coarse)]
                                            : coarse);
                }
            }
            out.primvars.push_back(std::move(r));
        }
    }
    for (Channel& channel : channels) {
        RefinedPrimvar r;
        r.name = channel.source->name;
        r.interpolation = channel.source->interpolation;
        r.components = channel.source->components;
        if (channel.source->interpolation == Interpolation::FaceVarying) {
            // Per corner of the refined mesh, gathered from the channel's vertices.
            const uint32_t cornerCount = static_cast<uint32_t>(channel.level.corners.size());
            auto corners = upload(channel.level.corners, 4, "subdiv.channel.corners");
            if (!corners) return std::move(corners).error();
            auto gathered = buffer(cornerCount, 16, "subdiv.channel.gathered");
            if (!gathered) return std::move(gathered).error();
            gpu::CommandBatch batch(device);
            gather_.dispatch(batch, {cornerCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["values"].setBinding(channel.values.rhi());
                cursor["corners"].setBinding(corners->rhi());
                cursor["gathered"].setBinding(gathered->rhi());
                cursor["gather"]["corners"].setData(cornerCount);
            });
            LRT_TRY(batch.submit(true));
            r.values = std::move(*gathered);
            r.count = cornerCount;
        } else {
            r.values = std::move(channel.values);
            r.count = level.vertices;
        }
        out.primvars.push_back(std::move(r));
    }
    return out;
}

Refined::AsInput Refined::asMeshInput(const std::string& source, uint64_t topology) const {
    AsInput out;
    out.mesh.source = source;
    out.mesh.devicePositions = &positions;
    out.mesh.devicePoints = points;
    out.mesh.faceVertexCounts = faceVertexCounts;
    out.mesh.faceVertexIndices = faceVertexIndices;
    out.mesh.holeIndices = holeIndices;
    out.mesh.smoothNormals = true;
    out.mesh.topology = topology;
    for (const RefinedPrimvar& r : primvars) {
        PrimvarInput p;
        p.name = r.name;
        p.interpolation = r.interpolation;
        p.components = r.components;
        if (r.values.valid()) {
            p.deviceValues = &r.values;
            p.deviceCount = r.count;
        } else {
            p.values = r.hostValues;
            p.indices = r.indices;
        }
        out.primvars.push_back(p);
    }
    out.mesh.primvars = out.primvars;
    return out;
}

}   // namespace lrt::geom
