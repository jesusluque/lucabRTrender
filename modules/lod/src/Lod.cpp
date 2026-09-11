// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/lod/Lod.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "lrt/core/Log.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::lod {
namespace {

constexpr uint32_t kLevels = 10;
constexpr uint32_t kMomentsHead = 13;

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

Result<scene::GpuSplats> packed(gpu::Device& device, uint32_t count, uint32_t restPerColour, uint32_t shWords,
                                const char* label) {
    scene::GpuSplats out;
    out.source = label;
    out.count = count;
    out.declared = count;
    out.restPerColour = restPerColour;
    out.shWords = shWords;
    auto p = buffer(device, count, 16, label);
    if (!p) return std::move(p).error();
    auto s = buffer(device, uint64_t{count} * 4, 4, label);
    if (!s) return std::move(s).error();
    auto h = buffer(device, uint64_t{count} * shWords, 4, label);
    if (!h) return std::move(h).error();
    out.positions = std::move(*p);
    out.shape = std::move(*s);
    out.sh = std::move(*h);
    return out;
}

void setBounds(rhi::ShaderCursor p, const LodCloud& cloud) {
    p["boundsLoX"].setData(cloud.boundsLo[0]);
    p["boundsLoY"].setData(cloud.boundsLo[1]);
    p["boundsLoZ"].setData(cloud.boundsLo[2]);
    p["extent"].setData(cloud.extent);
}

}   // namespace

Result<LodBuilder> LodBuilder::create(gpu::ShaderLibrary& library) {
    LodBuilder b;
    b.device_ = &library.device();
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    b.sort_ = std::move(*sort);
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    b.prefix_ = std::move(*prefix);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    LRT_TRY(make(b.morton_, "lrt/lod/lod_morton", "lodMorton"));
    LRT_TRY(make(b.reorder_, "lrt/lod/lod_reorder", "lodReorder"));
    LRT_TRY(make(b.boundaries_, "lrt/lod/lod_boundaries", "lodBoundaries"));
    LRT_TRY(make(b.groups_, "lrt/lod/lod_groups", "lodGroups"));
    LRT_TRY(make(b.leafMoments_, "lrt/lod/lod_leaf_moments", "lodLeafMoments"));
    LRT_TRY(make(b.mergeMoments_, "lrt/lod/lod_merge_moments", "lodMergeMoments"));
    LRT_TRY(make(b.finalize_, "lrt/lod/lod_finalize", "lodFinalize"));
    return b;
}

Result<LodCloud> LodBuilder::build(const scene::GpuSplats& cloud, const LodBuildSettings& settings) {
    gpu::Device& device = *device_;
    const uint32_t n = cloud.count;
    if (n == 0) {
        return Error(ErrorCode::InvalidArgument, "an empty cloud has no levels of detail");
    }
    LodCloud lod;
    for (int k = 0; k < 3; ++k) {
        lod.boundsLo[k] = cloud.bounds.min[static_cast<size_t>(k)];
        lod.extent = std::max(lod.extent, cloud.bounds.max[static_cast<size_t>(k)] - cloud.bounds.min[static_cast<size_t>(k)]);
    }
    lod.extent = std::max(lod.extent, 1e-6F);
    // Slightly past the bounds, so the far corner quantises inside the last cell.
    lod.extent *= 1.0001F;

    // Morton order.
    gpu::SortBuffers sorting;
    const auto assign = [&](gpu::Buffer& into, const char* label) -> Result<void> {
        auto made = buffer(device, n, 4, label);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    LRT_TRY(assign(sorting.keysLo, "lod.keys"));
    LRT_TRY(assign(sorting.values, "lod.order"));
    LRT_TRY(assign(sorting.scratchKeysLo, "lod.keys2"));
    LRT_TRY(assign(sorting.scratchValues, "lod.order2"));
    auto splats = packed(device, n, cloud.restPerColour, cloud.shWords, "lod.splats");
    if (!splats) return std::move(splats).error();
    {
        gpu::CommandBatch batch(device);
        morton_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud.positions.rhi());
            cursor["keys"].setBinding(sorting.keysLo.rhi());
            cursor["values"].setBinding(sorting.values.rhi());
            cursor["params"]["count"].setData(n);
            setBounds(cursor["params"], lod);
        });
        LRT_TRY(sort_.sort(batch, sorting, n, 30));
        reorder_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["order"].setBinding(sorting.values.rhi());
            cursor["srcPositions"].setBinding(cloud.positions.rhi());
            cursor["srcShape"].setBinding(cloud.shape.rhi());
            cursor["srcSh"].setBinding(cloud.sh.rhi());
            cursor["positions"].setBinding(splats->positions.rhi());
            cursor["shape"].setBinding(splats->shape.rhi());
            cursor["sh"].setBinding(splats->sh.rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["shWords"].setData(cloud.shWords);
        });
        LRT_TRY(batch.submit(true));
    }
    splats->source = cloud.source;
    splats->bounds = cloud.bounds;
    lod.splats = std::move(*splats);
    lod.keys = sorting.keysLo;

    // Every level's groups, and how many there are: one small read each.
    struct Level {
        uint32_t    groups = 0;
        gpu::Buffer group, starts, cells;
    };
    std::array<Level, kLevels + 1> levels;
    auto boundary = buffer(device, n, 4, "lod.boundary");
    if (!boundary) return std::move(boundary).error();
    auto before = buffer(device, n, 4, "lod.before");
    if (!before) return std::move(before).error();
    auto total = buffer(device, 1, 4, "lod.total");
    if (!total) return std::move(total).error();
    const auto groupsOf = [&](uint32_t r) -> Result<void> {
        Level& level = levels[r];
        gpu::CommandBatch batch(device);
        boundaries_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["keys"].setBinding(lod.keys.rhi());
            cursor["boundary"].setBinding(boundary->rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["level"].setData(r);
        });
        LRT_TRY(prefix_.apply(batch, *boundary, *before, *total, n));
        LRT_TRY(batch.submit(true));
        LRT_TRY(total->read(device, 0, sizeof(level.groups), &level.groups));
        auto group = buffer(device, n, 4, "lod.group");
        if (!group) return std::move(group).error();
        auto starts = buffer(device, level.groups, 4, "lod.starts");
        if (!starts) return std::move(starts).error();
        auto cells = buffer(device, level.groups, 4, "lod.cells");
        if (!cells) return std::move(cells).error();
        level.group = std::move(*group);
        level.starts = std::move(*starts);
        level.cells = std::move(*cells);
        gpu::CommandBatch write(device);
        groups_.dispatch(write, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["keys"].setBinding(lod.keys.rhi());
            cursor["boundary"].setBinding(boundary->rhi());
            cursor["before"].setBinding(before->rhi());
            cursor["group"].setBinding(level.group.rhi());
            cursor["starts"].setBinding(level.starts.rhi());
            cursor["cells"].setBinding(level.cells.rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["level"].setData(r);
        });
        return write.submit(true);
    };

    // The finest level worth storing: cells holding at least 1 / fraction
    // splats on average. Found coarse to fine; group counts only grow.
    const uint32_t coarsest = std::clamp<uint32_t>(settings.coarsestLevel, 1, kLevels);
    uint32_t finest = coarsest;
    for (uint32_t r = coarsest; r <= kLevels; ++r) {
        LRT_TRY(groupsOf(r));
        if (static_cast<float>(levels[r].groups) > settings.maxGroupFraction * static_cast<float>(n) && r > coarsest) {
            levels[r] = Level{};
            break;
        }
        finest = r;
    }

    // Moments from the finest level up; a Gaussian per group at each.
    const uint32_t keep = cloud.restPerColour;
    const uint32_t stride = kMomentsHead + keep * 3;
    gpu::Buffer fineMoments;
    std::vector<LodLevel> stored;
    for (uint32_t r = finest + 1; r-- > coarsest;) {
        Level& level = levels[r];
        auto moments = buffer(device, uint64_t{level.groups} * stride, 4, "lod.moments");
        if (!moments) return std::move(moments).error();
        auto gaussians = packed(device, level.groups, keep, cloud.shWords, "lod.merged");
        if (!gaussians) return std::move(gaussians).error();
        gpu::CommandBatch batch(device);
        if (r == finest) {
            leafMoments_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(lod.splats.positions.rhi());
                cursor["shape"].setBinding(lod.splats.shape.rhi());
                cursor["sh"].setBinding(lod.splats.sh.rhi());
                cursor["starts"].setBinding(level.starts.rhi());
                cursor["moments"].setBinding(moments->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(n);
                p["groups"].setData(level.groups);
                p["keep"].setData(keep);
                p["shWords"].setData(cloud.shWords);
                p["stride"].setData(stride);
            });
        } else {
            const Level& fine = levels[r + 1];
            mergeMoments_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["starts"].setBinding(level.starts.rhi());
                cursor["fineGroup"].setBinding(fine.group.rhi());
                cursor["fineMoments"].setBinding(fineMoments.rhi());
                cursor["moments"].setBinding(moments->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(n);
                p["groups"].setData(level.groups);
                p["fineGroups"].setData(fine.groups);
                p["stride"].setData(stride);
            });
        }
        finalize_.dispatch(batch, {level.groups, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["moments"].setBinding(moments->rhi());
            cursor["positions"].setBinding(gaussians->positions.rhi());
            cursor["shape"].setBinding(gaussians->shape.rhi());
            cursor["sh"].setBinding(gaussians->sh.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["groups"].setData(level.groups);
            p["keep"].setData(keep);
            p["shWords"].setData(cloud.shWords);
            p["stride"].setData(stride);
        });
        LRT_TRY(batch.submit(true));
        gaussians->bounds = cloud.bounds;
        LodLevel out;
        out.level = r;
        out.gaussians = std::move(*gaussians);
        out.cells = level.cells;
        stored.push_back(std::move(out));
        fineMoments = std::move(*moments);
        if (r + 1 <= kLevels) {
            levels[r + 1] = Level{};   // its groups were only needed to build this one
        }
    }
    std::reverse(stored.begin(), stored.end());
    lod.levels = std::move(stored);
    log::info("{}: levels of detail {}..{}, {} merged Gaussians over {} splats", cloud.source, coarsest, finest,
              lod.mergedGaussians(), n);
    return lod;
}

struct CutSelector::Frame {
    scene::GpuSplats cloud;
    uint32_t         capacity = 0;
    std::vector<gpu::Buffer> selected, dest, totals;
};

CutSelector::CutSelector() = default;
CutSelector::CutSelector(CutSelector&&) noexcept = default;
CutSelector& CutSelector::operator=(CutSelector&&) noexcept = default;
CutSelector::~CutSelector() = default;

Result<CutSelector> CutSelector::create(gpu::ShaderLibrary& library) {
    CutSelector c;
    c.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    c.prefix_ = std::move(*prefix);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    LRT_TRY(make(c.cutGroups_, "lrt/lod/lod_cut", "lodCutGroups"));
    LRT_TRY(make(c.cutSplats_, "lrt/lod/lod_cut", "lodCutSplats"));
    LRT_TRY(make(c.gather_, "lrt/lod/lod_gather", "lodGather"));
    return c;
}

Result<std::vector<render::SplatInstance>> CutSelector::select(const render::Projection& projection,
                                                               std::span<const LodInstance> instances,
                                                               float threshold, std::vector<CutStats>* stats) {
    gpu::Device& device = *device_;
    std::vector<render::SplatInstance> out;
    if (stats != nullptr) {
        stats->clear();
    }
    while (frames_.size() < instances.size()) {
        frames_.push_back(std::make_unique<Frame>());
    }
    for (size_t k = 0; k < instances.size(); ++k) {
        const LodInstance& instance = instances[k];
        if (instance.cloud == nullptr || instance.cloud->splats.count == 0) {
            continue;
        }
        const LodCloud& lod = *instance.cloud;
        Frame& frame = *frames_[k];
        const size_t parts = lod.levels.size() + 1;   // the levels, then the splats
        frame.selected.resize(parts);
        frame.dest.resize(parts);
        frame.totals.resize(parts);
        const auto countOf = [&](size_t part) {
            return part < lod.levels.size() ? lod.levels[part].gaussians.count : lod.splats.count;
        };
        for (size_t part = 0; part < parts; ++part) {
            const uint32_t count = countOf(part);
            if (frame.selected[part].count() < count) {
                auto s = buffer(device, count, 4, "cut.selected");
                if (!s) return std::move(s).error();
                auto d = buffer(device, count, 4, "cut.dest");
                if (!d) return std::move(d).error();
                frame.selected[part] = std::move(*s);
                frame.dest[part] = std::move(*d);
            }
            if (!frame.totals[part].valid()) {
                auto t = buffer(device, 1, 4, "cut.total");
                if (!t) return std::move(t).error();
                frame.totals[part] = std::move(*t);
            }
        }

        // The eye in the cloud's space. Perspective size is edge over distance,
        // which a uniform scale leaves alone; orthographic size is not.
        const render::Mat4 toCloud = aofx::xform::inverseAffine(instance.objectToWorld);
        const render::Vec3 eye = toCloud.point(projection.eyeWorld);
        const double scale = std::max({aofx::xform::length(instance.objectToWorld.column(0)),
                                       aofx::xform::length(instance.objectToWorld.column(1)),
                                       aofx::xform::length(instance.objectToWorld.column(2))});
        const float pixelsPerUnit = static_cast<float>(projection.orthographic ? projection.focalX * scale
                                                                               : projection.focalX);
        const auto setCut = [&](rhi::ShaderCursor p, uint32_t count, uint32_t level, bool coarsest) {
            p["count"].setData(count);
            p["level"].setData(level);
            setBounds(p, lod);
            p["eyeX"].setData(static_cast<float>(eye.x));
            p["eyeY"].setData(static_cast<float>(eye.y));
            p["eyeZ"].setData(static_cast<float>(eye.z));
            p["pixelsPerUnit"].setData(pixelsPerUnit);
            p["threshold"].setData(threshold);
            p["orthographic"].setData(uint32_t{projection.orthographic ? 1u : 0u});
            p["coarsest"].setData(uint32_t{coarsest ? 1u : 0u});
        };
        {
            gpu::CommandBatch batch(device);
            for (size_t part = 0; part < parts; ++part) {
                const uint32_t count = countOf(part);
                if (part < lod.levels.size()) {
                    const LodLevel& level = lod.levels[part];
                    cutGroups_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                        cursor["cells"].setBinding(level.cells.rhi());
                        cursor["selected"].setBinding(frame.selected[part].rhi());
                        setCut(cursor["params"], count, level.level, part == 0);
                    });
                } else {
                    const uint32_t finest = lod.levels.empty() ? 1 : lod.levels.back().level;
                    cutSplats_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                        cursor["cells"].setBinding(lod.keys.rhi());
                        cursor["selected"].setBinding(frame.selected[part].rhi());
                        // No merged level at all: nothing can stand in for a splat.
                        setCut(cursor["params"], count, finest, false);
                        if (lod.levels.empty()) {
                            cursor["params"]["threshold"].setData(0.0F);
                        }
                    });
                }
                LRT_TRY(prefix_.apply(batch, frame.selected[part], frame.dest[part], frame.totals[part], count));
            }
            LRT_TRY(batch.submit(true));
        }
        std::vector<uint32_t> totals(parts, 0);
        uint32_t drawn = 0;
        for (size_t part = 0; part < parts; ++part) {
            LRT_TRY(frame.totals[part].read(device, 0, sizeof(uint32_t), &totals[part]));
            drawn += totals[part];
        }
        if (frame.capacity < drawn || frame.cloud.restPerColour != lod.splats.restPerColour ||
            frame.cloud.shWords != lod.splats.shWords || !frame.cloud.positions.valid()) {
            const uint32_t capacity = std::max(drawn, frame.capacity + frame.capacity / 2);
            auto made = packed(device, std::max(capacity, 1u), lod.splats.restPerColour, lod.splats.shWords,
                               "cut.frame");
            if (!made) return std::move(made).error();
            frame.cloud = std::move(*made);
            frame.capacity = std::max(capacity, 1u);
        }
        frame.cloud.source = lod.splats.source;
        frame.cloud.count = drawn;
        frame.cloud.declared = lod.splats.count;
        frame.cloud.bounds = lod.splats.bounds;
        {
            gpu::CommandBatch batch(device);
            uint32_t base = 0;
            for (size_t part = 0; part < parts; ++part) {
                if (totals[part] == 0) {
                    continue;
                }
                const scene::GpuSplats& source = part < lod.levels.size() ? lod.levels[part].gaussians : lod.splats;
                gather_.dispatch(batch, {countOf(part), 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["selected"].setBinding(frame.selected[part].rhi());
                    cursor["dest"].setBinding(frame.dest[part].rhi());
                    cursor["srcPositions"].setBinding(source.positions.rhi());
                    cursor["srcShape"].setBinding(source.shape.rhi());
                    cursor["srcSh"].setBinding(source.sh.rhi());
                    cursor["positions"].setBinding(frame.cloud.positions.rhi());
                    cursor["shape"].setBinding(frame.cloud.shape.rhi());
                    cursor["sh"].setBinding(frame.cloud.sh.rhi());
                    cursor["params"]["count"].setData(countOf(part));
                    cursor["params"]["base"].setData(base);
                    cursor["params"]["shWords"].setData(lod.splats.shWords);
                });
                base += totals[part];
            }
            LRT_TRY(batch.submit(true));
        }
        if (stats != nullptr) {
            CutStats s;
            s.splats = totals.back();
            s.merged = drawn - totals.back();
            s.available = lod.splats.count;
            stats->push_back(s);
        }
        if (drawn > 0) {
            out.push_back({&frame.cloud, instance.objectToWorld, instance.edit});
        }
    }
    return out;
}

}   // namespace lrt::lod
