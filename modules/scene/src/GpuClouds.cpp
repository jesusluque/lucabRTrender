// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/scene/GpuClouds.h"
#include "lrt/scene/DecodeParams.h"

#include <algorithm>

#include "lrt/core/Log.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::scene {
namespace {

constexpr uint32_t kChunk = 4096;

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element,
                                 const std::string& label, const void* data = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, data);
}

}   // namespace

void setDecodeParams(rhi::ShaderCursor cursor, const io::SplatEncoding& e, uint32_t count,
                     uint32_t base, uint32_t keepPerColour, uint32_t shWords) {
    rhi::ShaderCursor p = cursor["params"];
    p["count"].setData(count);
    p["stride"].setData(e.floatsPerRecord);
    p["base"].setData(base);
    p["keepPerColour"].setData(keepPerColour);
    p["x"].setData(e.x);
    p["y"].setData(e.y);
    p["z"].setData(e.z);
    p["opacity"].setData(e.opacity);
    p["scale0"].setData(e.scale0);
    p["scale1"].setData(e.scale1);
    p["scale2"].setData(e.scale2);
    p["rotW"].setData(e.rotW);
    p["rotX"].setData(e.rotX);
    p["rotY"].setData(e.rotY);
    p["rotZ"].setData(e.rotZ);
    p["dc0"].setData(e.dc0);
    p["dc1"].setData(e.dc1);
    p["dc2"].setData(e.dc2);
    p["restBase"].setData(e.restBase);
    p["filePerColour"].setData(e.restPerColour);
    p["restColourOuter"].setData(e.restColourOuter);
    p["opacityMode"].setData(static_cast<uint32_t>(e.opacity_));
    p["scaleMode"].setData(static_cast<uint32_t>(e.scale_));
    p["colourMode"].setData(static_cast<uint32_t>(e.colour));
    p["rotationMode"].setData(static_cast<uint32_t>(e.rotation));
    p["restMode"].setData(static_cast<uint32_t>(e.rest));
    p["flipYZ"].setData(uint32_t{e.flipYZ ? 1u : 0u});
    p["positionScale"].setData(e.positionScale);
    p["shWords"].setData(shWords);
}


Result<CloudLoader> CloudLoader::create(gpu::ShaderLibrary& library) {
    CloudLoader loader;
    loader.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    loader.prefix_ = std::move(*prefix);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    LRT_TRY(make(loader.splatValidate_, "lrt/scene/splat_validate", "splatValidate"));
    LRT_TRY(make(loader.splatDecode_, "lrt/scene/splat_decode", "splatDecode"));
    LRT_TRY(make(loader.pointsValidate_, "lrt/scene/points_validate", "pointsValidate"));
    LRT_TRY(make(loader.pointsDecode_, "lrt/scene/points_decode", "pointsDecode"));
    LRT_TRY(make(loader.boundsChunks_, "lrt/scene/bounds_chunks", "boundsChunks"));
    LRT_TRY(make(loader.boundsReduce_, "lrt/scene/bounds_reduce", "boundsReduce"));
    return loader;
}

Result<Bounds> CloudLoader::boundsOf(const gpu::Buffer& positions, uint32_t count) {
    Bounds bounds;
    if (count == 0) {
        return bounds;
    }
    const uint32_t chunks = (count + kChunk - 1) / kChunk;
    auto extents = deviceBuffer(*device_, uint64_t{chunks} * 2, 16, "bounds.extents");
    if (!extents) return std::move(extents).error();
    auto result = deviceBuffer(*device_, 2, 16, "bounds.result");
    if (!result) return std::move(result).error();
    gpu::CommandBatch batch(*device_);
    const auto params = [&](rhi::ShaderCursor cursor) {
        cursor["params"]["count"].setData(count);
        cursor["params"]["chunkSize"].setData(kChunk);
        cursor["params"]["chunkCount"].setData(chunks);
    };
    boundsChunks_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(positions.rhi());
        cursor["extents"].setBinding(extents->rhi());
        params(cursor);
    });
    boundsReduce_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["extents"].setBinding(extents->rhi());
        cursor["result"].setBinding(result->rhi());
        params(cursor);
    });
    LRT_TRY(batch.submit(true));
    float extent[8];
    LRT_TRY(result->read(*device_, 0, sizeof(extent), extent));
    bounds.min = {extent[0], extent[1], extent[2]};
    bounds.max = {extent[4], extent[5], extent[6]};
    return bounds;
}

Result<GpuSplats> CloudLoader::upload(const io::RawSplats& raw, uint32_t maxDegree) {
    const io::SplatEncoding& e = raw.encoding;
    if (raw.count == 0 || e.floatsPerRecord == 0 ||
        raw.records.size() < size_t{raw.count} * e.floatsPerRecord) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no splat records", raw.source);
    }
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = std::min(e.restPerColour, kPerDegree[std::min(maxDegree, 3u)]);

    GpuSplats splats;
    splats.source = raw.source;
    splats.declared = raw.count;
    splats.restPerColour = keep;
    splats.shWords = keep == 0 ? 1 : (keep * 3 + 1) / 2;

    auto positions = deviceBuffer(*device_, raw.count, 16, "splats.positions");
    if (!positions) return std::move(positions).error();
    auto shape = deviceBuffer(*device_, uint64_t{raw.count} * 4, 4, "splats.shape");
    if (!shape) return std::move(shape).error();
    auto sh = deviceBuffer(*device_, keep == 0 ? 1 : uint64_t{raw.count} * splats.shWords, 4,
                           "splats.sh");
    if (!sh) return std::move(sh).error();
    splats.positions = *positions;
    splats.shape = *shape;
    splats.sh = *sh;

    const uint64_t recordBytes = uint64_t{e.floatsPerRecord} * 4;
    const uint32_t perSlice =
        static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    uint32_t written = 0;
    for (uint32_t first = 0; first < raw.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, raw.count - first);
        auto rawBuffer = deviceBuffer(*device_, uint64_t{n} * e.floatsPerRecord, 4, "splats.raw",
                                      raw.records.data() + size_t{first} * e.floatsPerRecord);
        if (!rawBuffer) return std::move(rawBuffer).error();
        auto valid = deviceBuffer(*device_, n, 4, "splats.valid");
        if (!valid) return std::move(valid).error();
        auto dest = deviceBuffer(*device_, n, 4, "splats.dest");
        if (!dest) return std::move(dest).error();
        auto total = deviceBuffer(*device_, 1, 4, "splats.total");
        if (!total) return std::move(total).error();

        gpu::CommandBatch batch(*device_);
        splatValidate_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["raw"].setBinding(rawBuffer->rhi());
            cursor["valid"].setBinding(valid->rhi());
            setDecodeParams(cursor, e, n, written, keep, splats.shWords);
        });
        LRT_TRY(prefix_.apply(batch, *valid, *dest, *total, n));
        // The decode does not need the total, so it is queued now: `base` is
        // this slice's start, known from the slices before it.
        splatDecode_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["raw"].setBinding(rawBuffer->rhi());
            cursor["valid"].setBinding(valid->rhi());
            cursor["dest"].setBinding(dest->rhi());
            cursor["positions"].setBinding(splats.positions.rhi());
            cursor["shape"].setBinding(splats.shape.rhi());
            cursor["sh"].setBinding(splats.sh.rhi());
            setDecodeParams(cursor, e, n, written, keep, splats.shWords);
        });
        LRT_TRY(batch.submit(true));
        uint32_t kept = 0;
        LRT_TRY(total->read(*device_, 0, sizeof(kept), &kept));
        written += kept;
    }
    splats.count = written;
    if (splats.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no splat survived validation",
                           raw.source);
    }
    auto bounds = boundsOf(splats.positions, splats.count);
    if (!bounds) return std::move(bounds).error();
    splats.bounds = *bounds;
    log::info("{}: {} splats of {} (degree {})", raw.source, splats.count, splats.declared,
              splats.degree());
    return splats;
}

Result<GpuPoints> CloudLoader::upload(const io::RawPoints& raw, float detail) {
    if (raw.count == 0 || raw.records.size() < size_t{raw.count} * 6) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no points", raw.source);
    }
    GpuPoints points;
    points.source = raw.source;
    points.declared = raw.count;
    auto positions = deviceBuffer(*device_, raw.count, 16, "points.positions");
    if (!positions) return std::move(positions).error();
    auto colours = deviceBuffer(*device_, uint64_t{raw.count} * 2, 4, "points.colours");
    if (!colours) return std::move(colours).error();
    points.positions = *positions;
    points.colours = *colours;

    const uint32_t perSlice = static_cast<uint32_t>(kSliceBytes / 24);
    uint32_t written = 0;
    for (uint32_t first = 0; first < raw.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, raw.count - first);
        auto rawBuffer = deviceBuffer(*device_, uint64_t{n} * 6, 4, "points.raw",
                                      raw.records.data() + size_t{first} * 6);
        if (!rawBuffer) return std::move(rawBuffer).error();
        auto valid = deviceBuffer(*device_, n, 4, "points.valid");
        if (!valid) return std::move(valid).error();
        auto dest = deviceBuffer(*device_, n, 4, "points.dest");
        if (!dest) return std::move(dest).error();
        auto total = deviceBuffer(*device_, 1, 4, "points.total");
        if (!total) return std::move(total).error();
        const auto params = [&](rhi::ShaderCursor cursor) {
            cursor["params"]["count"].setData(n);
            cursor["params"]["base"].setData(first);
            cursor["params"]["colourKind"].setData(raw.colourKind);
            cursor["params"]["detail"].setData(detail);
        };
        gpu::CommandBatch batch(*device_);
        pointsValidate_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["raw"].setBinding(rawBuffer->rhi());
            cursor["valid"].setBinding(valid->rhi());
            params(cursor);
        });
        LRT_TRY(prefix_.apply(batch, *valid, *dest, *total, n));
        LRT_TRY(batch.submit(true));
        gpu::CommandBatch decode(*device_);
        pointsDecode_.dispatch(decode, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["raw"].setBinding(rawBuffer->rhi());
            cursor["valid"].setBinding(valid->rhi());
            cursor["dest"].setBinding(dest->rhi());
            cursor["positions"].setBinding(points.positions.rhi());
            cursor["colours"].setBinding(points.colours.rhi());
            params(cursor);
            // The decode writes at base + dest; base here is where this
            // slice's survivors start, not where its records started.
            cursor["params"]["base"].setData(written);
        });
        LRT_TRY(decode.submit(true));
        uint32_t kept = 0;
        LRT_TRY(total->read(*device_, 0, sizeof(kept), &kept));
        written += kept;
    }
    points.count = written;
    if (points.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no point survived", raw.source);
    }
    auto bounds = boundsOf(points.positions, points.count);
    if (!bounds) return std::move(bounds).error();
    points.bounds = *bounds;
    log::info("{}: {} points of {}", raw.source, points.count, points.declared);
    return points;
}

}   // namespace lrt::scene
