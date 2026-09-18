// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/scene/GpuClouds.h"

#include <array>
#include "lrt/scene/DecodeParams.h"
#include "lrt/io/Readers.h"

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
    p["metallic"].setData(e.metallic);
    p["roughness"].setData(e.roughness);
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
    LRT_TRY(make(loader.sogDecode_, "lrt/scene/sog_decode", "sogDecode"));
    LRT_TRY(make(loader.splatDecode_, "lrt/scene/splat_decode", "splatDecode"));
    LRT_TRY(make(loader.pointsValidate_, "lrt/scene/points_validate", "pointsValidate"));
    LRT_TRY(make(loader.pointsDecode_, "lrt/scene/points_decode", "pointsDecode"));
    LRT_TRY(make(loader.splatStreams_, "lrt/scene/streams", "splatStreams"));
    LRT_TRY(make(loader.pointStreams_, "lrt/scene/streams", "pointStreams"));
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

Result<GpuSplats> CloudLoader::startSplats(const std::string& source, uint32_t declared, uint32_t keep,
                                           bool withPbr) {
    GpuSplats splats;
    splats.source = source;
    splats.declared = declared;
    splats.restPerColour = keep;
    splats.shWords = keep == 0 ? 1 : (keep * 3 + 1) / 2;
    auto positions = deviceBuffer(*device_, declared, 16, "splats.positions");
    if (!positions) return std::move(positions).error();
    auto shape = deviceBuffer(*device_, uint64_t{declared} * 4, 4, "splats.shape");
    if (!shape) return std::move(shape).error();
    auto sh = deviceBuffer(*device_, keep == 0 ? 1 : uint64_t{declared} * splats.shWords, 4, "splats.sh");
    if (!sh) return std::move(sh).error();
    splats.positions = *positions;
    splats.shape = *shape;
    splats.sh = *sh;
    if (withPbr) {
        auto pbr = deviceBuffer(*device_, declared, 4, "splats.pbr");
        if (!pbr) return std::move(pbr).error();
        splats.pbr = *pbr;
    }
    return splats;
}

Result<uint32_t> CloudLoader::decodeSlice(const gpu::Buffer& raw, const io::SplatEncoding& e, uint32_t n,
                                          uint32_t written, uint32_t keep, GpuSplats& splats) {
    auto valid = deviceBuffer(*device_, n, 4, "splats.valid");
    if (!valid) return std::move(valid).error();
    auto dest = deviceBuffer(*device_, n, 4, "splats.dest");
    if (!dest) return std::move(dest).error();
    auto total = deviceBuffer(*device_, 1, 4, "splats.total");
    if (!total) return std::move(total).error();

    gpu::CommandBatch batch(*device_);
    splatValidate_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
        cursor["valid"].setBinding(valid->rhi());
        setDecodeParams(cursor, e, n, written, keep, splats.shWords);
    });
    LRT_TRY(prefix_.apply(batch, *valid, *dest, *total, n));
    // The decode does not need the total, so it is queued now: `base` is
    // this slice's start, known from the slices before it.
    splatDecode_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
        cursor["valid"].setBinding(valid->rhi());
        cursor["dest"].setBinding(dest->rhi());
        cursor["positions"].setBinding(splats.positions.rhi());
        cursor["shape"].setBinding(splats.shape.rhi());
        cursor["sh"].setBinding(splats.sh.rhi());
        // Bound whether or not there is one: the offsets in the parameters
        // are what say whether anything is written.
        cursor["pbr"].setBinding(splats.hasPbr() ? splats.pbr.rhi() : splats.shape.rhi());
        setDecodeParams(cursor, e, n, written, keep, splats.shWords);
    });
    LRT_TRY(batch.submit(true));
    uint32_t kept = 0;
    LRT_TRY(total->read(*device_, 0, sizeof(kept), &kept));
    return kept;
}

Result<void> CloudLoader::finishSplats(GpuSplats& splats, uint32_t written) {
    splats.count = written;
    if (splats.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no splat survived validation", splats.source);
    }
    auto bounds = boundsOf(splats.positions, splats.count);
    if (!bounds) return std::move(bounds).error();
    splats.bounds = *bounds;
    log::info("{}: {} splats of {} (degree {})", splats.source, splats.count, splats.declared, splats.degree());
    return ok();
}

Result<GpuSplats> CloudLoader::upload(const io::RawSplats& raw, uint32_t maxDegree) {
    const io::SplatEncoding& e = raw.encoding;
    if (raw.count == 0 || e.floatsPerRecord == 0 ||
        raw.records.size() < size_t{raw.count} * e.floatsPerRecord) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no splat records", raw.source);
    }
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = std::min(e.restPerColour, kPerDegree[std::min(maxDegree, 3u)]);
    const bool pbr = e.metallic != io::SplatEncoding::kNoField || e.roughness != io::SplatEncoding::kNoField;
    auto splats = startSplats(raw.source, raw.count, keep, pbr);
    if (!splats) return std::move(splats).error();

    const uint64_t recordBytes = uint64_t{e.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    uint32_t written = 0;
    for (uint32_t first = 0; first < raw.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, raw.count - first);
        auto rawBuffer = deviceBuffer(*device_, uint64_t{n} * e.floatsPerRecord, 4, "splats.raw",
                                      raw.records.data() + size_t{first} * e.floatsPerRecord);
        if (!rawBuffer) return std::move(rawBuffer).error();
        auto kept = decodeSlice(*rawBuffer, e, n, written, keep, *splats);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    LRT_TRY(finishSplats(*splats, written));
    return splats;
}

struct CloudLoader::SogOnDevice {
    const io::RawSog*  sog = nullptr;
    uint32_t           keep = 0;
    io::SplatEncoding  encoding;
    gpu::Buffer        meansL, meansU, quats, scales, sh0, labels, centroids, codebooks;
};

Result<CloudLoader::SogOnDevice> CloudLoader::sogOnDevice(const io::RawSog& sog, uint32_t maxDegree) {
    SogOnDevice on;
    on.sog = &sog;
    on.keep = std::min(sog.shCoefficients, std::array<uint32_t, 4>{0, 3, 8, 15}[std::min(maxDegree, 3u)]);
    io::SplatEncoding& e = on.encoding;
    e.floatsPerRecord = 14 + on.keep * 3;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14;
    e.restPerColour = on.keep;
    e.restColourOuter = 0;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Log;
    e.colour = io::SplatEncoding::Colour::ShDc;
    e.rotation = io::SplatEncoding::Rotation::Float;

    const auto image = [&](const io::SogImage& from, const char* label, gpu::Buffer& into) -> Result<void> {
        static const uint32_t kPlaceholder = 0;
        const bool none = from.empty();
        auto made = deviceBuffer(*device_, none ? 1 : from.texels.size(), 4, label,
                                 none ? &kPlaceholder : from.texels.data());
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    LRT_TRY(image(sog.meansL, "sog.meansL", on.meansL));
    LRT_TRY(image(sog.meansU, "sog.meansU", on.meansU));
    LRT_TRY(image(sog.quats, "sog.quats", on.quats));
    LRT_TRY(image(sog.scales, "sog.scales", on.scales));
    LRT_TRY(image(sog.sh0, "sog.sh0", on.sh0));
    LRT_TRY(image(on.keep > 0 ? sog.shLabels : io::SogImage{}, "sog.shLabels", on.labels));
    LRT_TRY(image(on.keep > 0 ? sog.shCentroids : io::SogImage{}, "sog.shCentroids", on.centroids));
    std::vector<float> books(768, 0.0F);
    const auto place = [&](const std::vector<float>& book, size_t at) {
        std::copy_n(book.begin(), std::min<size_t>(book.size(), 256), books.begin() + static_cast<long>(at));
    };
    place(sog.scalesBook, 0);
    place(sog.sh0Book, 256);
    place(sog.shNBook, 512);
    auto codebooks = deviceBuffer(*device_, books.size(), 4, "sog.codebooks", books.data());
    if (!codebooks) return std::move(codebooks).error();
    on.codebooks = std::move(*codebooks);
    return on;
}

Result<void> CloudLoader::sogSlice(const SogOnDevice& on, uint32_t first, uint32_t n, const gpu::Buffer& into) {
    const io::RawSog& sog = *on.sog;
    gpu::CommandBatch batch(*device_);
    sogDecode_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["meansL"].setBinding(on.meansL.rhi());
        cursor["meansU"].setBinding(on.meansU.rhi());
        cursor["quats"].setBinding(on.quats.rhi());
        cursor["scales"].setBinding(on.scales.rhi());
        cursor["sh0"].setBinding(on.sh0.rhi());
        cursor["shLabels"].setBinding(on.labels.rhi());
        cursor["shCentroids"].setBinding(on.centroids.rhi());
        cursor["codebooks"].setBinding(on.codebooks.rhi());
        cursor["records"].setBinding(into.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["count"].setData(n);
        p["first"].setData(first);
        p["stride"].setData(on.encoding.floatsPerRecord);
        p["keep"].setData(on.keep);
        p["version"].setData(sog.version);
        p["coefficients"].setData(sog.shCoefficients);
        p["centroidsWidth"].setData(sog.shCentroids.width);
        static constexpr const char* kMeans[6] = {"meansMinX", "meansMinY", "meansMinZ",
                                                  "meansMaxX", "meansMaxY", "meansMaxZ"};
        static constexpr const char* kScales[6] = {"scalesMinX", "scalesMinY", "scalesMinZ",
                                                   "scalesMaxX", "scalesMaxY", "scalesMaxZ"};
        static constexpr const char* kSh0[8] = {"sh0MinR", "sh0MinG", "sh0MinB", "sh0MinA",
                                                "sh0MaxR", "sh0MaxG", "sh0MaxB", "sh0MaxA"};
        for (size_t k = 0; k < 3; ++k) {
            p[kMeans[k]].setData(sog.meansMin[k]);
            p[kMeans[k + 3]].setData(sog.meansMax[k]);
            p[kScales[k]].setData(sog.scalesMin[k]);
            p[kScales[k + 3]].setData(sog.scalesMax[k]);
        }
        for (size_t k = 0; k < 4; ++k) {
            p[kSh0[k]].setData(sog.sh0Min[k]);
            p[kSh0[k + 4]].setData(sog.sh0Max[k]);
        }
        p["shNMin"].setData(sog.shNMin);
        p["shNMax"].setData(sog.shNMax);
    });
    return batch.submit(true);
}

Result<GpuSplats> CloudLoader::upload(const io::RawSog& sog, uint32_t maxDegree) {
    auto on = sogOnDevice(sog, maxDegree);
    if (!on) return std::move(on).error();
    auto splats = startSplats(sog.source, sog.count, on->keep);
    if (!splats) return std::move(splats).error();
    const uint64_t recordBytes = uint64_t{on->encoding.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    uint32_t written = 0;
    for (uint32_t first = 0; first < sog.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, sog.count - first);
        auto records = deviceBuffer(*device_, uint64_t{n} * on->encoding.floatsPerRecord, 4, "sog.records");
        if (!records) return std::move(records).error();
        LRT_TRY(sogSlice(*on, first, n, *records));
        auto kept = decodeSlice(*records, on->encoding, n, written, on->keep, *splats);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    LRT_TRY(finishSplats(*splats, written));
    return splats;
}

Result<io::RawSplats> CloudLoader::records(const io::RawSog& sog, uint32_t maxDegree) {
    auto on = sogOnDevice(sog, maxDegree);
    if (!on) return std::move(on).error();
    io::RawSplats raw;
    raw.source = sog.source;
    raw.count = sog.count;
    raw.encoding = on->encoding;
    raw.records.resize(size_t{sog.count} * on->encoding.floatsPerRecord);
    const uint64_t recordBytes = uint64_t{on->encoding.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    for (uint32_t first = 0; first < sog.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, sog.count - first);
        auto buffer = deviceBuffer(*device_, uint64_t{n} * on->encoding.floatsPerRecord, 4, "sog.records");
        if (!buffer) return std::move(buffer).error();
        LRT_TRY(sogSlice(*on, first, n, *buffer));
        LRT_TRY(buffer->read(*device_, 0, uint64_t{n} * recordBytes,
                             raw.records.data() + size_t{first} * on->encoding.floatsPerRecord));
    }
    return raw;
}

Result<uint32_t> CloudLoader::decodePoints(const gpu::Buffer& raw, uint32_t n, uint32_t first,
                                           uint32_t colourKind, float detail, uint32_t written,
                                           GpuPoints& points) {
    auto valid = deviceBuffer(*device_, n, 4, "points.valid");
    if (!valid) return std::move(valid).error();
    auto dest = deviceBuffer(*device_, n, 4, "points.dest");
    if (!dest) return std::move(dest).error();
    auto total = deviceBuffer(*device_, 1, 4, "points.total");
    if (!total) return std::move(total).error();
    const auto params = [&](rhi::ShaderCursor cursor) {
        cursor["params"]["count"].setData(n);
        cursor["params"]["base"].setData(first);
        cursor["params"]["colourKind"].setData(colourKind);
        cursor["params"]["detail"].setData(detail);
    };
    gpu::CommandBatch batch(*device_);
    pointsValidate_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
        cursor["valid"].setBinding(valid->rhi());
        params(cursor);
    });
    LRT_TRY(prefix_.apply(batch, *valid, *dest, *total, n));
    LRT_TRY(batch.submit(true));
    gpu::CommandBatch decode(*device_);
    pointsDecode_.dispatch(decode, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
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
    return kept;
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
        auto kept = decodePoints(*rawBuffer, n, first, raw.colourKind, detail, written, points);
        if (!kept) return std::move(kept).error();
        written += *kept;
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

Result<gpu::Buffer> CloudLoader::streamBuffer(const FloatStream& stream, const char* label) {
    // Words of four bytes; an odd count of halves leaves the last word short,
    // so the bytes are written into a buffer already the whole word long.
    const uint64_t words = std::max<uint64_t>((stream.bytes.size() + 3) / 4, 1);
    auto made = deviceBuffer(*device_, words, 4, label);
    if (!made) return std::move(made).error();
    if (!stream.empty()) {
        LRT_TRY(made->write(*device_, 0, stream.bytes.size(), stream.bytes.data()));
    }
    return made;
}

namespace {

constexpr uint32_t kPositions = 1, kRotations = 2, kScales = 4, kOpacities = 8, kSh = 16, kColours = 32,
                   kMetallic = 64, kRoughness = 128;

}   // namespace

Result<GpuSplats> CloudLoader::upload(const SplatStreams& in, uint32_t maxDegree) {
    const uint32_t n = in.count;
    const auto holds = [&](const FloatStream& stream, uint64_t perElement) {
        return stream.empty() || stream.values() >= uint64_t{n} * perElement;
    };
    const bool shapes = in.coefficients == 0 || in.coefficients == 1 || in.coefficients == 4 ||
                        in.coefficients == 9 || in.coefficients == 16;
    if (n == 0 || in.positions.values() < uint64_t{n} * 3 || !holds(in.rotations, 4) || !holds(in.scales, 3) ||
        !holds(in.opacities, 1) || !shapes || !holds(in.sh, uint64_t{in.coefficients} * 3)) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': splat arrays of the wrong lengths", in.source);
    }
    const bool haveSh = !in.sh.empty() && in.coefficients > 0;
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = haveSh ? std::min(in.coefficients - 1, kPerDegree[std::min(maxDegree, 3u)]) : 0;

    const bool havePbr = !in.metallic.empty() || !in.roughness.empty();
    io::SplatEncoding e;
    e.floatsPerRecord = 14 + keep * 3 + (havePbr ? 2 : 0);
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = keep; e.restColourOuter = 0;
    if (havePbr) {
        e.metallic = 14 + keep * 3;
        e.roughness = e.metallic + 1;
    }
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    e.colour = io::SplatEncoding::Colour::ShDc;
    e.rotation = io::SplatEncoding::Rotation::Float;

    uint32_t present = kPositions;
    uint32_t halves = in.positions.half ? kPositions : 0;
    const auto note = [&](const FloatStream& stream, uint32_t bit) {
        present |= stream.empty() ? 0 : bit;
        halves |= (!stream.empty() && stream.half) ? bit : 0;
    };
    note(in.rotations, kRotations);
    note(in.scales, kScales);
    note(in.opacities, kOpacities);
    note(in.metallic, kMetallic);
    note(in.roughness, kRoughness);
    if (haveSh) {
        note(in.sh, kSh);
    }
    auto positions = streamBuffer(in.positions, "splats.stream.positions");
    if (!positions) return std::move(positions).error();
    auto rotations = streamBuffer(in.rotations, "splats.stream.rotations");
    if (!rotations) return std::move(rotations).error();
    auto scales = streamBuffer(in.scales, "splats.stream.scales");
    if (!scales) return std::move(scales).error();
    auto opacities = streamBuffer(in.opacities, "splats.stream.opacities");
    if (!opacities) return std::move(opacities).error();
    auto sh = streamBuffer(haveSh ? in.sh : FloatStream{}, "splats.stream.sh");
    if (!sh) return std::move(sh).error();
    auto none = streamBuffer({}, "splats.stream.none");
    if (!none) return std::move(none).error();

    auto metallic = streamBuffer(in.metallic, "splats.stream.metallic");
    if (!metallic) return std::move(metallic).error();
    auto roughness = streamBuffer(in.roughness, "splats.stream.roughness");
    if (!roughness) return std::move(roughness).error();
    auto splats = startSplats(in.source, n, keep, havePbr);
    if (!splats) return std::move(splats).error();
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / (uint64_t{e.floatsPerRecord} * 4)));
    uint32_t written = 0;
    for (uint32_t first = 0; first < n; first += perSlice) {
        const uint32_t count = std::min(perSlice, n - first);
        auto records = deviceBuffer(*device_, uint64_t{count} * e.floatsPerRecord, 4, "splats.raw");
        if (!records) return std::move(records).error();
        {
            gpu::CommandBatch batch(*device_);
            splatStreams_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(positions->rhi());
                cursor["rotations"].setBinding(rotations->rhi());
                cursor["scales"].setBinding(scales->rhi());
                cursor["opacities"].setBinding(opacities->rhi());
                cursor["sh"].setBinding(sh->rhi());
                cursor["colours"].setBinding(none->rhi());
                cursor["metallic"].setBinding(metallic->rhi());
                cursor["roughness"].setBinding(roughness->rhi());
                cursor["records"].setBinding(records->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["first"].setData(first);
                p["stride"].setData(e.floatsPerRecord);
                p["keep"].setData(keep);
                p["coefficients"].setData(in.coefficients);
                p["present"].setData(present);
                p["halves"].setData(halves);
                p["colourMode"].setData(uint32_t{0});
                p["pbrBase"].setData(havePbr ? e.metallic : io::SplatEncoding::kNoField);
            });
            LRT_TRY(batch.submit(true));
        }
        auto kept = decodeSlice(*records, e, count, written, keep, *splats);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    LRT_TRY(finishSplats(*splats, written));
    return splats;
}

Result<GpuPoints> CloudLoader::upload(const PointStreams& in, float detail) {
    const uint32_t n = in.count;
    const uint64_t colours = in.colours.values() / 3;
    if (n == 0 || in.positions.values() < uint64_t{n} * 3 || (!in.colours.empty() && colours != 1 && colours < n)) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': point arrays of the wrong lengths", in.source);
    }
    const uint32_t colourMode = in.colours.empty() ? 0 : colours >= n && n > 1 ? 2 : 1;
    GpuPoints points;
    points.source = in.source;
    points.declared = n;
    auto positionsOut = deviceBuffer(*device_, n, 16, "points.positions");
    if (!positionsOut) return std::move(positionsOut).error();
    auto coloursOut = deviceBuffer(*device_, uint64_t{n} * 2, 4, "points.colours");
    if (!coloursOut) return std::move(coloursOut).error();
    points.positions = *positionsOut;
    points.colours = *coloursOut;

    auto positions = streamBuffer(in.positions, "points.stream.positions");
    if (!positions) return std::move(positions).error();
    auto colourStream = streamBuffer(in.colours, "points.stream.colours");
    if (!colourStream) return std::move(colourStream).error();
    auto none = streamBuffer({}, "points.stream.none");
    if (!none) return std::move(none).error();
    const uint32_t halves = (in.positions.half ? kPositions : 0) | (in.colours.half ? kColours : 0);

    const uint32_t perSlice = static_cast<uint32_t>(kSliceBytes / 24);
    uint32_t written = 0;
    for (uint32_t first = 0; first < n; first += perSlice) {
        const uint32_t count = std::min(perSlice, n - first);
        auto records = deviceBuffer(*device_, uint64_t{count} * 6, 4, "points.raw");
        if (!records) return std::move(records).error();
        {
            gpu::CommandBatch batch(*device_);
            pointStreams_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(positions->rhi());
                cursor["colours"].setBinding(colourStream->rhi());
                for (const char* unused : {"rotations", "scales", "opacities", "sh", "metallic", "roughness"}) {
                    cursor[unused].setBinding(none->rhi());
                }
                cursor["records"].setBinding(records->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["first"].setData(first);
                p["stride"].setData(uint32_t{6});
                p["keep"].setData(uint32_t{0});
                p["coefficients"].setData(uint32_t{0});
                p["present"].setData(kPositions | (colourMode != 0 ? kColours : 0));
                p["halves"].setData(halves);
                p["colourMode"].setData(colourMode);
                p["pbrBase"].setData(io::SplatEncoding::kNoField);
            });
            LRT_TRY(batch.submit(true));
        }
        auto kept = decodePoints(*records, count, first, colourMode != 0 ? 2u : 0u, detail, written, points);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    points.count = written;
    if (points.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no point survived", in.source);
    }
    auto bounds = boundsOf(points.positions, points.count);
    if (!bounds) return std::move(bounds).error();
    points.bounds = *bounds;
    log::info("{}: {} points of {}", in.source, points.count, points.declared);
    return points;
}

bool isSog(const std::filesystem::path& path) {
    return path.extension() == ".sog" || path.filename() == "meta.json";
}

Result<GpuSplats> loadSplatFile(CloudLoader& loader, const std::filesystem::path& path, uint32_t maxDegree) {
    if (isSog(path)) {
        auto sog = io::readSog(path);
        if (!sog) return std::move(sog).error();
        return loader.upload(*sog, maxDegree);
    }
    auto raw = io::readSplats(path);
    if (!raw) return std::move(raw).error();
    return loader.upload(*raw, maxDegree);
}

Result<io::RawSplats> readSplatRecords(CloudLoader& loader, const std::filesystem::path& path, uint32_t maxDegree) {
    if (isSog(path)) {
        auto sog = io::readSog(path);
        if (!sog) return std::move(sog).error();
        return loader.records(*sog, maxDegree);
    }
    return io::readSplats(path);
}

}   // namespace lrt::scene
