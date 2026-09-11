// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/lod/Lrtc.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include "lrt/core/Log.h"
#include "lrt/core/Platform.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"

namespace lrt::lod {
namespace {

constexpr uint64_t kPage = 4096;
constexpr uint32_t kVersion = 1;
constexpr char     kMagic[4] = {'L', 'R', 'T', 'C'};
/// Uploads staged before a submit: the staging heap holds them until then.
constexpr uint64_t kStageBytes = uint64_t{256} << 20;

struct FileHeader {
    char     magic[4];
    uint32_t version;
    uint32_t count;           // splats
    uint32_t restPerColour;
    uint32_t shWords;
    uint32_t levels;
    uint32_t chunkSplats;
    uint32_t chunks;
    uint32_t finestGroups;
    float    boundsLo[3];
    float    extent;
    float    boundsMin[3];
    float    boundsMax[3];
    uint32_t pad;
    uint64_t levelTable;      // LevelEntry[levels]
    uint64_t chunkTable;      // ChunkEntry[chunks]
    uint64_t starts;          // uint[finestGroups]
};
struct LevelEntry {
    uint32_t level;
    uint32_t groups;
    uint64_t offset;
};
struct ChunkEntry {
    uint64_t offset;
    uint32_t count;
    uint32_t pad;
};
static_assert(sizeof(FileHeader) == 104);
static_assert(sizeof(LevelEntry) == 16);
static_assert(sizeof(ChunkEntry) == 16);

uint64_t aligned(uint64_t at) {
    return (at + kPage - 1) / kPage * kPage;
}

/// Bytes a level group or a splat takes: position, shape, SH, and a uint.
uint64_t elementBytes(uint32_t shWords) {
    return 16 + 16 + 4 * uint64_t{shWords} + 4;
}

struct Layout {
    FileHeader              header{};
    std::vector<LevelEntry> levels;
    std::vector<ChunkEntry> chunks;
};

Error bad(const std::filesystem::path& path, const std::string& why) {
    return Error(ErrorCode::IoFailure, path.string() + ": not a readable .lrtc (" + why + ")");
}

Result<Layout> parse(const platform::MappedFile& file, const std::filesystem::path& path) {
    const std::span<const std::byte> bytes = file.bytes();
    Layout layout;
    FileHeader& h = layout.header;
    if (bytes.size() < kPage) {
        return bad(path, "shorter than its header");
    }
    std::memcpy(&h, bytes.data(), sizeof h);
    if (std::memcmp(h.magic, kMagic, 4) != 0) {
        return bad(path, "no LRTC magic");
    }
    if (h.version != kVersion) {
        return bad(path, "version " + std::to_string(h.version) + ", this reads " + std::to_string(kVersion));
    }
    const auto within = [&](uint64_t offset, uint64_t size) {
        return offset <= bytes.size() && size <= bytes.size() - offset;
    };
    if (h.count == 0 || h.levels == 0 || h.chunkSplats == 0 || h.shWords == 0 || h.shWords > 64 ||
        h.chunks != (uint64_t{h.count} + h.chunkSplats - 1) / h.chunkSplats) {
        return bad(path, "inconsistent counts");
    }
    if (!within(h.levelTable, uint64_t{h.levels} * sizeof(LevelEntry)) ||
        !within(h.chunkTable, uint64_t{h.chunks} * sizeof(ChunkEntry)) ||
        !within(h.starts, uint64_t{h.finestGroups} * 4)) {
        return bad(path, "tables past the end");
    }
    layout.levels.resize(h.levels);
    layout.chunks.resize(h.chunks);
    std::memcpy(layout.levels.data(), bytes.data() + h.levelTable, layout.levels.size() * sizeof(LevelEntry));
    std::memcpy(layout.chunks.data(), bytes.data() + h.chunkTable, layout.chunks.size() * sizeof(ChunkEntry));
    const uint64_t per = elementBytes(h.shWords);
    for (size_t l = 0; l < layout.levels.size(); ++l) {
        const LevelEntry& e = layout.levels[l];
        if (e.groups == 0 || !within(e.offset, e.groups * per) ||
            (l > 0 && e.level != layout.levels[l - 1].level + 1) || e.level == 0 || e.level > 10) {
            return bad(path, "level " + std::to_string(l));
        }
    }
    if (layout.levels.back().groups != h.finestGroups) {
        return bad(path, "finest level's groups");
    }
    for (uint32_t c = 0; c < h.chunks; ++c) {
        const ChunkEntry& e = layout.chunks[c];
        const uint32_t expected = std::min(h.chunkSplats, h.count - c * h.chunkSplats);
        if (e.count != expected || !within(e.offset, e.count * per)) {
            return bad(path, "chunk " + std::to_string(c));
        }
    }
    return layout;
}

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                                 const void* initial = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, count > 0 ? initial : nullptr);
}

/// A block's four arrays, where they are in the mapping.
struct Block {
    const std::byte* positions, *shape, *sh, *tail;
};
Block blockAt(const std::byte* at, uint64_t n, uint32_t shWords) {
    Block b;
    b.positions = at;
    b.shape = b.positions + n * 16;
    b.sh = b.shape + n * 16;
    b.tail = b.sh + n * 4 * uint64_t{shWords};
    return b;
}

scene::GpuSplats splatsLike(const FileHeader& h, const std::filesystem::path& path) {
    scene::GpuSplats s;
    s.source = path.filename().string();
    s.restPerColour = h.restPerColour;
    s.shWords = h.shWords;
    std::copy(h.boundsMin, h.boundsMin + 3, s.bounds.min.begin());
    std::copy(h.boundsMax, h.boundsMax + 3, s.bounds.max.begin());
    return s;
}

/// The cloud but its chunks: header fields, levels and starts on the device.
Result<LodCloud> openCloud(gpu::Device& device, const platform::MappedFile& file, const Layout& layout,
                           const std::filesystem::path& path) {
    const FileHeader& h = layout.header;
    LodCloud lod;
    lod.count = h.count;
    lod.chunkSplats = h.chunkSplats;
    std::copy(h.boundsLo, h.boundsLo + 3, lod.boundsLo);
    lod.extent = h.extent;
    const std::byte* base = file.bytes().data();
    for (const LevelEntry& e : layout.levels) {
        const Block b = blockAt(base + e.offset, e.groups, h.shWords);
        LodLevel level;
        level.level = e.level;
        level.gaussians = splatsLike(h, path);
        level.gaussians.count = e.groups;
        level.gaussians.declared = e.groups;
        auto p = deviceBuffer(device, e.groups, 16, "lrtc.level", b.positions);
        if (!p) return std::move(p).error();
        auto s = deviceBuffer(device, uint64_t{e.groups} * 4, 4, "lrtc.level", b.shape);
        if (!s) return std::move(s).error();
        auto sh = deviceBuffer(device, uint64_t{e.groups} * h.shWords, 4, "lrtc.level", b.sh);
        if (!sh) return std::move(sh).error();
        auto cells = deviceBuffer(device, e.groups, 4, "lrtc.cells", b.tail);
        if (!cells) return std::move(cells).error();
        level.gaussians.positions = std::move(*p);
        level.gaussians.shape = std::move(*s);
        level.gaussians.sh = std::move(*sh);
        level.cells = std::move(*cells);
        lod.levels.push_back(std::move(level));
    }
    auto starts = deviceBuffer(device, h.finestGroups, 4, "lrtc.starts", base + h.starts);
    if (!starts) return std::move(starts).error();
    lod.starts = std::move(*starts);
    return lod;
}

/// A store of `slots` chunks, and the per-chunk flags, all empty.
Result<void> makeStore(gpu::Device& device, LodCloud& lod, const FileHeader& h, uint32_t slots,
                       const std::filesystem::path& path) {
    const uint64_t n = uint64_t{slots} * h.chunkSplats;
    lod.splats = splatsLike(h, path);
    lod.splats.count = static_cast<uint32_t>(std::min<uint64_t>(n, h.count));
    lod.splats.declared = h.count;
    auto p = deviceBuffer(device, n, 16, "lrtc.store");
    if (!p) return std::move(p).error();
    auto s = deviceBuffer(device, n * 4, 4, "lrtc.store");
    if (!s) return std::move(s).error();
    auto sh = deviceBuffer(device, n * h.shWords, 4, "lrtc.store");
    if (!sh) return std::move(sh).error();
    auto g = deviceBuffer(device, n, 4, "lrtc.groups");
    if (!g) return std::move(g).error();
    lod.splats.positions = std::move(*p);
    lod.splats.shape = std::move(*s);
    lod.splats.sh = std::move(*sh);
    lod.groups = std::move(*g);
    lod.slots.assign(h.chunks, -1);
    const std::vector<uint32_t> none(h.chunks, 0);
    auto resident = gpu::Buffer::fromSpan(device, std::span<const uint32_t>(none), "lrtc.resident");
    if (!resident) return std::move(resident).error();
    lod.resident = std::move(*resident);
    return ok();
}

/// Records chunk `chunk`'s bytes (one block, `from`) into slot `slot`.
void place(gpu::CommandBatch& batch, LodCloud& lod, uint32_t chunk, uint32_t slot, const std::byte* from) {
    const uint64_t n = lod.chunkCount(chunk);
    const uint64_t at = uint64_t{slot} * lod.chunkSplats;
    const Block b = blockAt(from, n, lod.splats.shWords);
    rhi::ICommandEncoder* e = batch.encoder();
    e->uploadBufferData(lod.splats.positions.rhi(), at * 16, n * 16, b.positions);
    e->uploadBufferData(lod.splats.shape.rhi(), at * 16, n * 16, b.shape);
    e->uploadBufferData(lod.splats.sh.rhi(), at * 4 * lod.splats.shWords, n * 4 * lod.splats.shWords, b.sh);
    e->uploadBufferData(lod.groups.rhi(), at * 4, n * 4, b.tail);
    const uint32_t one = 1;
    e->uploadBufferData(lod.resident.rhi(), uint64_t{chunk} * 4, 4, &one);
    batch.markDirty();
    lod.slots[chunk] = static_cast<int32_t>(slot);
}

}   // namespace

bool isLrtc(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".lrtc";
}

Result<void> writeLrtc(gpu::Device& device, const LodCloud& cloud, const std::filesystem::path& path) {
    if (cloud.levels.empty() || cloud.count == 0) {
        return Error(ErrorCode::InvalidArgument, "a .lrtc holds a cloud with levels of detail");
    }
    for (uint32_t c = 0; c < cloud.chunks(); ++c) {
        if (cloud.slots[c] < 0) {
            return Error(ErrorCode::InvalidArgument, "writing a cloud whose chunks are not all on the device");
        }
    }
    const uint32_t shWords = cloud.splats.shWords;
    const uint64_t per = elementBytes(shWords);
    FileHeader h{};
    std::memcpy(h.magic, kMagic, 4);
    h.version = kVersion;
    h.count = cloud.count;
    h.restPerColour = cloud.splats.restPerColour;
    h.shWords = shWords;
    h.levels = static_cast<uint32_t>(cloud.levels.size());
    h.chunkSplats = cloud.chunkSplats;
    h.chunks = cloud.chunks();
    h.finestGroups = cloud.levels.back().gaussians.count;
    std::copy(cloud.boundsLo, cloud.boundsLo + 3, h.boundsLo);
    h.extent = cloud.extent;
    std::copy(cloud.splats.bounds.min.begin(), cloud.splats.bounds.min.end(), h.boundsMin);
    std::copy(cloud.splats.bounds.max.begin(), cloud.splats.bounds.max.end(), h.boundsMax);
    h.levelTable = kPage;
    h.chunkTable = h.levelTable + uint64_t{h.levels} * sizeof(LevelEntry);
    h.starts = aligned(h.chunkTable + uint64_t{h.chunks} * sizeof(ChunkEntry));
    uint64_t at = aligned(h.starts + uint64_t{h.finestGroups} * 4);
    std::vector<LevelEntry> levels;
    for (const LodLevel& l : cloud.levels) {
        levels.push_back({l.level, l.gaussians.count, at});
        at = aligned(at + l.gaussians.count * per);
    }
    std::vector<ChunkEntry> chunks;
    for (uint32_t c = 0; c < h.chunks; ++c) {
        chunks.push_back({at, cloud.chunkCount(c), 0});
        at = aligned(at + cloud.chunkCount(c) * per);
    }

    // Beside the destination first: a failed write never leaves a file that
    // looks whole.
    std::filesystem::path partial = path;
    partial += ".partial";
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Error(ErrorCode::IoFailure, "cannot write " + partial.string());
    }
    uint64_t written = 0;
    std::vector<char> bytes;
    const auto put = [&](const void* data, uint64_t size) {
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        written += size;
    };
    const auto padTo = [&](uint64_t offset) {
        static const char zeros[kPage] = {};
        while (written < offset) {
            put(zeros, std::min<uint64_t>(kPage, offset - written));
        }
    };
    const auto copy = [&](const gpu::Buffer& from, uint64_t offset, uint64_t size) -> Result<void> {
        bytes.resize(size);
        LRT_TRY(from.read(device, offset, size, bytes.data()));
        put(bytes.data(), size);
        return ok();
    };
    put(&h, sizeof h);
    padTo(h.levelTable);
    put(levels.data(), levels.size() * sizeof(LevelEntry));
    put(chunks.data(), chunks.size() * sizeof(ChunkEntry));
    padTo(h.starts);
    LRT_TRY(copy(cloud.starts, 0, uint64_t{h.finestGroups} * 4));
    for (size_t l = 0; l < cloud.levels.size(); ++l) {
        const LodLevel& level = cloud.levels[l];
        const uint64_t n = level.gaussians.count;
        padTo(levels[l].offset);
        LRT_TRY(copy(level.gaussians.positions, 0, n * 16));
        LRT_TRY(copy(level.gaussians.shape, 0, n * 16));
        LRT_TRY(copy(level.gaussians.sh, 0, n * 4 * shWords));
        LRT_TRY(copy(level.cells, 0, n * 4));
    }
    for (uint32_t c = 0; c < h.chunks; ++c) {
        const uint64_t n = chunks[c].count;
        const uint64_t first = static_cast<uint64_t>(cloud.slots[c]) * cloud.chunkSplats;
        padTo(chunks[c].offset);
        LRT_TRY(copy(cloud.splats.positions, first * 16, n * 16));
        LRT_TRY(copy(cloud.splats.shape, first * 16, n * 16));
        LRT_TRY(copy(cloud.splats.sh, first * 4 * shWords, n * 4 * shWords));
        LRT_TRY(copy(cloud.groups, first * 4, n * 4));
    }
    padTo(aligned(written));
    out.close();
    if (!out) {
        return Error(ErrorCode::IoFailure, "writing " + partial.string() + " failed");
    }
    std::error_code ec;
    std::filesystem::rename(partial, path, ec);
    if (ec) {
        return Error(ErrorCode::IoFailure, "cannot move " + partial.string() + " to " + path.string() + ": " +
                                             ec.message());
    }
    log::info("{}: {} splats in {} chunks, {} levels, {:.1f} MB", path.string(), h.count, h.chunks, h.levels,
              static_cast<double>(written) / (1 << 20));
    return ok();
}

Result<LodCloud> readLrtc(gpu::Device& device, const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) return std::move(file).error();
    auto layout = parse(*file, path);
    if (!layout) return std::move(layout).error();
    auto lod = openCloud(device, *file, *layout, path);
    if (!lod) return std::move(lod).error();
    const FileHeader& h = layout->header;
    LRT_TRY(makeStore(device, *lod, h, h.chunks, path));
    gpu::CommandBatch batch(device);
    uint64_t staged = 0;
    for (uint32_t c = 0; c < h.chunks; ++c) {
        place(batch, *lod, c, c, file->bytes().data() + layout->chunks[c].offset);
        staged += lod->chunkCount(c) * elementBytes(h.shWords);
        if (staged >= kStageBytes) {
            LRT_TRY(batch.submit(true));
            staged = 0;
        }
    }
    LRT_TRY(batch.submit(true));
    return std::move(*lod);
}

// --- streaming ---------------------------------------------------------------

struct StreamingPool::Impl {
    gpu::Device*          device = nullptr;
    platform::MappedFile  file;
    Layout                layout;
    LodCloud              cloud;
    std::vector<int32_t>  owner;          // per slot: its chunk, or -1
    std::vector<uint32_t> wantedNow;      // per chunk: how much this frame wants it (CutStats::needs)
    std::vector<uint64_t> lastWanted;     // per chunk: the frame it was last wanted
    std::vector<uint8_t>  inFlight;       // per chunk: queued, loading, or loaded and not placed
    uint64_t              frame = 1;
    Status                counters;

    // Shared with the loaders.
    std::mutex                                            mutex;
    std::condition_variable                               work, done;
    std::deque<uint32_t>                                  queue;
    std::vector<std::pair<uint32_t, std::vector<std::byte>>> loaded;
    uint32_t                                              busy = 0;   // loads a loader has taken
    bool                                                  stop = false;
    std::vector<std::thread>                              loaders;

    void load() {
        const uint64_t per = elementBytes(layout.header.shWords);
        std::unique_lock lock(mutex);
        for (;;) {
            work.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop) {
                return;
            }
            const uint32_t chunk = queue.front();
            queue.pop_front();
            ++busy;
            lock.unlock();
            // Copied off the mapping here, so the page faults are this
            // thread's and not the frame's.
            const ChunkEntry& e = layout.chunks[chunk];
            std::vector<std::byte> bytes(e.count * per);
            std::memcpy(bytes.data(), file.bytes().data() + e.offset, bytes.size());
            lock.lock();
            loaded.emplace_back(chunk, std::move(bytes));
            --busy;
            done.notify_all();
        }
    }
};

StreamingPool::StreamingPool() : impl_(std::make_unique<Impl>()) {}

StreamingPool::~StreamingPool() {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->work.notify_all();
    for (std::thread& t : impl_->loaders) {
        t.join();
    }
}

Result<std::unique_ptr<StreamingPool>> StreamingPool::open(gpu::Device& device, const std::filesystem::path& path,
                                                           const StreamingSettings& settings) {
    std::unique_ptr<StreamingPool> pool(new StreamingPool());
    Impl& p = *pool->impl_;
    p.device = &device;
    auto file = platform::MappedFile::open(path);
    if (!file) return std::move(file).error();
    p.file = std::move(*file);
    auto layout = parse(p.file, path);
    if (!layout) return std::move(layout).error();
    p.layout = std::move(*layout);
    auto cloud = openCloud(device, p.file, p.layout, path);
    if (!cloud) return std::move(cloud).error();
    p.cloud = std::move(*cloud);
    const FileHeader& h = p.layout.header;
    const uint32_t slots = static_cast<uint32_t>(
        std::clamp<uint64_t>(settings.budgetSplats / h.chunkSplats, 1, h.chunks));
    LRT_TRY(makeStore(device, p.cloud, h, slots, path));
    p.cloud.streamed = true;
    p.owner.assign(slots, -1);
    p.wantedNow.assign(h.chunks, 0);
    p.lastWanted.assign(h.chunks, 0);
    p.inFlight.assign(h.chunks, 0);
    p.counters.slots = slots;
    for (uint32_t k = 0; k < std::max<uint32_t>(settings.loaders, 1); ++k) {
        p.loaders.emplace_back([&p] { p.load(); });
    }
    log::info("{}: streaming {} chunks of {} splats into {} slots", path.filename().string(), h.chunks,
              h.chunkSplats, slots);
    return pool;
}

const LodCloud& StreamingPool::cloud() const noexcept {
    return impl_->cloud;
}

void StreamingPool::want(std::span<const uint32_t> needs) {
    Impl& p = *impl_;
    const size_t n = std::min(needs.size(), p.wantedNow.size());
    for (size_t c = 0; c < n; ++c) {
        p.wantedNow[c] = std::max(p.wantedNow[c], needs[c]);
    }
}

Result<uint32_t> StreamingPool::update(bool wait) {
    Impl& p = *impl_;
    LodCloud& lod = p.cloud;
    const uint32_t chunks = lod.chunks();
    const uint32_t slots = static_cast<uint32_t>(p.owner.size());
    for (uint32_t c = 0; c < chunks; ++c) {
        if (p.wantedNow[c] != 0) {
            p.lastWanted[c] = p.frame;
        }
    }
    // What a slot's place is worth keeping: a free slot least, then chunks not
    // wanted now (least recently wanted first), then by how much they are.
    const auto worth = [&](uint32_t slot) -> std::pair<uint64_t, uint64_t> {
        const int32_t chunk = p.owner[slot];
        if (chunk < 0) {
            return {0, 0};
        }
        const auto c = static_cast<uint32_t>(chunk);
        return p.wantedNow[c] == 0 ? std::pair<uint64_t, uint64_t>{1, p.lastWanted[c]}
                                   : std::pair<uint64_t, uint64_t>{2, p.wantedNow[c]};
    };
    // A chunk wanted `want` may take a slot worth `w`.
    const auto mayTake = [](uint32_t want, std::pair<uint64_t, uint64_t> w) {
        return w.first < 2 ? (w.first == 0 || want > 0) : uint64_t{want} > 2 * w.second;
    };
    std::vector<uint32_t> bySlotWorth(slots);
    const auto sortSlots = [&] {
        for (uint32_t k = 0; k < slots; ++k) {
            bySlotWorth[k] = k;
        }
        std::sort(bySlotWorth.begin(), bySlotWorth.end(), [&](uint32_t a, uint32_t b) { return worth(a) < worth(b); });
    };

    std::vector<std::pair<uint32_t, std::vector<std::byte>>> arrived;
    {
        std::unique_lock lock(p.mutex);
        // Queued chunks no longer wanted are not loaded after all.
        for (auto it = p.queue.begin(); it != p.queue.end();) {
            if (p.wantedNow[*it] == 0) {
                p.inFlight[*it] = 0;
                it = p.queue.erase(it);
            } else {
                ++it;
            }
        }
        // Missing chunks, most wanted first, each against the next cheapest
        // slot not already promised to a load in flight.
        std::vector<uint32_t> missing;
        for (uint32_t c = 0; c < chunks; ++c) {
            if (p.wantedNow[c] != 0 && lod.slots[c] < 0 && p.inFlight[c] == 0) {
                missing.push_back(c);
            }
        }
        std::sort(missing.begin(), missing.end(), [&](uint32_t a, uint32_t b) {
            return p.wantedNow[a] != p.wantedNow[b] ? p.wantedNow[a] > p.wantedNow[b] : a < b;
        });
        sortSlots();
        size_t next = p.queue.size() + p.busy + p.loaded.size();
        for (const uint32_t c : missing) {
            if (next >= slots || !mayTake(p.wantedNow[c], worth(bySlotWorth[next]))) {
                break;
            }
            p.inFlight[c] = 1;
            p.queue.push_back(c);
            ++next;
        }
        std::stable_sort(p.queue.begin(), p.queue.end(),
                         [&](uint32_t a, uint32_t b) { return p.wantedNow[a] > p.wantedNow[b]; });
        p.work.notify_all();
        if (wait) {
            p.done.wait(lock, [&] { return p.queue.empty() && p.busy == 0; });
        }
        arrived.swap(p.loaded);
    }

    // Placed most wanted first, each in the cheapest slot it may take.
    std::sort(arrived.begin(), arrived.end(),
              [&](const auto& a, const auto& b) { return p.wantedNow[a.first] > p.wantedNow[b.first]; });
    uint32_t placed = 0;
    gpu::CommandBatch batch(*p.device);
    for (auto& [chunk, bytes] : arrived) {
        p.inFlight[chunk] = 0;
        if (lod.slots[chunk] >= 0) {
            continue;
        }
        uint32_t slot = 0;
        for (uint32_t k = 1; k < slots; ++k) {
            slot = worth(k) < worth(slot) ? k : slot;
        }
        const auto w = worth(slot);
        if (!(w.first == 0 || mayTake(p.wantedNow[chunk], w))) {
            continue;   // not worth a place now: asked for again when it is
        }
        const int32_t evicted = p.owner[slot];
        if (evicted >= 0) {
            lod.slots[static_cast<uint32_t>(evicted)] = -1;
            const uint32_t zero = 0;
            batch.encoder()->uploadBufferData(lod.resident.rhi(), uint64_t{static_cast<uint32_t>(evicted)} * 4, 4,
                                              &zero);
            ++p.counters.evictions;
        }
        place(batch, lod, chunk, slot, bytes.data());
        p.owner[slot] = static_cast<int32_t>(chunk);
        ++placed;
    }
    LRT_TRY(batch.submit(true));
    p.counters.loads += placed;

    uint32_t resident = 0;
    uint32_t missing = 0;
    for (uint32_t c = 0; c < chunks; ++c) {
        resident += lod.slots[c] >= 0 ? 1 : 0;
        missing += (p.wantedNow[c] != 0 && lod.slots[c] < 0) ? 1 : 0;
    }
    p.counters.resident = resident;
    p.counters.missing = missing;
    std::fill(p.wantedNow.begin(), p.wantedNow.end(), 0);
    ++p.frame;
    return placed;
}

StreamingPool::Status StreamingPool::status() const {
    Impl& p = *impl_;
    Status s = p.counters;
    std::lock_guard lock(p.mutex);
    s.inFlight = static_cast<uint32_t>(p.queue.size() + p.busy + p.loaded.size());
    return s;
}

}   // namespace lrt::lod
