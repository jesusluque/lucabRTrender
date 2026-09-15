// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/VolumeSet.h"

#include <cstring>
#include <limits>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::world {

Result<VolumeSet> VolumeSet::create(gpu::ShaderLibrary& library) {
    VolumeSet set;
    set.device_ = &library.device();
    auto bounds = gpu::ComputeKernel::create(library, "lrt/volume/volume_prepare", "volumeBounds");
    if (!bounds) return std::move(bounds).error();
    auto leafMax = gpu::ComputeKernel::create(library, "lrt/volume/volume_prepare", "volumeLeafMax");
    if (!leafMax) return std::move(leafMax).error();
    auto finish = gpu::ComputeKernel::create(library, "lrt/volume/volume_prepare", "volumeFinish");
    if (!finish) return std::move(finish).error();
    set.boundsKernel_ = std::move(*bounds);
    set.leafMaxKernel_ = std::move(*leafMax);
    set.finishKernel_ = std::move(*finish);
    return set;
}

namespace {

uint32_t bitsOf(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    return bits;
}

}   // namespace

Result<void> VolumeSet::set(gpu::CommandBatch& batch, std::span<const VolumeInput> volumes) {
    placements_.clear();
    const uint32_t count = static_cast<uint32_t>(volumes.size());
    std::vector<uint32_t> words(kHeaderWords + count * kRecordWords, 0u);
    words[0] = count;
    // The grids, each 32-byte aligned (eight words), then the leaf maxima.
    uint32_t leaves = 0;
    for (uint32_t v = 0; v < count; ++v) {
        const VolumeInput& in = volumes[v];
        if (in.grid == nullptr) {
            return Error(ErrorCode::InvalidArgument, "volumes: a volume with no grid");
        }
        while (words.size() % 8 != 0) words.push_back(0u);
        VolumeGridPlacement placement;
        placement.gridWord = static_cast<uint32_t>(words.size());
        placement.leafCount = in.grid->leafCount;
        placements_.push_back(placement);
        words.insert(words.end(), in.grid->words.begin(), in.grid->words.end());
        leaves += in.grid->leafCount;
    }
    for (uint32_t v = 0; v < count; ++v) {
        placements_[v].leafBase = static_cast<uint32_t>(words.size());
        words.resize(words.size() + std::max<size_t>(placements_[v].leafCount, 1), 0u);
    }
    // The records: what the host knows. World to index is the inverse of
    // index to object (the grid's own map) then object to world (the
    // prim's). OpenVDB holds its map row-vector, translation in the last row;
    // Mat4 is column-vector, applied first rightmost -- so the map is
    // transposed as it is read, and composed on the right. Read untransposed
    // and composed on the left, a translated volume sat offset by its
    // translation in voxels rather than in world units, which a pure scale
    // (the first fixture) could not show.
    for (uint32_t v = 0; v < count; ++v) {
        const VolumeInput& in = volumes[v];
        render::Mat4 indexToObject = render::Mat4::identity();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                indexToObject.at(r, c) = in.grid->indexToWorld[static_cast<size_t>(c * 4 + r)];
            }
        }
        const render::Mat4 worldToIndex = aofx::xform::inverseAffine(in.objectToWorld * indexToObject);
        const std::array<float, 12> rows = worldToIndex.rows3x4();
        const uint32_t b = kHeaderWords + v * kRecordWords;
        words[b + 0] = placements_[v].gridWord;
        words[b + 1] = placements_[v].leafBase;
        words[b + 2] = placements_[v].leafCount;
        words[b + 3] = 0;   // leaf 0's word: the finish kernel's
        for (size_t k = 0; k < 12; ++k) words[b + 4 + k] = bitsOf(rows[k]);
        words[b + 22] = bitsOf(in.densityScale);
        words[b + 23] = bitsOf(in.albedo[0]);
        words[b + 24] = bitsOf(in.albedo[1]);
        words[b + 25] = bitsOf(in.albedo[2]);
        words[b + 26] = bitsOf(in.g);
    }
    auto madeWords = gpu::Buffer::fromSpan<uint32_t>(*device_, words, "volumes.words");
    if (!madeWords) return std::move(madeWords).error();
    words_ = std::move(*madeWords);
    // Bounds start from the empty box, so the first leaf sets them.
    std::vector<int32_t> bounds;
    for (size_t g = 0; g < std::max<size_t>(count, 1); ++g) {
        for (int k = 0; k < 3; ++k) bounds.push_back(std::numeric_limits<int32_t>::max());
        for (int k = 0; k < 3; ++k) bounds.push_back(std::numeric_limits<int32_t>::min());
    }
    auto madeBounds = gpu::Buffer::fromSpan<int32_t>(*device_, bounds, "volumes.bounds");
    if (!madeBounds) return std::move(madeBounds).error();
    bounds_ = std::move(*madeBounds);
    const std::vector<uint32_t> zeroBits(std::max<size_t>(count, 1), 0u);
    auto madeMajorant = gpu::Buffer::fromSpan<uint32_t>(*device_, zeroBits, "volumes.majorant");
    if (!madeMajorant) return std::move(madeMajorant).error();
    majorant_ = std::move(*madeMajorant);
    for (uint32_t v = 0; v < count; ++v) {
        const VolumeGridPlacement& p = placements_[v];
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(words_.rhi());
            cursor["wordsOut"].setBinding(words_.rhi());
            cursor["bounds"].setBinding(bounds_.rhi());
            cursor["majorant"].setBinding(majorant_.rhi());
            cursor["params"]["gridWord"].setData(p.gridWord);
            cursor["params"]["leafCount"].setData(p.leafCount);
            cursor["params"]["leafBase"].setData(p.leafBase);
            cursor["params"]["slot"].setData(v);
        };
        if (p.leafCount > 0) {
            boundsKernel_.dispatch(batch, {p.leafCount, 1, 1}, bind);
            leafMaxKernel_.dispatch(batch, {p.leafCount, 1, 1}, bind);
        }
        finishKernel_.dispatch(batch, {1, 1, 1}, bind);
    }
    return ok();
}

}   // namespace lrt::world
