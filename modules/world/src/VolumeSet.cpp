// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/VolumeSet.h"

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
    set.boundsKernel_ = std::move(*bounds);
    set.leafMaxKernel_ = std::move(*leafMax);
    return set;
}

Result<void> VolumeSet::set(gpu::CommandBatch& batch, std::span<const io::NanoGrid> grids) {
    placements_.clear();
    std::vector<uint32_t> words;
    uint32_t leaves = 0;
    for (const io::NanoGrid& grid : grids) {
        // NanoVDB wants its grids 32-byte aligned: eight words.
        while (words.size() % 8 != 0) words.push_back(0u);
        VolumeGridPlacement placement;
        placement.gridWord = static_cast<uint32_t>(words.size());
        placement.leafCount = grid.leafCount;
        placement.leafBase = leaves;
        placements_.push_back(placement);
        words.insert(words.end(), grid.words.begin(), grid.words.end());
        leaves += grid.leafCount;
    }
    if (words.empty()) words.push_back(0u);
    auto madeWords = gpu::Buffer::fromSpan<uint32_t>(*device_, words, "volumes.words");
    if (!madeWords) return std::move(madeWords).error();
    words_ = std::move(*madeWords);
    // Bounds start from the empty box, so the first leaf sets them.
    std::vector<int32_t> bounds;
    for (size_t g = 0; g < std::max<size_t>(placements_.size(), 1); ++g) {
        for (int k = 0; k < 3; ++k) bounds.push_back(std::numeric_limits<int32_t>::max());
        for (int k = 0; k < 3; ++k) bounds.push_back(std::numeric_limits<int32_t>::min());
    }
    auto madeBounds = gpu::Buffer::fromSpan<int32_t>(*device_, bounds, "volumes.bounds");
    if (!madeBounds) return std::move(madeBounds).error();
    bounds_ = std::move(*madeBounds);
    const std::vector<float> zeros(std::max<uint32_t>(leaves, 1), 0.0F);
    auto madeLeafMax = gpu::Buffer::fromSpan<float>(*device_, zeros, "volumes.leafMax");
    if (!madeLeafMax) return std::move(madeLeafMax).error();
    leafMax_ = std::move(*madeLeafMax);
    const std::vector<uint32_t> zeroBits(std::max<size_t>(placements_.size(), 1), 0u);
    auto madeMajorant = gpu::Buffer::fromSpan<uint32_t>(*device_, zeroBits, "volumes.majorant");
    if (!madeMajorant) return std::move(madeMajorant).error();
    majorant_ = std::move(*madeMajorant);
    for (size_t g = 0; g < placements_.size(); ++g) {
        const VolumeGridPlacement& p = placements_[g];
        if (p.leafCount == 0) continue;
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(words_.rhi());
            cursor["bounds"].setBinding(bounds_.rhi());
            cursor["leafMax"].setBinding(leafMax_.rhi());
            cursor["majorant"].setBinding(majorant_.rhi());
            cursor["params"]["gridWord"].setData(p.gridWord);
            cursor["params"]["leafCount"].setData(p.leafCount);
            cursor["params"]["leafBase"].setData(p.leafBase);
            cursor["params"]["slot"].setData(static_cast<uint32_t>(g));
        };
        boundsKernel_.dispatch(batch, {p.leafCount, 1, 1}, bind);
        leafMaxKernel_.dispatch(batch, {p.leafCount, 1, 1}, bind);
    }
    return ok();
}

}   // namespace lrt::world
