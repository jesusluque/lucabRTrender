// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/SplatShadows.h"

#include "lrt/core/Log.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {
namespace {

/// The packed layout: the four tables end to end, in float4 entries, each
/// starting where the one before it ended. The order is the order the query
/// reads them in, which is nobody's business but this file's and the
/// shader's.
struct Packing {
    PackedShadowLayout layout;
    uint64_t           particles = 0;
    uint64_t           colourEntries = 0;
    uint64_t           total = 0;
};

[[nodiscard]] Packing packingFor(const render::ShadowScene& scene) {
    Packing p;
    p.particles = scene.frames != nullptr ? scene.frames->count() / 4 : 0;
    p.colourEntries = scene.colours != nullptr ? scene.colours->count() : 0;
    p.layout.instanceCount = scene.instances;
    p.layout.frames = 0;
    p.layout.colours = static_cast<uint32_t>(p.particles * 4);
    p.layout.instances = static_cast<uint32_t>(p.layout.colours + p.colourEntries);
    p.layout.indices = static_cast<uint32_t>(p.layout.instances + uint64_t{scene.instances} * 3);
    p.total = uint64_t{p.layout.indices} + scene.instances;
    return p;
}

}   // namespace

Result<SplatShadows> SplatShadows::create(gpu::ShaderLibrary& library) {
    SplatShadows shadows;
    shadows.device_ = &library.device();
    auto pack = gpu::ComputeKernel::create(library, "lrt/rt/rt_shadow_pack", "rtShadowPack");
    if (!pack) return std::move(pack).error();
    shadows.pack_.emplace(std::move(*pack));
    return shadows;
}

Result<void> SplatShadows::prepare(gpu::CommandBatch& batch, const render::ShadowScene& scene) {
    layout_ = {};
    tlas_ = nullptr;
    if (scene.tlas == nullptr || scene.instances == 0 || scene.frames == nullptr || scene.colours == nullptr ||
        scene.instanceData == nullptr || scene.instanceIndices == nullptr || !scene.frames->valid()) {
        return ok();
    }
    const Packing packing = packingFor(scene);
    if (packing.total == 0) {
        return ok();
    }
    if (!packed_.valid() || packed_.count() < packing.total) {
        gpu::BufferDesc desc;
        desc.bytes = packing.total * 16;
        desc.elementBytes = 16;
        desc.label = "shadow.packed";
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        packed_ = std::move(*made);
    }
    const uint32_t total = static_cast<uint32_t>(packing.total);
    // Threads, not groups: ComputeKernel::dispatch divides by the kernel's own
    // size, so groups passed here are divided again. A cloud of 741872
    // particles wants 3.7 million entries packed and got 14592 -- the frames,
    // and not one colour -- so every shadow ray came back saying the cloud
    // was transparent. A small cloud hid it: one group is 256 threads, which
    // covers everything a handful of particles needs.
    pack_->dispatch(batch, {total, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["frames"].setBinding(scene.frames->rhi());
        cursor["colours"].setBinding(scene.colours->rhi());
        cursor["instanceData"].setBinding(scene.instanceData->rhi());
        cursor["instanceIndices"].setBinding(scene.instanceIndices->rhi());
        cursor["packed"].setBinding(packed_.rhi());
        cursor["pack"]["where"]["frames"].setData(packing.layout.frames);
        cursor["pack"]["where"]["colours"].setData(packing.layout.colours);
        cursor["pack"]["where"]["instances"].setData(packing.layout.instances);
        cursor["pack"]["where"]["indices"].setData(packing.layout.indices);
        cursor["pack"]["where"]["instanceCount"].setData(packing.layout.instanceCount);
        cursor["pack"]["particles"].setData(static_cast<uint32_t>(packing.particles));
        cursor["pack"]["colourEntries"].setData(static_cast<uint32_t>(packing.colourEntries));
        cursor["pack"]["total"].setData(total);
    });
    lrt::log::info("pack: instances {} particles {} colours {} total {}", scene.instances, packing.particles,
                   packing.colourEntries, packing.total);
    layout_ = packing.layout;
    tlas_ = scene.tlas;
    return ok();
}

}   // namespace lrt::technique
