// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/world/Instancing.h"

#include <algorithm>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::world {
namespace {

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

Result<gpu::Buffer> bytesBuffer(gpu::Device& device, std::span<const std::byte> bytes, const char* label) {
    auto made = deviceBuffer(device, std::max<uint64_t>((bytes.size() + 3) / 4, 1), 4, label);
    if (!made) return std::move(made).error();
    if (!bytes.empty()) {
        LRT_TRY(made->write(device, 0, bytes.size(), bytes.data()));
    }
    return made;
}

void setRows(rhi::ShaderCursor p, const char* prefix, const render::Mat4& m) {
    const std::array<float, 12> r = m.rows3x4();
    static constexpr const char* kSuffix[12] = {"00", "01", "02", "03", "10", "11", "12", "13", "20", "21", "22", "23"};
    for (size_t k = 0; k < 12; ++k) {
        p[(std::string(prefix) + kSuffix[k]).c_str()].setData(r[k]);
    }
}

}   // namespace

Result<Instancing> Instancing::create(gpu::ShaderLibrary& library) {
    Instancing instancing;
    instancing.device_ = &library.device();
    auto level = gpu::ComputeKernel::create(library, "lrt/world/instancing", "instancerLevel");
    if (!level) return std::move(level).error();
    auto compose = gpu::ComputeKernel::create(library, "lrt/world/instancing", "instancerCompose");
    if (!compose) return std::move(compose).error();
    instancing.level_ = std::move(*level);
    instancing.compose_ = std::move(*compose);
    return instancing;
}

Result<InstanceChain> Instancing::level(const InstancerLevel& in) {
    gpu::Device& device = *device_;
    InstanceChain chain;
    chain.count = static_cast<uint32_t>(in.indices.size());
    auto indices = bytesBuffer(device, std::as_bytes(in.indices), "instancer.indices");
    if (!indices) return std::move(indices).error();
    auto translations = bytesBuffer(device, in.translations.bytes, "instancer.translations");
    if (!translations) return std::move(translations).error();
    auto rotations = bytesBuffer(device, in.rotations.bytes, "instancer.rotations");
    if (!rotations) return std::move(rotations).error();
    auto scales = bytesBuffer(device, in.scales.bytes, "instancer.scales");
    if (!scales) return std::move(scales).error();
    auto transforms = bytesBuffer(device, in.transforms.bytes, "instancer.transforms");
    if (!transforms) return std::move(transforms).error();
    auto rows = deviceBuffer(device, uint64_t{chain.count} * 3, 16, "instancer.rows");
    if (!rows) return std::move(rows).error();
    chain.rows = std::move(*rows);
    if (chain.count == 0) {
        return chain;
    }
    gpu::CommandBatch batch(device);
    level_.dispatch(batch, {chain.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["indices"].setBinding(indices->rhi());
        cursor["translationWords"].setBinding(translations->rhi());
        cursor["rotationWords"].setBinding(rotations->rhi());
        cursor["scaleWords"].setBinding(scales->rhi());
        cursor["transformWords"].setBinding(transforms->rhi());
        cursor["rows"].setBinding(chain.rows.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["count"].setData(chain.count);
        p["translations"].setData(in.translations.kind());
        p["rotations"].setData(in.rotations.kind());
        p["scales"].setData(in.scales.kind());
        p["transforms"].setData(in.transforms.kind());
        setRows(p, "m", in.instancerTransform);
    });
    LRT_TRY(batch.submit(true));
    return chain;
}

Result<InstanceChain> Instancing::compose(std::span<const InstancerLevel> levels) {
    if (levels.empty()) {
        return Error(ErrorCode::InvalidArgument, "an instance chain needs an instancer");
    }
    auto inner = level(levels.front());
    if (!inner) return std::move(inner).error();
    InstanceChain chain = std::move(*inner);
    for (size_t k = 1; k < levels.size(); ++k) {
        auto parent = level(levels[k]);
        if (!parent) return std::move(parent).error();
        const uint64_t count = uint64_t{parent->count} * chain.count;
        if (count > UINT32_MAX) {
            return Error(ErrorCode::OutOfMemory, "more than 2^32 instances in one chain");
        }
        InstanceChain composed;
        composed.count = static_cast<uint32_t>(count);
        auto rows = deviceBuffer(*device_, count * 3, 16, "instancer.composed");
        if (!rows) return std::move(rows).error();
        composed.rows = std::move(*rows);
        if (count > 0) {
            gpu::CommandBatch batch(*device_);
            compose_.dispatch(batch, {composed.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["parentRows"].setBinding(parent->rows.rhi());
                cursor["levelRows"].setBinding(chain.rows.rhi());
                cursor["rows"].setBinding(composed.rows.rhi());
                cursor["params"]["count"].setData(composed.count);
                cursor["params"]["inner"].setData(chain.count);
            });
            LRT_TRY(batch.submit(true));
        }
        chain = std::move(composed);
    }
    return chain;
}

}   // namespace lrt::world
