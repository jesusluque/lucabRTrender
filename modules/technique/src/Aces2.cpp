// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/Aces2.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::technique {

namespace {
constexpr uint32_t kTableSize = 360;
constexpr uint32_t kTableWords = 5 * (kTableSize + 2);   // aces2.slang's kTableWords
// Aces2Params is three JMhParams, a TSParams and a dozen scalars: well under this.
constexpr uint64_t kParamsBytes = 4096;
}   // namespace

Result<Aces2Tables> Aces2Tables::create(gpu::ShaderLibrary& library) {
    Aces2Tables made;
    made.device_ = &library.device();
    auto params = gpu::ComputeKernel::create(library, "lrt/technique/aces2_prepare", "aces2Params");
    if (!params) return std::move(params).error();
    auto tables = gpu::ComputeKernel::create(library, "lrt/technique/aces2_prepare", "aces2Tables");
    if (!tables) return std::move(tables).error();
    auto wrap = gpu::ComputeKernel::create(library, "lrt/technique/aces2_prepare", "aces2Wrap");
    if (!wrap) return std::move(wrap).error();
    made.paramsKernel_ = std::move(*params);
    made.tablesKernel_ = std::move(*tables);
    made.wrapKernel_ = std::move(*wrap);
    gpu::BufferDesc desc;
    desc.bytes = kParamsBytes;
    desc.elementBytes = kParamsBytes;
    desc.label = "aces2.params";
    auto paramsBuffer = gpu::Buffer::create(*made.device_, desc);
    if (!paramsBuffer) return std::move(paramsBuffer).error();
    desc.bytes = uint64_t{kTableWords} * 4;
    desc.elementBytes = 4;
    desc.label = "aces2.tables";
    auto tablesBuffer = gpu::Buffer::create(*made.device_, desc);
    if (!tablesBuffer) return std::move(tablesBuffer).error();
    made.params_ = std::move(*paramsBuffer);
    made.tables_ = std::move(*tablesBuffer);
    return made;
}

Result<void> Aces2Tables::prepare(gpu::CommandBatch& batch, float peakLuminance, Aces2Limiting limiting) {
    if (peakLuminance <= 0.0F) {
        return Error(ErrorCode::InvalidArgument, "ACES 2.0: the peak luminance must be positive");
    }
    if (ready_ && peakLuminance == peakLuminance_ && limiting == limiting_) {
        return ok();
    }
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["prepare"]["peakLuminance"].setData(peakLuminance);
        cursor["prepare"]["limiting"].setData(static_cast<uint32_t>(limiting));
        cursor["params"].setBinding(params_.rhi());
        cursor["tables"].setBinding(tables_.rhi());
    };
    paramsKernel_.dispatch(batch, {1, 1, 1}, bind);
    tablesKernel_.dispatch(batch, {kTableSize, 1, 1}, bind);
    wrapKernel_.dispatch(batch, {1, 1, 1}, bind);
    ready_ = true;
    peakLuminance_ = peakLuminance;
    limiting_ = limiting;
    return ok();
}

}   // namespace lrt::technique
