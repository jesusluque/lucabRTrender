// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/geom/Skinner.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::geom {
namespace {

Result<gpu::Buffer> upload(gpu::Device& device, std::span<const std::byte> bytes, uint32_t element,
                           const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(bytes.size(), element);
    desc.elementBytes = element;
    desc.label = label;
    auto made = gpu::Buffer::create(device, desc);
    if (!made) return std::move(made).error();
    if (!bytes.empty()) {
        LRT_TRY(made->write(device, 0, bytes.size(), bytes.data()));
    }
    return made;
}

// GfMatrix4f is row-major with the translation in the last row and vectors
// on the left (v * M); the kernel multiplies M * v with rows, so the matrix
// goes over transposed.
std::array<float, 16> transposed(const std::array<float, 16>& m) {
    std::array<float, 16> t{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            t[static_cast<size_t>(r * 4 + c)] = m[static_cast<size_t>(c * 4 + r)];
        }
    }
    return t;
}

std::vector<float> transposedJoints(std::span<const std::byte> xforms) {
    const size_t count = xforms.size() / (16 * sizeof(float));
    std::vector<float> out(count * 16);
    for (size_t j = 0; j < count; ++j) {
        std::array<float, 16> m{};
        std::memcpy(m.data(), xforms.data() + j * 16 * sizeof(float), 16 * sizeof(float));
        const std::array<float, 16> t = transposed(m);
        std::copy(t.begin(), t.end(), out.begin() + static_cast<std::ptrdiff_t>(j * 16));
    }
    return out;
}

std::vector<float> paddedScales(std::span<const std::byte> scales) {
    const size_t count = scales.size() / (9 * sizeof(float));
    std::vector<float> out(count * 12, 0.0F);
    for (size_t j = 0; j < count; ++j) {
        std::array<float, 9> m{};
        std::memcpy(m.data(), scales.data() + j * 9 * sizeof(float), 9 * sizeof(float));
        // Row-major 3x3 with vectors on the left: transposed, padded to float4 rows.
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out[j * 12 + static_cast<size_t>(r * 4 + c)] = m[static_cast<size_t>(c * 3 + r)];
            }
        }
    }
    return out;
}

}   // namespace

Result<Skinner> Skinner::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "lrt/geom/skinning", "skinPoints");
    if (!kernel) return std::move(kernel).error();
    auto points = gpu::ComputeKernel::create(library, "lrt/geom/mesh_topology", "meshPoints");
    if (!points) return std::move(points).error();
    Skinner s;
    s.device_ = &library.device();
    s.kernel_ = std::move(*kernel);
    s.points_ = std::move(*points);
    return s;
}

Result<gpu::Buffer> Skinner::skin(const SkinningInput& in) {
    gpu::Device& device = *device_;
    if (in.points == 0) {
        return Error(ErrorCode::InvalidArgument, "skinning: no points");
    }
    // The rest points as float4, by the builder's own decode.
    auto restWords = upload(device, in.restPoints, 4, "skin.rest.words");
    if (!restWords) return std::move(restWords).error();
    gpu::BufferDesc restDesc;
    restDesc.bytes = uint64_t{in.points} * 16;
    restDesc.elementBytes = 16;
    restDesc.label = "skin.rest";
    auto restBuffer = gpu::Buffer::create(device, restDesc);
    if (!restBuffer) return std::move(restBuffer).error();
    {
        gpu::CommandBatch batch(device);
        points_.dispatch(batch, {in.points, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(restWords->rhi());
            cursor["positions"].setBinding(restBuffer->rhi());
            cursor["params"]["count"].setData(in.points);
            cursor["params"]["half"].setData(uint32_t{0});
        });
        LRT_TRY(batch.submit(true));
    }
    auto offsets = upload(device, in.blendShapeOffsets, 16, "skin.offsets");
    if (!offsets) return std::move(offsets).error();
    auto ranges = upload(device, in.blendShapeOffsetRanges, 8, "skin.ranges");
    if (!ranges) return std::move(ranges).error();
    auto weights = upload(device, in.blendShapeWeights, 4, "skin.weights");
    if (!weights) return std::move(weights).error();
    auto influences = upload(device, in.influences, 8, "skin.influences");
    if (!influences) return std::move(influences).error();
    const std::vector<float> joints = transposedJoints(in.skinningXforms);
    auto xforms = upload(device, std::as_bytes(std::span<const float>(joints)), 16, "skin.xforms");
    if (!xforms) return std::move(xforms).error();
    auto dualQuats = upload(device, in.skinningDualQuats, 16, "skin.dualQuats");
    if (!dualQuats) return std::move(dualQuats).error();
    const std::vector<float> scales = paddedScales(in.skinningScaleXforms);
    auto scaleXforms = upload(device, std::as_bytes(std::span<const float>(scales)), 16, "skin.scales");
    if (!scaleXforms) return std::move(scaleXforms).error();
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{in.points} * 16;
    desc.elementBytes = 16;
    desc.label = "skin.positions";
    auto out = gpu::Buffer::create(device, desc);
    if (!out) return std::move(out).error();

    const uint32_t ranged = static_cast<uint32_t>(in.blendShapeOffsetRanges.size() / 8);
    const std::array<float, 16> geomBind = transposed(in.geomBindXform);
    const std::array<float, 16> skelToWorld = transposed(in.skelLocalToWorld);
    const std::array<float, 16> worldToPrim = transposed(in.primWorldToLocal);
    gpu::CommandBatch batch(device);
    kernel_.dispatch(batch, {in.points, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["restPoints"].setBinding(restBuffer->rhi());
        cursor["blendShapeOffsets"].setBinding(offsets->rhi());
        cursor["blendShapeOffsetRanges"].setBinding(ranges->rhi());
        cursor["blendShapeWeights"].setBinding(weights->rhi());
        cursor["influences"].setBinding(influences->rhi());
        cursor["skinningXforms"].setBinding(xforms->rhi());
        cursor["skinningDualQuats"].setBinding(dualQuats->rhi());
        cursor["skinningScaleXforms"].setBinding(scaleXforms->rhi());
        cursor["skinned"].setBinding(out->rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["count"].setData(in.points);
        p["numInfluences"].setData(in.numInfluencesPerPoint);
        p["constantInfluences"].setData(uint32_t{in.constantInfluences ? 1u : 0u});
        p["numOffsetRanges"].setData(ranged);
        p["method"].setData(static_cast<uint32_t>(in.method));
        p["hasScaleXforms"].setData(uint32_t{in.skinningScaleXforms.empty() ? 0u : 1u});
        static constexpr const char* kRows[3][4] = {{"geomBind0", "geomBind1", "geomBind2", "geomBind3"},
                                                    {"skelToWorld0", "skelToWorld1", "skelToWorld2", "skelToWorld3"},
                                                    {"worldToPrim0", "worldToPrim1", "worldToPrim2", "worldToPrim3"}};
        const std::array<float, 16>* matrices[3] = {&geomBind, &skelToWorld, &worldToPrim};
        for (size_t m = 0; m < 3; ++m) {
            for (size_t r = 0; r < 4; ++r) {
                p[kRows[m][r]].setData(matrices[m]->data() + r * 4, sizeof(float) * 4);
            }
        }
    });
    LRT_TRY(batch.submit(true));
    return out;
}

}   // namespace lrt::geom
