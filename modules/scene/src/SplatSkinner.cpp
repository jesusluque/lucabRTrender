// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/scene/SplatSkinner.h"

#include <array>

#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/scene/GpuClouds.h"

namespace lrt::scene {
namespace {

// GfMatrix4f is row-major with the translation in the last row and vectors on
// the left (v * M); the kernel multiplies M * v with rows, so the matrix goes
// over transposed -- the same turn `geom::Skinner` makes for a mesh, and for
// the same reason.
std::array<float, 16> transposed(const std::array<float, 16>& m) {
    std::array<float, 16> t{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            t[static_cast<size_t>(r * 4 + c)] = m[static_cast<size_t>(c * 4 + r)];
        }
    }
    return t;
}

void setRows(rhi::ShaderCursor cursor, const char* name, const std::array<float, 16>& m) {
    const std::array<float, 16> t = transposed(m);
    for (int r = 0; r < 4; ++r) {
        const std::string field = std::string(name) + std::to_string(r);
        cursor[field.c_str()].setData(t.data() + r * 4, 16);
    }
}

}   // namespace

Result<SplatSkinner> SplatSkinner::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "lrt/scene/splat_skin", "splatSkin");
    if (!kernel) return std::move(kernel).error();
    SplatSkinner made;
    made.kernel_ = std::move(*kernel);
    return made;
}

Result<void> SplatSkinner::skin(gpu::CommandBatch& batch, const SplatSkinInput& input,
                                gpu::Buffer& positions, gpu::Buffer& shape) {
    if (input.rest == nullptr || input.influences == nullptr || input.skinningXforms == nullptr) {
        return Error(ErrorCode::InvalidArgument, "a skinned cloud wants a rest cloud, its joints and their transforms");
    }
    const uint32_t count = input.rest->count;
    if (count == 0) {
        return ok();
    }
    kernel_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["restPositions"].setBinding(input.rest->positions.rhi());
        cursor["restShape"].setBinding(input.rest->shape.rhi());
        cursor["influences"].setBinding(input.influences->rhi());
        cursor["skinningXforms"].setBinding(input.skinningXforms->rhi());
        cursor["positions"].setBinding(positions.rhi());
        cursor["shape"].setBinding(shape.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["count"].setData(count);
        p["perSplat"].setData(std::max(input.perSplat, 1U));
        setRows(p, "geomBind", input.geomBindTransform);
        setRows(p, "skelToWorld", input.skelLocalToWorld);
        setRows(p, "worldToPrim", input.primWorldToLocal);
    });
    return ok();
}

}   // namespace lrt::scene
