// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/technique/EmissiveTable.h"

#include <bit>
#include <vector>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/world/GpuScene.h"

namespace lrt::technique {

namespace {

const char* kKernels = R"(
RWStructuredBuffer<float> emissive;
uniform uint              rowCount;
uniform uint              recordCount;
uniform uint              triangleCount;

static const uint kHeader = 4;

uint recordBase(uint i) {
    return asuint(emissive[kHeader + rowCount + i]);
}

uint cumulativeAt() {
    return kHeader + rowCount + recordCount + 1;
}

/// Each row's emission, evaluated once at a neutral point: one material call
/// site, a loop over the rows.
[shader("compute")]
[numthreads(1, 1, 1)]
void probeRows(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    for (uint r = 0; r < rowCount; ++r) {
        MaterialInputs inputs;
        inputs.positionWorld = float3(0.0);
        inputs.normalWorld = float3(0.0, 0.0, 1.0);
        inputs.tangentWorld = float3(1.0, 0.0, 0.0);
        inputs.bitangentWorld = float3(0.0, 1.0, 0.0);
        inputs.positionObject = float3(0.0);
        inputs.normalObject = float3(0.0, 0.0, 1.0);
        inputs.tangentObject = float3(1.0, 0.0, 0.0);
        inputs.bitangentObject = float3(0.0, 1.0, 0.0);
        inputs.viewPosition = float3(0.0, 0.0, 1.0);
        inputs.uvDx = float2(0.0);
        inputs.uvDy = float2(0.0);
        inputs.frame = 0.0;
        inputs.time = lookup.time;
        inputs.inside = false;
        inputs.mesh = meshes[0];
        inputs.triangle = 0;
        inputs.points = uint3(0);
        inputs.weights = float3(1.0, 0.0, 0.0);
        inputs.displayColor = float4(0.0);
        inputs.worldFromObject0 = float4(1.0, 0.0, 0.0, 0.0);
        inputs.worldFromObject1 = float4(0.0, 1.0, 0.0, 0.0);
        inputs.worldFromObject2 = float4(0.0, 0.0, 1.0, 0.0);
        const MaterialRecord m = materials[r];
        evaluateMaterial(m.function, inputs, m.blob);
        emissive[kHeader + r] = max(dot(gLrtResult.emission, float3(0.2126, 0.7152, 0.0722)), 0.0);
    }
}

/// Triangle g's power: its world area times its row's luminance.
[shader("compute")]
[numthreads(256, 1, 1)]
void trianglePower(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= triangleCount) {
        return;
    }
    uint lo = 0;
    uint hi = recordCount;
    while (hi - lo > 1) {
        const uint mid = (lo + hi) / 2;
        if (recordBase(mid) <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Surface s;
    s.instanceIndex = lo;
    s.instance = instances[lo];
    s.mesh = meshes[s.instance.mesh];
    s.triangle = g - recordBase(lo);
    float power = 0.0;
    const uint face = triangleSubsets[s.mesh.firstTriangle + s.triangle];
    if ((face & 0x80000000u) == 0) {
        const float luminance = emissive[kHeader + materialRowOf(s)];
        if (luminance > 0.0) {
            const uint t = s.mesh.firstTriangle + s.triangle;
            const InstanceRecord r = s.instance;
            const float3 a = rowsApply(r.world0, r.world1, r.world2, positions[s.mesh.firstPoint + indices[t * 3]].xyz, 1.0);
            const float3 b = rowsApply(r.world0, r.world1, r.world2, positions[s.mesh.firstPoint + indices[t * 3 + 1]].xyz, 1.0);
            const float3 c = rowsApply(r.world0, r.world1, r.world2, positions[s.mesh.firstPoint + indices[t * 3 + 2]].xyz, 1.0);
            power = 0.5 * length(cross(b - a, c - a)) * luminance;
        }
    }
    emissive[cumulativeAt() + 1 + g] = power;
}

/// The powers accumulated in order, in place, and the total in the header.
[shader("compute")]
[numthreads(1, 1, 1)]
void accumulatePower(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint base = cumulativeAt();
    float running = 0.0;
    emissive[base] = 0.0;
    for (uint g = 0; g < triangleCount; ++g) {
        running += emissive[base + 1 + g];
        emissive[base + 1 + g] = running;
    }
    emissive[2] = running;
}
)";

}   // namespace

Result<EmissiveTable> EmissiveTable::create(gpu::ShaderLibrary& library) {
    EmissiveTable table;
    table.library_ = &library;
    table.device_ = &library.device();
    return table;
}

Result<void> EmissiveTable::build(const MaterialFrame& frame, uint32_t rows) {
    total_ = 0.0F;
    triangles_ = 0;
    if (frame.programs == nullptr || frame.scene == nullptr || frame.records == nullptr || frame.blob == nullptr ||
        frame.textures == nullptr || frame.scene->instanceCount() == 0 || rows == 0) {
        return ok();
    }
    if (frame.programs->module() != module_ || !probe_.has_value()) {
        const std::string name = frame.programs->module() + "_emissive";
        const std::string source = "import " + frame.programs->module() + ";\n" + kKernels;
        auto program = library_->loadSource(name, source, {"probeRows", "trianglePower", "accumulatePower"});
        if (!program) return std::move(program).error();
        auto probe = gpu::ComputeKernel::create(*library_, name, "probeRows");
        if (!probe) return std::move(probe).error();
        auto power = gpu::ComputeKernel::create(*library_, name, "trianglePower");
        if (!power) return std::move(power).error();
        auto accumulate = gpu::ComputeKernel::create(*library_, name, "accumulatePower");
        if (!accumulate) return std::move(accumulate).error();
        probe_.emplace(std::move(*probe));
        power_.emplace(std::move(*power));
        accumulate_.emplace(std::move(*accumulate));
        module_ = frame.programs->module();
    }
    // Each record's first triangle: bookkeeping over the draws' counts.
    const world::GpuScene& scene = *frame.scene;
    const uint32_t records = scene.instanceCount();
    std::vector<uint32_t> bases(size_t{records} + 1, 0);
    for (const world::DrawRange& draw : scene.draws()) {
        for (uint32_t k = 0; k < draw.instances; ++k) {
            const uint32_t i = draw.firstInstance + k;
            if (i < records) {
                bases[size_t{i} + 1] = scene.meshes()[draw.mesh]->triangles;
            }
        }
    }
    uint64_t triangles = 0;
    for (size_t i = 1; i < bases.size(); ++i) {
        triangles += bases[i];
        if (triangles > UINT32_MAX) {
            return Error(ErrorCode::OutOfMemory, "emissive table: more than 2^32 triangles");
        }
        bases[i] = static_cast<uint32_t>(triangles);
    }
    triangles_ = static_cast<uint32_t>(triangles);
    std::vector<float> words(4 + size_t{rows} + bases.size() + size_t{triangles_} + 1, 0.0F);
    words[0] = std::bit_cast<float>(triangles_);
    words[1] = std::bit_cast<float>(records);
    words[3] = std::bit_cast<float>(rows);
    for (size_t i = 0; i < bases.size(); ++i) {
        words[4 + rows + i] = std::bit_cast<float>(bases[i]);
    }
    auto made = gpu::Buffer::fromSpan<float>(*device_, words, "emissive.table");
    if (!made) return std::move(made).error();
    table_ = std::move(*made);
    const render::Projection none;
    const auto bind = [&](rhi::ShaderCursor cursor) {
        bindMaterialFrame(cursor, frame, none);
        cursor["emissive"].setBinding(table_.rhi());
        cursor["rowCount"].setData(rows);
        cursor["recordCount"].setData(records);
        cursor["triangleCount"].setData(triangles_);
    };
    {
        gpu::CommandBatch batch(*device_);
        probe_->dispatch(batch, {1, 1, 1}, bind);
        if (triangles_ > 0) {
            power_->dispatch(batch, {triangles_, 1, 1}, bind);
            accumulate_->dispatch(batch, {1, 1, 1}, bind);
        }
        LRT_TRY(batch.submit(true));
    }
    LRT_TRY(table_.read(*device_, 8, sizeof(float), &total_));
    return ok();
}

}   // namespace lrt::technique
