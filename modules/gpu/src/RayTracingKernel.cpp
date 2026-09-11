// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/RayTracingKernel.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::gpu {

Result<RayTracingKernel> RayTracingKernel::create(ShaderLibrary& library, const RayTracingDesc& desc) {
    if (!library.device().caps().rayTracing) {
        return Error::make(ErrorCode::Unsupported, "{}: no ray tracing pipelines on {} (trace in compute)",
                           desc.module, toString(library.device().backend()));
    }
    std::vector<std::string> entries{desc.rayGen};
    entries.insert(entries.end(), desc.misses.begin(), desc.misses.end());
    for (const auto& group : desc.hitGroups) {
        for (const std::string& entry : {group.closestHit, group.anyHit, group.intersection}) {
            if (!entry.empty()) {
                entries.push_back(entry);
            }
        }
    }
    auto program = library.load(desc.module, entries);
    if (!program) return std::move(program).error();

    RayTracingKernel kernel;
    kernel.name_ = desc.module + ":" + desc.rayGen;
    kernel.program_ = *program;
    std::vector<rhi::HitGroupDesc> groups;
    for (const auto& group : desc.hitGroups) {
        rhi::HitGroupDesc g;
        g.hitGroupName = group.name.c_str();
        g.closestHitEntryPoint = group.closestHit.empty() ? nullptr : group.closestHit.c_str();
        g.anyHitEntryPoint = group.anyHit.empty() ? nullptr : group.anyHit.c_str();
        g.intersectionEntryPoint = group.intersection.empty() ? nullptr : group.intersection.c_str();
        groups.push_back(g);
    }
    rhi::RayTracingPipelineDesc pipeline;
    pipeline.program = kernel.program_->program.get();
    pipeline.hitGroupCount = static_cast<uint32_t>(groups.size());
    pipeline.hitGroups = groups.data();
    pipeline.maxRecursion = desc.maxRecursion;
    pipeline.maxRayPayloadSize = desc.payloadBytes;
    rhi::IDevice* device = library.device().rhi();
    if (SLANG_FAILED(device->createRayTracingPipeline(pipeline, kernel.pipeline_.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot make a ray tracing pipeline for {}", kernel.name_);
    }
    std::vector<const char*> rayGens{desc.rayGen.c_str()};
    std::vector<const char*> misses;
    for (const std::string& miss : desc.misses) {
        misses.push_back(miss.c_str());
    }
    std::vector<const char*> hitNames;
    for (const auto& group : desc.hitGroups) {
        hitNames.push_back(group.name.c_str());
    }
    rhi::ShaderTableDesc table;
    table.program = kernel.program_->program.get();
    table.rayGenShaderCount = 1;
    table.rayGenShaderEntryPointNames = rayGens.data();
    table.missShaderCount = static_cast<uint32_t>(misses.size());
    table.missShaderEntryPointNames = misses.data();
    table.hitGroupCount = static_cast<uint32_t>(hitNames.size());
    table.hitGroupNames = hitNames.data();
    if (SLANG_FAILED(device->createShaderTable(table, kernel.table_.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot make a shader table for {}", kernel.name_);
    }
    return kernel;
}

void RayTracingKernel::dispatch(CommandBatch& batch, uint32_t width, uint32_t height, uint32_t depth,
                                const Bind& bind) const {
    if (width == 0 || height == 0 || depth == 0) {
        return;
    }
    rhi::IRayTracingPassEncoder* pass = batch.encoder()->beginRayTracingPass();
    rhi::IShaderObject* root = pass->bindPipeline(pipeline_.get(), table_.get());
    if (bind) {
        bind(rhi::ShaderCursor(root));
    }
    pass->dispatchRays(0, width, height, depth);
    pass->end();
    batch.markDirty();
}

}   // namespace lrt::gpu
