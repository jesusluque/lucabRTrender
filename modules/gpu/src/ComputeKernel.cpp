// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/ComputeKernel.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"

namespace lrt::gpu {

Result<ComputeKernel> ComputeKernel::create(ShaderLibrary& library, const std::string& module,
                                            const std::string& entry) {
    auto program = library.load(module, {entry});
    if (!program) {
        return std::move(program).error();
    }
    ComputeKernel kernel;
    kernel.name_ = module + ":" + entry;
    kernel.program_ = *program;
    rhi::ComputePipelineDesc desc;
    desc.program = kernel.program_->program.get();
    if (SLANG_FAILED(library.device().rhi()->createComputePipeline(
            desc, kernel.pipeline_.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot make a pipeline for {}",
                           kernel.name_);
    }
    return kernel;
}

void ComputeKernel::dispatch(CommandBatch& batch, std::array<uint32_t, 3> threads,
                             const Bind& bind) const {
    if (threads[0] == 0 || threads[1] == 0 || threads[2] == 0) {
        return;
    }
    rhi::IComputePassEncoder* pass = batch.encoder()->beginComputePass();
    rhi::IShaderObject* root = pass->bindPipeline(pipeline_.get());
    if (bind) {
        bind(rhi::ShaderCursor(root));
    }
    const auto& group = program_->threadGroup;
    pass->dispatchCompute(groupsFor(threads[0], group[0]), groupsFor(threads[1], group[1]),
                          groupsFor(threads[2], group[2]));
    pass->end();
    batch.dirty_ = true;
}

}   // namespace lrt::gpu
