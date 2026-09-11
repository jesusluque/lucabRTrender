// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/RasterKernel.h"

#include <algorithm>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::gpu {

Result<RasterKernel> RasterKernel::create(ShaderLibrary& library, const RasterDesc& desc) {
    if (!library.device().caps().rasterization) {
        return Error::make(ErrorCode::Unsupported, "{}: this device does not rasterise", desc.module);
    }
    auto program = library.load(desc.module, {desc.vertexEntry, desc.fragmentEntry});
    if (!program) return std::move(program).error();
    RasterKernel kernel;
    kernel.name_ = desc.module + ":" + desc.vertexEntry + "+" + desc.fragmentEntry;
    kernel.program_ = *program;
    rhi::RenderPipelineDesc pipeline;
    pipeline.program = kernel.program_->program.get();
    pipeline.primitiveTopology = desc.topology;
    pipeline.targets = desc.targets.data();
    pipeline.targetCount = static_cast<uint32_t>(desc.targets.size());
    pipeline.depthStencil.format = desc.depthFormat;
    pipeline.depthStencil.depthTestEnable = desc.depthFormat != rhi::Format::Undefined && desc.depthTest;
    pipeline.depthStencil.depthWriteEnable = desc.depthFormat != rhi::Format::Undefined && desc.depthWrite;
    pipeline.depthStencil.depthFunc = desc.depthFunc;
    pipeline.rasterizer.cullMode = desc.cull;
    pipeline.rasterizer.frontFace = desc.frontFace;
    if (SLANG_FAILED(library.device().rhi()->createRenderPipeline(pipeline, kernel.pipeline_.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot make a render pipeline for {}", kernel.name_);
    }
    return kernel;
}

void RasterKernel::run(CommandBatch& batch, const RasterPass& pass, std::span<const RasterDraw> draws) const {
    std::vector<rhi::RenderPassColorAttachment> colours(pass.colours.size());
    for (size_t k = 0; k < colours.size(); ++k) {
        colours[k].view = pass.colours[k];
        if (k < pass.clearColours.size()) {
            colours[k].loadOp = rhi::LoadOp::Clear;
            for (int c = 0; c < 4; ++c) {
                colours[k].clearValue[c] = pass.clearColours[k][static_cast<size_t>(c)];
            }
        } else {
            colours[k].loadOp = rhi::LoadOp::Load;
        }
    }
    rhi::RenderPassDepthStencilAttachment depth;
    depth.view = pass.depth;
    depth.depthLoadOp = pass.clearDepth ? rhi::LoadOp::Clear : rhi::LoadOp::Load;
    depth.depthClearValue = pass.depthClear;
    rhi::RenderPassDesc desc;
    desc.colorAttachments = colours.data();
    desc.colorAttachmentCount = static_cast<uint32_t>(colours.size());
    desc.depthStencilAttachment = pass.depth != nullptr ? &depth : nullptr;
    rhi::IRenderPassEncoder* encoder = batch.encoder()->beginRenderPass(desc);
    rhi::RenderState state;
    state.viewports[0] = rhi::Viewport::fromSize(static_cast<float>(pass.width), static_cast<float>(pass.height));
    state.viewportCount = 1;
    state.scissorRects[0] = rhi::ScissorRect::fromSize(pass.width, pass.height);
    state.scissorRectCount = 1;
    bool shared = false;   // the pass's root object is bound
    for (const RasterDraw& draw : draws) {
        if (draw.vertexCount == 0 || draw.instanceCount == 0) {
            continue;
        }
        if (draw.bind || !shared) {
            rhi::IShaderObject* root = encoder->bindPipeline(pipeline_.get());
            if (pass.bind) {
                pass.bind(rhi::ShaderCursor(root));
            }
            if (draw.bind) {
                draw.bind(rhi::ShaderCursor(root));
            }
            shared = !draw.bind;
        }
        rhi::RenderState drawState = state;
        if (draw.scissor[2] > draw.scissor[0] && draw.scissor[3] > draw.scissor[1]) {
            drawState.scissorRects[0].minX = std::min(draw.scissor[0], pass.width);
            drawState.scissorRects[0].minY = std::min(draw.scissor[1], pass.height);
            drawState.scissorRects[0].maxX = std::min(draw.scissor[2], pass.width);
            drawState.scissorRects[0].maxY = std::min(draw.scissor[3], pass.height);
        }
        encoder->setRenderState(drawState);
        rhi::DrawArguments args;
        args.vertexCount = draw.vertexCount;
        args.instanceCount = draw.instanceCount;
        args.startVertexLocation = draw.firstVertex;
        args.startInstanceLocation = draw.firstInstance;
        encoder->draw(args);
    }
    encoder->end();
    batch.markDirty();
}

}   // namespace lrt::gpu
