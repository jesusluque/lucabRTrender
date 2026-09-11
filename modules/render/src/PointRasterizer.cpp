// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/render/PointRasterizer.h"

#include <algorithm>

#include "FrameParams.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::render {
namespace {

constexpr rhi::Format kColourFormat = rhi::Format::RGBA32Float;

Result<rhi::ComPtr<rhi::IRenderPipeline>> makePipeline(gpu::ShaderLibrary& library,
                                                       const char* vertexEntry, bool writeColour,
                                                       bool additive, bool depthWrite,
                                                       rhi::ComparisonFunc depthFunc) {
    auto program = library.load("lrt/points/points_raster", {vertexEntry, "pointsFragment"});
    if (!program) {
        return std::move(program).error();
    }
    rhi::ColorTargetDesc targets[2];
    for (rhi::ColorTargetDesc& target : targets) {
        target.format = kColourFormat;
        target.writeMask = writeColour ? rhi::RenderTargetWriteMask::All : rhi::RenderTargetWriteMask::None;
        if (additive) {
            target.enableBlend = true;
            target.color = {rhi::BlendFactor::One, rhi::BlendFactor::One, rhi::BlendOp::Add};
            target.alpha = {rhi::BlendFactor::One, rhi::BlendFactor::One, rhi::BlendOp::Add};
        }
    }
    // The depth target keeps the nearest view z, not a sum.
    if (additive) {
        targets[1].enableBlend = false;
    }
    rhi::RenderPipelineDesc desc;
    desc.program = (*program)->program.get();
    desc.targets = targets;
    desc.targetCount = 2;
    desc.depthStencil.format = rhi::Format::D32Float;
    desc.depthStencil.depthTestEnable = true;
    desc.depthStencil.depthWriteEnable = depthWrite;
    desc.depthStencil.depthFunc = depthFunc;
    desc.rasterizer.cullMode = rhi::CullMode::None;
    rhi::ComPtr<rhi::IRenderPipeline> pipeline;
    if (SLANG_FAILED(library.device().rhi()->createRenderPipeline(desc, pipeline.writeRef()))) {
        return Error(ErrorCode::ShaderFailure, "cannot make the point raster pipeline");
    }
    return pipeline;
}

}   // namespace

Result<PointRasterizer> PointRasterizer::create(gpu::ShaderLibrary& library) {
    if (!library.device().caps().rasterization) {
        return Error(ErrorCode::Unsupported,
                     "this device does not rasterise; draw points as discs in the splat pipeline");
    }
    PointRasterizer r;
    r.device_ = &library.device();
    auto opaque = makePipeline(library, "pointsVertex", true, false, true, rhi::ComparisonFunc::Less);
    if (!opaque) return std::move(opaque).error();
    auto depth = makePipeline(library, "pointsDepthVertex", false, false, true, rhi::ComparisonFunc::Less);
    if (!depth) return std::move(depth).error();
    auto accumulate =
        makePipeline(library, "pointsVertex", true, true, false, rhi::ComparisonFunc::LessEqual);
    if (!accumulate) return std::move(accumulate).error();
    r.opaque_ = *opaque;
    r.surfaceDepth_ = *depth;
    r.surfaceAccumulate_ = *accumulate;
    auto resolve = gpu::ComputeKernel::create(library, "lrt/points/points_resolve", "pointsResolve");
    if (!resolve) return std::move(resolve).error();
    auto edl = gpu::ComputeKernel::create(library, "lrt/points/points_edl", "pointsEdl");
    if (!edl) return std::move(edl).error();
    r.resolve_ = std::move(*resolve);
    r.edl_ = std::move(*edl);
    return r;
}

Result<void> PointRasterizer::reserve(uint32_t width, uint32_t height) {
    if (width == width_ && height == height_ && colourTarget_ != nullptr) {
        return ok();
    }
    rhi::IDevice* rhiDevice = device_->rhi();
    const auto texture = [&](rhi::Format format, rhi::TextureUsage usage, rhi::ResourceState state,
                             rhi::ComPtr<rhi::ITexture>& into, rhi::ComPtr<rhi::ITextureView>& view) -> Result<void> {
        rhi::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2D;
        desc.size = {width, height, 1};
        desc.format = format;
        desc.usage = usage;
        desc.defaultState = state;
        if (SLANG_FAILED(rhiDevice->createTexture(desc, nullptr, into.writeRef()))) {
            return Error(ErrorCode::OutOfMemory, "cannot make a point render target");
        }
        rhi::TextureViewDesc viewDesc;
        viewDesc.format = format;
        if (SLANG_FAILED(rhiDevice->createTextureView(into, viewDesc, view.writeRef()))) {
            return Error(ErrorCode::DeviceFailure, "cannot view a point render target");
        }
        return ok();
    };
    const auto colourUsage = rhi::TextureUsage::RenderTarget | rhi::TextureUsage::ShaderResource;
    LRT_TRY(texture(kColourFormat, colourUsage, rhi::ResourceState::RenderTarget, colourTarget_, colourView_));
    LRT_TRY(texture(kColourFormat, colourUsage, rhi::ResourceState::RenderTarget, depthTarget_, depthView_));
    LRT_TRY(texture(rhi::Format::D32Float, rhi::TextureUsage::DepthStencil, rhi::ResourceState::DepthWrite,
                    zBuffer_, zView_));
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{width} * height * 16;
    desc.elementBytes = 16;
    desc.label = "points.shaded";
    auto shaded = gpu::Buffer::create(*device_, desc);
    if (!shaded) return std::move(shaded).error();
    shaded_ = *shaded;
    width_ = width;
    height_ = height;
    return ok();
}

Result<void> PointRasterizer::render(const Camera& camera, std::span<const PointInstance> instances,
                                     const RenderSettings& settings, RenderTargets& targets) {
    return render(projectionFor(camera, settings.width, settings.height), instances, settings, targets);
}

Result<void> PointRasterizer::render(const Projection& projection,
                                     std::span<const PointInstance> instances,
                                     const RenderSettings& settings, RenderTargets& targets) {
    LRT_TRY(reserve(settings.width, settings.height));
    const uint64_t pixels = uint64_t{settings.width} * settings.height;
    if (targets.width != settings.width || targets.height != settings.height || !targets.colour.valid()) {
        gpu::BufferDesc desc;
        desc.bytes = pixels * 16;
        desc.elementBytes = 16;
        desc.label = "points.colour";
        auto colour = gpu::Buffer::create(*device_, desc);
        if (!colour) return std::move(colour).error();
        desc.bytes = pixels * 4;
        desc.elementBytes = 4;
        desc.label = "points.depth";
        auto depth = gpu::Buffer::create(*device_, desc);
        if (!depth) return std::move(depth).error();
        targets.colour = *colour;
        targets.depth = *depth;
        targets.width = settings.width;
        targets.height = settings.height;
    }

    const bool surface = std::any_of(instances.begin(), instances.end(), [](const PointInstance& i) {
        return i.style.surfaceDepthOffset > 0.0F;
    });
    float edlStrength = 0.0F;
    float edlRadius = 1.0F;
    for (const PointInstance& instance : instances) {
        if (instance.style.edlStrength > edlStrength) {
            edlStrength = instance.style.edlStrength;
            edlRadius = instance.style.edlRadius;
        }
    }

    gpu::CommandBatch batch(*device_);
    const auto pass = [&](rhi::IRenderPipeline* pipeline, bool clear, bool surfaceMode) {
        rhi::RenderPassColorAttachment colour;
        colour.view = colourView_;
        colour.loadOp = clear ? rhi::LoadOp::Clear : rhi::LoadOp::Load;
        rhi::RenderPassColorAttachment depthColour;
        depthColour.view = depthView_;
        depthColour.loadOp = clear ? rhi::LoadOp::Clear : rhi::LoadOp::Load;
        rhi::RenderPassColorAttachment attachments[2] = {colour, depthColour};
        rhi::RenderPassDepthStencilAttachment z;
        z.view = zView_;
        z.depthLoadOp = clear ? rhi::LoadOp::Clear : rhi::LoadOp::Load;
        z.depthClearValue = 1.0F;
        rhi::RenderPassDesc desc;
        desc.colorAttachments = attachments;
        desc.colorAttachmentCount = 2;
        desc.depthStencilAttachment = &z;
        rhi::IRenderPassEncoder* encoder = batch.encoder()->beginRenderPass(desc);
        for (const PointInstance& instance : instances) {
            const scene::GpuPoints* cloud = instance.points;
            if (cloud == nullptr || cloud->count == 0) {
                continue;
            }
            rhi::IShaderObject* root = encoder->bindPipeline(pipeline);
            rhi::ShaderCursor cursor(root);
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["colours"].setBinding(cloud->colours.rhi());
            PointStyle style = instance.style;
            if (!surfaceMode) {
                style.surfaceDepthOffset = 0.0F;
            } else if (style.surfaceDepthOffset <= 0.0F) {
                style.surfaceDepthOffset = 1e-6F;
            }
            setPointParams(cursor["params"], projection, settings.width, settings.height,
                           projection.worldToView * instance.objectToWorld, style, cloud->count);
            rhi::RenderState state;
            state.viewports[0] = rhi::Viewport::fromSize(static_cast<float>(settings.width),
                                                         static_cast<float>(settings.height));
            state.viewportCount = 1;
            state.scissorRects[0] = rhi::ScissorRect::fromSize(settings.width, settings.height);
            state.scissorRectCount = 1;
            encoder->setRenderState(state);
            rhi::DrawArguments args;
            args.vertexCount = cloud->count * 6;
            encoder->draw(args);
        }
        encoder->end();
        batch.markDirty();
    };
    if (surface) {
        pass(surfaceDepth_, true, true);
        pass(surfaceAccumulate_, false, true);
    } else {
        pass(opaque_, true, false);
    }

    resolve_.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colourTarget"].setBinding(colourTarget_);
        cursor["depthTarget"].setBinding(depthTarget_);
        cursor["colour"].setBinding(edlStrength > 0.0F ? shaded_.rhi() : targets.colour.rhi());
        cursor["depth"].setBinding(targets.depth.rhi());
        cursor["params"]["width"].setData(settings.width);
        cursor["params"]["height"].setData(settings.height);
        cursor["params"]["normalise"].setData(uint32_t{surface ? 1u : 0u});
    });
    if (edlStrength > 0.0F) {
        PointStyle style;
        style.edlStrength = edlStrength;
        style.edlRadius = edlRadius;
        edl_.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colourIn"].setBinding(shaded_.rhi());
            cursor["depth"].setBinding(targets.depth.rhi());
            cursor["colourOut"].setBinding(targets.colour.rhi());
            setPointParams(cursor["params"], projection, settings.width, settings.height,
                           Mat4::identity(), style, 0);
        });
    }
    return batch.submit(true);
}

}   // namespace lrt::render
