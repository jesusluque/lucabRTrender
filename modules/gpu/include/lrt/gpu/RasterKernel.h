// Copyright (c) 2026 lucabRTrender contributors.
//
// A rasterisation pipeline -- a vertex and a fragment entry point -- bound by
// name like every kernel, and the render passes it draws in. No vertex
// buffers or input layouts: vertex shaders pull what they draw from
// StructuredBuffers by SV_VertexID, as the point rasteriser always has.
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "lrt/core/Result.h"

namespace lrt::gpu {

class CommandBatch;
class ShaderLibrary;
struct Program;

struct RasterDesc {
    std::string                         module;
    std::string                         vertexEntry;
    std::string                         fragmentEntry;
    std::vector<rhi::ColorTargetDesc>   targets;
    rhi::Format                         depthFormat = rhi::Format::Undefined;   ///< none
    bool                                depthTest = true;
    bool                                depthWrite = true;
    rhi::ComparisonFunc                 depthFunc = rhi::ComparisonFunc::Less;
    rhi::CullMode                       cull = rhi::CullMode::None;
    rhi::FrontFaceMode                  frontFace = rhi::FrontFaceMode::CounterClockwise;
    rhi::PrimitiveTopology              topology = rhi::PrimitiveTopology::TriangleList;
};

/// Where a pass draws: its targets, and whether they start cleared.
struct RasterPass {
    uint32_t                              width = 0;
    uint32_t                              height = 0;
    std::vector<rhi::ITextureView*>       colours;
    std::vector<std::array<float, 4>>     clearColours;   ///< per colour target; empty loads instead
    rhi::ITextureView*                    depth = nullptr;
    bool                                  clearDepth = true;
    float                                 depthClear = 1.0F;
    /// What every draw binds. Draws with no `bind` of their own share one
    /// root object, so thousands of them cost no binding each: what differs
    /// between them travels in their start locations (SV_StartVertexLocation,
    /// SV_StartInstanceLocation).
    std::function<void(rhi::ShaderCursor)> bind;
};

/// One draw: its counts, and what it binds beyond the pass (a fresh root
/// object when it binds anything).
struct RasterDraw {
    uint32_t                               vertexCount = 0;
    uint32_t                               instanceCount = 1;
    uint32_t                               firstVertex = 0;
    uint32_t                               firstInstance = 0;
    std::function<void(rhi::ShaderCursor)> bind;
};

class RasterKernel {
public:
    RasterKernel() = default;

    [[nodiscard]] static Result<RasterKernel> create(ShaderLibrary& library, const RasterDesc& desc);

    /// Records one render pass with `draws`, in order, into `batch`.
    void run(CommandBatch& batch, const RasterPass& pass, std::span<const RasterDraw> draws) const;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    std::string                        name_;
    std::shared_ptr<const Program>     program_;
    rhi::ComPtr<rhi::IRenderPipeline>  pipeline_;
};

}   // namespace lrt::gpu
