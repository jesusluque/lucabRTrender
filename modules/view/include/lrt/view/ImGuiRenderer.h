// Copyright (c) 2026 lucabRTrender contributors.
//
// Dear ImGui drawn by the engine's slang-rhi device: its textures made and
// updated as ImGui asks (ImGuiBackendFlags_RendererHasTextures), its draw
// lists uploaded whole and drawn over a display texture with a vertex-pulling
// pipeline (shaders/lrt/view/imgui.slang). The UI's triangles are chrome
// ImGui tessellates on the CPU; nothing of the scene passes through here.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include <slang-rhi.h>

#include "lrt/core/Result.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/RasterKernel.h"
#include "lrt/gpu/Texture.h"

struct ImDrawData;

namespace lrt::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace lrt::view {

class ImGuiRenderer {
public:
    /// For the current ImGui context.
    [[nodiscard]] static Result<std::unique_ptr<ImGuiRenderer>> create(gpu::ShaderLibrary& library);
    ~ImGuiRenderer();

    /// `data`'s texture requests, then its lists over `target` (loaded, not
    /// cleared): `width` x `height` pixels of `format`.
    [[nodiscard]] Result<void> render(gpu::CommandBatch& batch, ImDrawData* data, rhi::ITextureView* target,
                                      rhi::Format format, uint32_t width, uint32_t height);

private:
    ImGuiRenderer() = default;
    [[nodiscard]] Result<void> updateTextures(ImDrawData* data);

    gpu::ShaderLibrary*                        library_ = nullptr;
    gpu::Device*                               device_ = nullptr;
    std::map<rhi::Format, gpu::RasterKernel>   passes_;   ///< a pipeline per target format
    gpu::Sampler                               sampler_;
    gpu::Buffer                                vertices_, indices_;
    std::vector<std::unique_ptr<gpu::Texture>> textures_;   ///< by ImTextureID - 1; null where destroyed
};

}   // namespace lrt::view
