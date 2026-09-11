// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/view/ImGuiRenderer.h"

#include <algorithm>
#include <cstring>
#include <span>

#include <imgui.h>

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

namespace lrt::view {

namespace {

struct UiVertex {
    float    position[2];
    float    uv[2];
    uint32_t colour;
};
static_assert(sizeof(ImDrawVert) == sizeof(UiVertex));
static_assert(sizeof(ImDrawIdx) == sizeof(uint32_t));

Result<void> grow(gpu::Device& device, gpu::Buffer& buffer, uint64_t bytes, uint32_t element, const char* label) {
    if (buffer.valid() && buffer.bytes() >= bytes) {
        return ok();
    }
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(bytes * 3 / 2, element * 1024);
    desc.bytes -= desc.bytes % element;
    desc.elementBytes = element;
    desc.label = label;
    auto made = gpu::Buffer::create(device, desc);
    if (!made) return std::move(made).error();
    buffer = std::move(*made);
    return ok();
}

}   // namespace

Result<std::unique_ptr<ImGuiRenderer>> ImGuiRenderer::create(gpu::ShaderLibrary& library) {
    auto renderer = std::unique_ptr<ImGuiRenderer>(new ImGuiRenderer());
    renderer->library_ = &library;
    renderer->device_ = &library.device();
    rhi::SamplerDesc sampling;
    sampling.addressU = rhi::TextureAddressingMode::ClampToEdge;
    sampling.addressV = rhi::TextureAddressingMode::ClampToEdge;
    sampling.mipFilter = rhi::TextureFilteringMode::Point;
    auto sampler = gpu::Sampler::create(*renderer->device_, sampling);
    if (!sampler) return std::move(sampler).error();
    renderer->sampler_ = std::move(*sampler);

    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "lrt_slang_rhi";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures;
    return renderer;
}

ImGuiRenderer::~ImGuiRenderer() {
    if (ImGui::GetCurrentContext() == nullptr) {
        return;
    }
    for (ImTextureData* texture : ImGui::GetPlatformIO().Textures) {
        if (texture->RefCount == 1) {
            texture->SetTexID(ImTextureID_Invalid);
            texture->SetStatus(ImTextureStatus_Destroyed);
        }
    }
    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = nullptr;
    io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
}

Result<void> ImGuiRenderer::updateTextures(ImDrawData* data) {
    if (data->Textures == nullptr) {
        return ok();
    }
    for (ImTextureData* texture : *data->Textures) {
        if (texture->Status == ImTextureStatus_OK) {
            continue;
        }
        if (texture->Status == ImTextureStatus_WantDestroy) {
            if (texture->UnusedFrames > 0) {
                const ImTextureID id = texture->GetTexID();
                if (id != ImTextureID_Invalid && id <= textures_.size()) {
                    textures_[static_cast<size_t>(id - 1)].reset();
                }
                texture->SetTexID(ImTextureID_Invalid);
                texture->SetStatus(ImTextureStatus_Destroyed);
            }
            continue;
        }
        // Create, or update: the whole atlas goes up again (it changes when
        // glyphs are added, rarely).
        if (texture->Format != ImTextureFormat_RGBA32) {
            return Error(ErrorCode::Unsupported, "imgui: only RGBA32 textures");
        }
        ImTextureID id = texture->GetTexID();
        if (texture->Status == ImTextureStatus_WantCreate || id == ImTextureID_Invalid ||
            textures_[static_cast<size_t>(id - 1)] == nullptr ||
            textures_[static_cast<size_t>(id - 1)]->width() != static_cast<uint32_t>(texture->Width) ||
            textures_[static_cast<size_t>(id - 1)]->height() != static_cast<uint32_t>(texture->Height)) {
            gpu::TextureDesc desc;
            desc.width = static_cast<uint32_t>(texture->Width);
            desc.height = static_cast<uint32_t>(texture->Height);
            desc.format = rhi::Format::RGBA8Unorm;
            desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopyDestination;
            desc.label = "imgui.texture";
            auto made = gpu::Texture::create(*device_, desc);
            if (!made) return std::move(made).error();
            if (id == ImTextureID_Invalid) {
                textures_.push_back(std::make_unique<gpu::Texture>(std::move(*made)));
                id = static_cast<ImTextureID>(textures_.size());
            } else {
                textures_[static_cast<size_t>(id - 1)] = std::make_unique<gpu::Texture>(std::move(*made));
            }
        }
        LRT_TRY(textures_[static_cast<size_t>(id - 1)]->upload(
            *device_, 0, 0,
            std::span<const std::byte>(static_cast<const std::byte*>(texture->GetPixels()),
                                       static_cast<size_t>(texture->GetSizeInBytes()))));
        texture->SetTexID(id);
        texture->SetStatus(ImTextureStatus_OK);
    }
    return ok();
}

Result<void> ImGuiRenderer::render(gpu::CommandBatch& batch, ImDrawData* data, rhi::ITextureView* target,
                                   rhi::Format format, uint32_t width, uint32_t height) {
    if (data == nullptr) {
        return ok();
    }
    auto found = passes_.find(format);
    if (found == passes_.end()) {
        gpu::RasterDesc desc;
        desc.module = "lrt/view/imgui";
        desc.vertexEntry = "uiVertex";
        desc.fragmentEntry = "uiFragment";
        rhi::ColorTargetDesc colour;
        colour.format = format;
        colour.enableBlend = true;
        colour.color = {rhi::BlendFactor::SrcAlpha, rhi::BlendFactor::InvSrcAlpha, rhi::BlendOp::Add};
        colour.alpha = {rhi::BlendFactor::One, rhi::BlendFactor::InvSrcAlpha, rhi::BlendOp::Add};
        desc.targets = {colour};
        auto made = gpu::RasterKernel::create(*library_, desc);
        if (!made) return std::move(made).error();
        found = passes_.emplace(format, std::move(*made)).first;
    }
    LRT_TRY(updateTextures(data));
    if (data->TotalVtxCount == 0 || data->DisplaySize.x <= 0.0F || data->DisplaySize.y <= 0.0F) {
        return ok();
    }
    // Every list's vertices and indices, end to end.
    LRT_TRY(grow(*device_, vertices_, uint64_t(data->TotalVtxCount) * sizeof(ImDrawVert), sizeof(ImDrawVert),
                 "imgui.vertices"));
    LRT_TRY(grow(*device_, indices_, uint64_t(data->TotalIdxCount) * sizeof(ImDrawIdx), sizeof(ImDrawIdx),
                 "imgui.indices"));
    std::vector<ImDrawVert> vertices;
    std::vector<ImDrawIdx> indices;
    vertices.reserve(static_cast<size_t>(data->TotalVtxCount));
    indices.reserve(static_cast<size_t>(data->TotalIdxCount));
    for (const ImDrawList* list : data->CmdLists) {
        vertices.insert(vertices.end(), list->VtxBuffer.begin(), list->VtxBuffer.end());
        indices.insert(indices.end(), list->IdxBuffer.begin(), list->IdxBuffer.end());
    }
    LRT_TRY(vertices_.write(*device_, 0, vertices.size() * sizeof(ImDrawVert), vertices.data()));
    LRT_TRY(indices_.write(*device_, 0, indices.size() * sizeof(ImDrawIdx), indices.data()));

    const float scaleX = 2.0F / data->DisplaySize.x;
    const float scaleY = -2.0F / data->DisplaySize.y;
    const float translateX = -1.0F - data->DisplayPos.x * scaleX;
    const float translateY = 1.0F - data->DisplayPos.y * scaleY;
    const uint32_t idsIncludeStart = device_->caps().drawIdsIncludeStart ? 1u : 0u;
    gpu::RasterPass pass;
    pass.width = width;
    pass.height = height;
    pass.colours = {target};
    std::vector<gpu::RasterDraw> draws;
    uint32_t listVertices = 0;
    uint32_t listIndices = 0;
    for (const ImDrawList* list : data->CmdLists) {
        for (const ImDrawCmd& command : list->CmdBuffer) {
            if (command.UserCallback != nullptr || command.ElemCount == 0) {
                continue;
            }
            const float fx = data->FramebufferScale.x;
            const float fy = data->FramebufferScale.y;
            const float x0 = std::max((command.ClipRect.x - data->DisplayPos.x) * fx, 0.0F);
            const float y0 = std::max((command.ClipRect.y - data->DisplayPos.y) * fy, 0.0F);
            const float x1 = std::min((command.ClipRect.z - data->DisplayPos.x) * fx, float(width));
            const float y1 = std::min((command.ClipRect.w - data->DisplayPos.y) * fy, float(height));
            if (x1 <= x0 || y1 <= y0) {
                continue;
            }
            const ImTextureID id = command.GetTexID();
            if (id == ImTextureID_Invalid || id > textures_.size() || textures_[static_cast<size_t>(id - 1)] == nullptr) {
                continue;
            }
            rhi::ITexture* texture = textures_[static_cast<size_t>(id - 1)]->rhi();
            gpu::RasterDraw draw;
            draw.vertexCount = command.ElemCount;
            draw.firstVertex = listIndices + command.IdxOffset;
            draw.scissor = {static_cast<uint32_t>(x0), static_cast<uint32_t>(y0), static_cast<uint32_t>(x1 + 0.5F),
                            static_cast<uint32_t>(y1 + 0.5F)};
            const uint32_t vertexBase = listVertices + command.VtxOffset;
            draw.bind = [this, texture, vertexBase, scaleX, scaleY, translateX, translateY,
                         idsIncludeStart](rhi::ShaderCursor cursor) {
                cursor["vertices"].setBinding(vertices_.rhi());
                cursor["indices"].setBinding(indices_.rhi());
                cursor["atlas"].setBinding(texture);
                cursor["atlasSampler"].setBinding(sampler_.rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["scaleX"].setData(scaleX);
                p["scaleY"].setData(scaleY);
                p["translateX"].setData(translateX);
                p["translateY"].setData(translateY);
                p["vertexBase"].setData(vertexBase);
                p["idsIncludeStart"].setData(idsIncludeStart);
            };
            draws.push_back(std::move(draw));
        }
        listVertices += static_cast<uint32_t>(list->VtxBuffer.Size);
        listIndices += static_cast<uint32_t>(list->IdxBuffer.Size);
    }
    found->second.run(batch, pass, draws);
    return ok();
}

}   // namespace lrt::view
