// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/material/TextureStore.h"

#include <algorithm>
#include <cstring>

#include <pxr/imaging/hio/image.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usdShade/udimUtils.h>

#include "lrt/core/Log.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace lrt::material {

namespace {

constexpr uint32_t kTextureSlots = 1024;   // texture_table.slang's kTextureSlots
constexpr uint32_t kSamplerSlots = 16;

/// shaders/lrt/material/texture_table.slang's TextureRecord.
struct TextureRecord {
    uint32_t slot = 0;
    uint32_t udimBase = 0;
    uint32_t flags = 0;
    uint32_t pad0 = 0;
    float    width = 0.0F;
    float    height = 0.0F;
    float    pad1 = 0.0F;
    float    pad2 = 0.0F;
};
static_assert(sizeof(TextureRecord) == 32);

struct Layout {
    uint32_t channels = 0;
    uint32_t component = 0;   // texture_decode.slang: 0 unorm8, 1 unorm16, 2 float16, 3 float32
    uint32_t componentBytes = 1;
    bool     srgb = false;
};

bool layoutOf(HioFormat format, Layout& out) {
    const auto set = [&](uint32_t channels, uint32_t component, uint32_t bytes, bool srgb) {
        out = {channels, component, bytes, srgb};
        return true;
    };
    switch (format) {
    case HioFormatUNorm8: return set(1, 0, 1, false);
    case HioFormatUNorm8Vec2: return set(2, 0, 1, false);
    case HioFormatUNorm8Vec3: return set(3, 0, 1, false);
    case HioFormatUNorm8Vec4: return set(4, 0, 1, false);
    case HioFormatUNorm8srgb: return set(1, 0, 1, true);
    case HioFormatUNorm8Vec2srgb: return set(2, 0, 1, true);
    case HioFormatUNorm8Vec3srgb: return set(3, 0, 1, true);
    case HioFormatUNorm8Vec4srgb: return set(4, 0, 1, true);
    case HioFormatUInt16: return set(1, 1, 2, false);
    case HioFormatUInt16Vec2: return set(2, 1, 2, false);
    case HioFormatUInt16Vec3: return set(3, 1, 2, false);
    case HioFormatUInt16Vec4: return set(4, 1, 2, false);
    case HioFormatFloat16: return set(1, 2, 2, false);
    case HioFormatFloat16Vec2: return set(2, 2, 2, false);
    case HioFormatFloat16Vec3: return set(3, 2, 2, false);
    case HioFormatFloat16Vec4: return set(4, 2, 2, false);
    case HioFormatFloat32: return set(1, 3, 4, false);
    case HioFormatFloat32Vec2: return set(2, 3, 4, false);
    case HioFormatFloat32Vec3: return set(3, 3, 4, false);
    case HioFormatFloat32Vec4: return set(4, 3, 4, false);
    default: return false;
    }
}

rhi::TextureAddressingMode addressOf(Wrap wrap) {
    switch (wrap) {
    case Wrap::Clamp: return rhi::TextureAddressingMode::ClampToEdge;
    case Wrap::Mirror: return rhi::TextureAddressingMode::MirrorRepeat;
    case Wrap::Black: return rhi::TextureAddressingMode::ClampToBorder;
    default: return rhi::TextureAddressingMode::Wrap;
    }
}

}   // namespace

Result<std::unique_ptr<TextureStore>> TextureStore::create(gpu::ShaderLibrary& library) {
    auto store = std::unique_ptr<TextureStore>(new TextureStore());
    store->device_ = &library.device();
    auto decode = gpu::ComputeKernel::create(library, "lrt/material/texture_decode", "textureDecode");
    if (!decode) return std::move(decode).error();
    store->decode_ = std::move(*decode);
    auto mips = gpu::MipGenerator::create(library);
    if (!mips) return std::move(mips).error();
    store->mips_ = std::make_unique<gpu::MipGenerator>(std::move(*mips));
    store->sampler(Wrap::Repeat, Wrap::Repeat, Filter::Linear);   // slot 0, for anything unset
    LRT_TRY(store->writeRecords());
    return store;
}

uint32_t TextureStore::request(const std::string& path, ColourSpace space) {
    const auto key = std::make_pair(path, space);
    if (const auto found = ids_.find(key); found != ids_.end()) {
        return found->second;
    }
    const uint32_t id = static_cast<uint32_t>(entries_.size());
    Entry entry;
    entry.info.path = path;
    entry.info.space = space;
    entry.info.udim = UsdShadeUdimUtils::IsUdimIdentifier(path);
    entries_.push_back(std::move(entry));
    ids_.emplace(key, id);
    return id;
}

uint32_t TextureStore::sampler(Wrap s, Wrap t, Filter filter) {
    const auto key = std::make_tuple(s, t, filter);
    if (const auto found = samplerIds_.find(key); found != samplerIds_.end()) {
        return found->second;
    }
    if (samplers_.size() >= kSamplerSlots) {
        log::warn("materials: more than {} sampler modes; the rest share the first", kSamplerSlots);
        return 0;
    }
    rhi::SamplerDesc desc;
    desc.addressU = addressOf(s);
    desc.addressV = addressOf(t);
    const rhi::TextureFilteringMode mode =
        filter == Filter::Nearest ? rhi::TextureFilteringMode::Point : rhi::TextureFilteringMode::Linear;
    desc.minFilter = mode;
    desc.magFilter = mode;
    desc.mipFilter = rhi::TextureFilteringMode::Linear;
    desc.maxAnisotropy = 8;
    auto made = gpu::Sampler::create(*device_, desc);
    if (!made) {
        log::warn("materials: {}", made.error().toString());
        return 0;
    }
    samplers_.push_back(std::move(*made));
    const uint32_t id = static_cast<uint32_t>(samplers_.size() - 1);
    samplerIds_.emplace(key, id);
    return id;
}

Result<uint32_t> TextureStore::loadFile(const std::string& path, ColourSpace space, TextureInfo& info) {
    if (slots_.size() >= kTextureSlots) {
        return Error::make(ErrorCode::OutOfMemory, "more than {} texture files", kTextureSlots);
    }
    const HioImage::SourceColorSpace source = space == ColourSpace::Raw    ? HioImage::Raw
                                              : space == ColourSpace::Srgb ? HioImage::SRGB
                                                                           : HioImage::Auto;
    HioImageSharedPtr image = HioImage::OpenForReading(path, 0, 0, source, /*suppressErrors=*/true);
    if (!image) {
        return Error::make(ErrorCode::IoFailure, "cannot read image '{}'", path);
    }
    Layout layout;
    if (!layoutOf(image->GetFormat(), layout)) {
        return Error::make(ErrorCode::Unsupported, "image '{}': pixel format {} not read", path,
                           static_cast<int>(image->GetFormat()));
    }
    // What the file says, unless the material said otherwise.
    const bool srgb = space == ColourSpace::Srgb || (space == ColourSpace::Auto && layout.srgb);
    const uint32_t w = static_cast<uint32_t>(image->GetWidth());
    const uint32_t h = static_cast<uint32_t>(image->GetHeight());
    const size_t bytes = size_t{w} * h * layout.channels * layout.componentBytes;
    std::vector<uint32_t> words((bytes + 3) / 4, 0);
    HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w);
    spec.height = static_cast<int>(h);
    spec.depth = 1;
    spec.format = image->GetFormat();
    spec.flipped = false;
    spec.data = words.data();
    if (!image->Read(spec)) {
        return Error::make(ErrorCode::IoFailure, "cannot decode image '{}'", path);
    }
    info.width = w;
    info.height = h;

    // 8-bit sRGB stays 8-bit behind an sRGB view; anything else sRGB is
    // decoded to light; floats keep their precision. A device whose store
    // into an 8-bit texture does not convert (CUDA) holds 8-bit images as
    // half floats, decoded to light like the rest.
    const bool eightBit = layout.component == 0 && device_->caps().unormStores;
    gpu::TextureDesc desc;
    desc.width = w;
    desc.height = h;
    desc.mipCount = 0;
    desc.format = eightBit ? rhi::Format::RGBA8Unorm
                  : layout.component == 3 ? rhi::Format::RGBA32Float
                                          : rhi::Format::RGBA16Float;
    desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::UnorderedAccess |
                 rhi::TextureUsage::CopySource;
    desc.label = path;
    auto texture = gpu::Texture::create(*device_, desc);
    if (!texture) return std::move(texture).error();
    gpu::BufferDesc upload;
    upload.bytes = std::max<uint64_t>(words.size(), 1) * 4;
    upload.elementBytes = 4;
    upload.label = "texture.bytes";
    auto buffer = gpu::Buffer::create(*device_, upload, words.data());
    if (!buffer) return std::move(buffer).error();
    auto level0 = texture->view(0);
    if (!level0) return std::move(level0).error();
    gpu::CommandBatch batch(*device_);
    decode_.dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["bytes"].setBinding(buffer->rhi());
        cursor["level"].setBinding((*level0).get());
        rhi::ShaderCursor p = cursor["params"];
        p["width"].setData(w);
        p["height"].setData(h);
        p["channels"].setData(layout.channels);
        p["component"].setData(layout.component);
        p["toLinear"].setData(uint32_t{srgb && !eightBit ? 1u : 0u});
    });
    LRT_TRY(mips_->generate(batch, *texture, srgb && eightBit));
    LRT_TRY(batch.submit(true));

    Slot slot;
    if (srgb && eightBit) {
        rhi::TextureViewDesc view;
        view.format = rhi::Format::RGBA8UnormSrgb;
        if (SLANG_FAILED(device_->rhi()->createTextureView(texture->rhi(), view, slot.view.writeRef()))) {
            return Error::make(ErrorCode::DeviceFailure, "no sRGB view of '{}'", path);
        }
    } else {
        rhi::TextureViewDesc view;
        if (SLANG_FAILED(device_->rhi()->createTextureView(texture->rhi(), view, slot.view.writeRef()))) {
            return Error::make(ErrorCode::DeviceFailure, "no view of '{}'", path);
        }
    }
    slot.texture = std::move(*texture);
    slots_.push_back(std::move(slot));
    return static_cast<uint32_t>(slots_.size() - 1);
}

Result<size_t> TextureStore::commit() {
    size_t loaded = 0;
    bool changed = false;
    for (Entry& entry : entries_) {
        if (!entry.pending) {
            continue;
        }
        entry.pending = false;
        changed = true;
        if (entry.info.udim) {
            entry.tiles.assign(100, 0);
            for (const auto& [tilePath, tile] : UsdShadeUdimUtils::ResolveUdimTilePaths(entry.info.path, SdfLayerHandle())) {
                const int number = std::atoi(tile.c_str());
                if (number < 1001 || number > 1100) {
                    continue;
                }
                auto slot = loadFile(tilePath, entry.info.space, entry.info);
                if (!slot) {
                    log::warn("materials: {}", slot.error().toString());
                    continue;
                }
                entry.tiles[static_cast<size_t>(number - 1001)] = *slot + 1;
                entry.info.loaded = true;
                ++loaded;
            }
            if (!entry.info.loaded) {
                entry.info.error = "no UDIM tiles found";
                log::warn("materials: '{}': no UDIM tiles found", entry.info.path);
            }
            continue;
        }
        auto slot = loadFile(entry.info.path, entry.info.space, entry.info);
        if (!slot) {
            entry.info.error = slot.error().toString();
            log::warn("materials: {}", entry.info.error);
            continue;
        }
        entry.slot = *slot;
        entry.info.loaded = true;
        ++loaded;
    }
    if (changed) {
        LRT_TRY(writeRecords());
    }
    return loaded;
}

Result<void> TextureStore::writeRecords() {
    std::vector<TextureRecord> records(std::max<size_t>(entries_.size(), 1));
    std::vector<uint32_t> cells;
    for (size_t k = 0; k < entries_.size(); ++k) {
        Entry& entry = entries_[k];
        TextureRecord& record = records[k];
        record.width = static_cast<float>(entry.info.width);
        record.height = static_cast<float>(entry.info.height);
        if (!entry.info.loaded) {
            continue;
        }
        record.flags = 1;
        if (entry.info.udim) {
            record.flags |= 2;
            entry.udimBase = static_cast<uint32_t>(cells.size());
            record.udimBase = entry.udimBase;
            cells.insert(cells.end(), entry.tiles.begin(), entry.tiles.end());
        } else {
            record.slot = entry.slot;
        }
    }
    cells.resize(std::max<size_t>(cells.size(), 1), 0);
    auto made = gpu::Buffer::fromSpan<TextureRecord>(*device_, records, "textures.records");
    if (!made) return std::move(made).error();
    records_ = std::move(*made);
    auto tiles = gpu::Buffer::fromSpan<uint32_t>(*device_, cells, "textures.udim");
    if (!tiles) return std::move(tiles).error();
    udim_ = std::move(*tiles);
    return ok();
}

void TextureStore::bind(rhi::ShaderCursor table) const {
    for (size_t k = 0; k < slots_.size(); ++k) {
        table["textures"][static_cast<uint32_t>(k)].setBinding(slots_[k].view.get());
    }
    for (size_t k = 0; k < kSamplerSlots; ++k) {
        table["samplers"][static_cast<uint32_t>(k)].setBinding(samplers_[k < samplers_.size() ? k : 0].rhi());
    }
    table["records"].setBinding(records_.rhi());
    table["udim"].setBinding(udim_.rhi());
}

}   // namespace lrt::material
