// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/gpu/algo/Mips.h"

#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/gpu/Texture.h"

namespace lrt::gpu {

Result<MipGenerator> MipGenerator::create(ShaderLibrary& library) {
    auto kernel = ComputeKernel::create(library, "lrt/algo/mips", "mipDownsample");
    if (!kernel) return std::move(kernel).error();
    MipGenerator mips;
    mips.downsample_ = std::move(*kernel);
    return mips;
}

Result<void> MipGenerator::generate(CommandBatch& batch, const Texture& texture) const {
    if (texture.desc().type != rhi::TextureType::Texture2D || texture.desc().arrayLength != 1) {
        return Error::make(ErrorCode::Unsupported, "mips of '{}': 2D textures only", texture.desc().label);
    }
    for (uint32_t mip = 1; mip < texture.mipCount(); ++mip) {
        auto source = texture.view(mip - 1);
        if (!source) return std::move(source).error();
        auto target = texture.view(mip);
        if (!target) return std::move(target).error();
        downsample_.dispatch(batch, {texture.width(mip), texture.height(mip), 1}, [&](rhi::ShaderCursor cursor) {
            cursor["source"].setBinding((*source).get());
            cursor["target"].setBinding((*target).get());
            cursor["params"]["sourceWidth"].setData(texture.width(mip - 1));
            cursor["params"]["sourceHeight"].setData(texture.height(mip - 1));
            cursor["params"]["width"].setData(texture.width(mip));
            cursor["params"]["height"].setData(texture.height(mip));
        });
    }
    return ok();
}

}   // namespace lrt::gpu
