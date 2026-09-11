// Copyright (c) 2026 lucabRTrender contributors.
//
// Textures, mips, raster and ray tracing kernels, generated and link-time
// specialised programs, the shader cache, and the HDR image comparison --
// checked by kernels, with only counters read back.
#include "GpuTest.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "lrt/gpu/RasterKernel.h"
#include "lrt/gpu/RayTracingKernel.h"
#include "lrt/gpu/Texture.h"
#include "lrt/gpu/algo/Mips.h"
#include "lrt/render/ReferenceRenderer.h"

using namespace lrt;

namespace {

gpu::Texture texture(test::Gpu& gpu, uint32_t width, uint32_t height, uint32_t mips, const char* label) {
    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.mipCount = mips;
    desc.format = rhi::Format::RGBA32Float;
    desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::UnorderedAccess |
                 rhi::TextureUsage::RenderTarget;
    desc.label = label;
    auto made = gpu::Texture::create(*gpu.device, desc);
    if (!made) FAIL(made.error().toString());
    return *made;
}

gpu::ComputeKernel kernelOf(test::Gpu& gpu, const char* entry) {
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "lrt/test/textures", entry);
    if (!kernel) FAIL(kernel.error().toString());
    return *kernel;
}

std::array<uint32_t, 2> check(test::Gpu& gpu, const gpu::Texture& t, uint32_t mode) {
    static gpu::ComputeKernel kCheck = kernelOf(gpu, "textureCheck");
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 2, "counts");
    auto view = t.view(0);
    REQUIRE(view);
    rhi::SamplerDesc desc;
    desc.minFilter = rhi::TextureFilteringMode::Linear;
    desc.magFilter = rhi::TextureFilteringMode::Linear;
    desc.addressU = rhi::TextureAddressingMode::ClampToEdge;
    desc.addressV = rhi::TextureAddressingMode::ClampToEdge;
    auto sampler = gpu::Sampler::create(*gpu.device, desc);
    REQUIRE(sampler);
    gpu::CommandBatch batch(*gpu.device);
    kCheck.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["texture"].setBinding((*view).get());
        cursor["linearClamp"].setBinding(sampler->rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["params"]["width"].setData(t.width());
        cursor["params"]["height"].setData(t.height());
        cursor["params"]["mode"].setData(mode);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 2> out{};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(out), out.data()));
    return out;
}

}   // namespace

TEST_CASE("a texture uploaded reads back through Load and through a sampler at texel centres", "[gpu][texture]") {
    LRT_REQUIRE_GPU(gpu);
    const uint32_t w = 37;
    const uint32_t h = 23;
    // The input, authored on the host as a file would be.
    std::vector<float> texels(size_t{w} * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const float* v = texels.data() + (size_t{y} * w + x) * 4;
            const_cast<float*>(v)[0] = static_cast<float>((x * 7 + y * 13) % 251) / 256.0F;
            const_cast<float*>(v)[1] = static_cast<float>((x * 3 + y * 5 + 1) % 241) / 256.0F;
            const_cast<float*>(v)[2] = static_cast<float>((x + y * 11 + 2) % 239) / 256.0F;
            const_cast<float*>(v)[3] = static_cast<float>((x * 5 + y + 3) % 233) / 256.0F;
        }
    }
    gpu::Texture t = texture(*gpu, w, h, 1, "roundtrip");
    REQUIRE(t.upload(*gpu->device, 0, 0, std::as_bytes(std::span<const float>(texels))));
    const auto loaded = check(*gpu, t, 0);
    const auto sampled = check(*gpu, t, 1);
    CHECK(loaded[1] == w * h);
    CHECK(loaded[0] == 0);
    CHECK(sampled[0] == 0);
}

TEST_CASE("every mip level keeps level 0's mean, whatever the sizes", "[gpu][texture][mips]") {
    LRT_REQUIRE_GPU(gpu);
    static gpu::ComputeKernel kFill = kernelOf(*gpu, "textureFill");
    static gpu::ComputeKernel kMean = kernelOf(*gpu, "textureMean");
    auto mips = gpu::MipGenerator::create(*gpu->library);
    if (!mips) FAIL(mips.error().toString());
    for (const auto [w, h] : {std::pair{64u, 64u}, std::pair{37u, 23u}, std::pair{1u, 9u}, std::pair{128u, 5u}}) {
        gpu::Texture t = texture(*gpu, w, h, 0, "mips");
        auto level0 = t.view(0);
        REQUIRE(level0);
        {
            gpu::CommandBatch batch(*gpu->device);
            kFill.dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["written"].setBinding((*level0).get());
                cursor["params"]["width"].setData(w);
                cursor["params"]["height"].setData(h);
            });
            REQUIRE(mips->generate(batch, t));
            REQUIRE(batch.submit(true));
        }
        std::vector<float> means;
        for (uint32_t mip = 0; mip < t.mipCount(); ++mip) {
            auto view = t.view(mip);
            REQUIRE(view);
            gpu::BufferDesc desc;
            desc.bytes = 4;
            desc.elementBytes = 4;
            auto mean = gpu::Buffer::create(*gpu->device, desc);
            REQUIRE(mean);
            gpu::CommandBatch batch(*gpu->device);
            kMean.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["texture"].setBinding((*view).get());
                cursor["mean"].setBinding(mean->rhi());
                cursor["params"]["width"].setData(t.width(mip));
                cursor["params"]["height"].setData(t.height(mip));
            });
            REQUIRE(batch.submit(true));
            float value = 0.0F;
            REQUIRE(mean->read(*gpu->device, 0, sizeof(value), &value));
            means.push_back(value);
        }
        std::printf("  %ux%u, %u levels: mean %.7f at 0, %.7f at the last\n", w, h, t.mipCount(),
                    static_cast<double>(means.front()), static_cast<double>(means.back()));
        CHECK(t.mipCount() == gpu::mipChain(w, h));
        for (const float m : means) {
            CHECK(std::abs(m - means.front()) <= 2e-6F);
        }
    }
}

TEST_CASE("a raster kernel covers exactly the pixels its triangles cover", "[gpu][raster]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    static gpu::ComputeKernel kCount = kernelOf(*gpu, "textureCountColour");
    gpu::RasterDesc desc;
    desc.module = "lrt/test/textures";
    desc.vertexEntry = "rasterHalfVertex";
    desc.fragmentEntry = "rasterHalfFragment";
    rhi::ColorTargetDesc target;
    target.format = rhi::Format::RGBA32Float;
    desc.targets = {target};
    auto kernel = gpu::RasterKernel::create(*gpu->library, desc);
    if (!kernel) FAIL(kernel.error().toString());
    const uint32_t w = 64;
    const uint32_t h = 40;
    gpu::Texture t = texture(*gpu, w, h, 1, "raster");
    auto view = t.view(0);
    REQUIRE(view);
    gpu::RasterPass pass;
    pass.width = w;
    pass.height = h;
    pass.colours = {(*view).get()};
    pass.clearColours = {{0.0F, 0.0F, 0.0F, 0.0F}};
    const std::array<float, 4> colour{0.25F, 0.5F, 0.75F, 1.0F};
    const gpu::RasterDraw draw{6, 1, 0, 0, [&](rhi::ShaderCursor cursor) {
                                   cursor["params"]["colourR"].setData(colour[0]);
                                   cursor["params"]["colourG"].setData(colour[1]);
                                   cursor["params"]["colourB"].setData(colour[2]);
                                   cursor["params"]["colourA"].setData(colour[3]);
                               }};
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    {
        gpu::CommandBatch batch(*gpu->device);
        kernel->run(batch, pass, std::span<const gpu::RasterDraw>(&draw, 1));
        REQUIRE(batch.submit(true));
    }
    auto read = t.view(0);
    REQUIRE(read);
    {
        gpu::CommandBatch batch(*gpu->device);
        kCount.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["texture"].setBinding((*read).get());
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
            cursor["params"]["colourR"].setData(colour[0]);
            cursor["params"]["colourG"].setData(colour[1]);
            cursor["params"]["colourB"].setData(colour[2]);
            cursor["params"]["colourA"].setData(colour[3]);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 2> covered{};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(covered), covered.data()));
    // The left half: pixel centres x + 0.5 < w / 2, every row.
    CHECK(covered[0] == (w / 2) * h);
}

TEST_CASE("ray tracing pipelines exist where the device has them, and say why not where it does not",
          "[gpu][raytracing]") {
    LRT_REQUIRE_GPU(gpu);
    gpu::RayTracingDesc desc;
    desc.module = "lrt/test/textures";
    desc.rayGen = "none";
    auto kernel = gpu::RayTracingKernel::create(*gpu->library, desc);
    if (!gpu->device->caps().rayTracing) {
        REQUIRE_FALSE(kernel);
        CHECK(kernel.error().code() == ErrorCode::Unsupported);
    } else {
        SUCCEED("pipelines on this device: covered by the path tracer's tests");
    }
}

TEST_CASE("link-time constants and generated modules make distinct programs", "[gpu][shaders]") {
    LRT_REQUIRE_GPU(gpu);
    gpu::Buffer out = test::uintBuffer(*gpu->device, 1, "linked");
    const auto run = [&](const std::shared_ptr<const gpu::Program>& program) {
        rhi::ComputePipelineDesc desc;
        desc.program = program->program.get();
        rhi::ComPtr<rhi::IComputePipeline> pipeline;
        REQUIRE(SLANG_SUCCEEDED(gpu->device->rhi()->createComputePipeline(desc, pipeline.writeRef())));
        gpu::CommandBatch batch(*gpu->device);
        rhi::IComputePassEncoder* pass = batch.encoder()->beginComputePass();
        rhi::ShaderCursor cursor(pass->bindPipeline(pipeline.get()));
        cursor["linkedOut"].setBinding(out.rhi());
        pass->dispatchCompute(1, 1, 1);
        pass->end();
        batch.markDirty();
        REQUIRE(batch.submit(true));
        uint32_t value = 0;
        REQUIRE(out.read(*gpu->device, 0, sizeof(value), &value));
        return value;
    };
    for (const uint32_t value : {7u, 9u}) {
        auto program = gpu->library->load("lrt/test/linked", {"linkedConstant"},
                                          {{"uint", "kLinkedValue", std::to_string(value)}});
        if (!program) FAIL(program.error().toString());
        CHECK(run(*program) == value);
    }
    const std::string source = "RWStructuredBuffer<uint> linkedOut;\n"
                               "[shader(\"compute\")] [numthreads(1, 1, 1)]\n"
                               "void generated(uint3 tid: SV_DispatchThreadID) { linkedOut[0] = 1234; }\n";
    auto generated = gpu->library->loadSource("lrt_test_generated", source, {"generated"});
    if (!generated) FAIL(generated.error().toString());
    CHECK(run(*generated) == 1234);
    auto clash = gpu->library->loadSource("lrt_test_generated", source + "// other\n", {"generated"});
    CHECK_FALSE(clash);
}

TEST_CASE("compiled shaders come back from the disk cache on a second device", "[gpu][shaders][cache]") {
    LRT_REQUIRE_GPU(gpu);
    const std::filesystem::path cache = std::filesystem::temp_directory_path() / "lrt_test_shader_cache";
    std::filesystem::remove_all(cache);
    const auto compile = [&]() {
        gpu::DeviceDesc desc;
        desc.shaderCache = cache;
        auto device = gpu::Device::create(desc);
        REQUIRE(device);
        gpu::ShaderLibrary library(*device);
        auto kernel = gpu::ComputeKernel::create(library, "lrt/test/textures", "textureMean");
        REQUIRE(kernel);
        return (*device)->shaderCacheStats();
    };
    const gpu::ShaderCacheStats first = compile();
    const gpu::ShaderCacheStats second = compile();
    std::printf("  shader cache: first %llu hits %llu writes, second %llu hits\n",
                static_cast<unsigned long long>(first.hits), static_cast<unsigned long long>(first.writes),
                static_cast<unsigned long long>(second.hits));
    CHECK(first.writes > 0);
    CHECK(second.hits > 0);
    std::filesystem::remove_all(cache);
}

TEST_CASE("HDR images and ID buffers compare on the device", "[gpu][compare]") {
    LRT_REQUIRE_GPU(gpu);
    const uint32_t w = 50;
    const uint32_t h = 30;
    std::vector<float> a(size_t{w} * h * 4);
    std::vector<float> b(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = 0.5F + static_cast<float>(i % 97) * 0.25F;   // up to 24.5: past what 8 bits show
        b[i] = a[i] * 1.1F;
    }
    gpu::BufferDesc desc;
    desc.bytes = a.size() * 4;
    desc.elementBytes = 16;
    auto bufferA = gpu::Buffer::create(*gpu->device, desc, a.data());
    auto bufferB = gpu::Buffer::create(*gpu->device, desc, b.data());
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    auto same = render::compareHdr(*gpu->library, *bufferA, *bufferA, w, h);
    REQUIRE(same);
    CHECK(same->pixels == uint64_t{w} * h);
    CHECK(same->relMse == 0.0);
    CHECK(same->maxRelative == 0.0);
    auto scaled = render::compareHdr(*gpu->library, *bufferA, *bufferB, w, h);
    REQUIRE(scaled);
    std::printf("  a against 1.1 a: relMSE %.5f, p99 relative %.4f, max %.4f\n", scaled->relMse,
                scaled->p99Relative, scaled->maxRelative);
    // |a - 1.1a| / 1.1a = 1/11 = 0.0909..., within one bin (9%) above.
    CHECK(scaled->p99Relative >= 1.0 / 11.0);
    CHECK(scaled->p99Relative <= 1.0 / 11.0 * 1.0906);

    std::vector<uint32_t> ids(10000);
    std::vector<uint32_t> other(10000);
    for (uint32_t i = 0; i < ids.size(); ++i) {
        ids[i] = i * 2654435761u;
        other[i] = (i % 37 == 0) ? ids[i] + 1 : ids[i];
    }
    auto idsA = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, ids, "ids.a");
    auto idsB = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, other, "ids.b");
    REQUIRE(idsA);
    REQUIRE(idsB);
    auto differing = render::countDifferent(*gpu->library, *idsA, *idsB, static_cast<uint32_t>(ids.size()));
    REQUIRE(differing);
    CHECK(*differing == (10000 + 36) / 37);
}
