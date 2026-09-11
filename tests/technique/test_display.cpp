// Copyright (c) 2026 lucabRTrender contributors.
//
// The display transform against its formulas evaluated again in another
// kernel: every view transform and display, exposure and a background.
#include "../gpu/GpuTest.h"

#include <cstdio>

#include "lrt/gpu/Texture.h"
#include "lrt/technique/DisplayTransform.h"

using namespace lrt;

TEST_CASE("the display transform is its formulas: Standard and AgX onto sRGB, Rec.709 and Display P3",
          "[technique][display]") {
    LRT_REQUIRE_GPU(gpu);
    const uint32_t w = 257;
    const uint32_t h = 64;
    auto generate = gpu::ComputeKernel::create(*gpu->library, "lrt/test/display_check", "displayGenerate");
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/display_check", "displayCheck");
    auto display = technique::DisplayTransform::create(*gpu->library);
    if (!generate) FAIL(generate.error().toString());
    if (!check) FAIL(check.error().toString());
    if (!display) FAIL(display.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{w} * h * 16;
    desc.elementBytes = 16;
    desc.label = "display.source";
    auto source = gpu::Buffer::create(*gpu->device, desc);
    REQUIRE(source);
    {
        gpu::CommandBatch batch(*gpu->device);
        generate->dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["source"].setBinding(source->rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
        });
        REQUIRE(batch.submit(true));
    }
    gpu::TextureDesc target;
    target.width = w;
    target.height = h;
    target.format = rhi::Format::RGBA32Float;
    target.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
    target.label = "display.shown";
    auto shown = gpu::Texture::create(*gpu->device, target);
    REQUIRE(shown);
    struct Case {
        technique::ViewTransform   view;
        technique::DisplayEncoding display;
        float                      exposure;
        const char*                name;
    };
    const Case cases[] = {
        {technique::ViewTransform::Standard, technique::DisplayEncoding::Srgb, 0.0F, "Standard, sRGB"},
        {technique::ViewTransform::Standard, technique::DisplayEncoding::Rec709, 1.5F, "Standard, Rec.709, +1.5"},
        {technique::ViewTransform::Standard, technique::DisplayEncoding::DisplayP3, -2.0F, "Standard, P3, -2"},
        {technique::ViewTransform::AgX, technique::DisplayEncoding::Srgb, 0.0F, "AgX, sRGB"},
        {technique::ViewTransform::AgX, technique::DisplayEncoding::DisplayP3, 3.0F, "AgX, P3, +3"},
        {technique::ViewTransform::AgX, technique::DisplayEncoding::Rec709, -4.0F, "AgX, Rec.709, -4"},
    };
    for (const Case& c : cases) {
        technique::DisplaySource from;
        from.kind = technique::DisplaySource::Kind::Colour;
        from.buffer = &*source;
        from.width = w;
        from.height = h;
        technique::DisplaySettings settings;
        settings.view = c.view;
        settings.display = c.display;
        settings.exposure = c.exposure;
        settings.background = {0.05F, 0.1F, 0.2F};
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(display->run(batch, from, settings, shown->rhi()));
            REQUIRE(batch.submit(true));
        }
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "display.counts");
        gpu::BufferDesc one;
        one.bytes = 4;
        one.elementBytes = 4;
        auto worst = gpu::Buffer::create(*gpu->device, one);
        REQUIRE(worst);
        auto view = shown->view(0);
        REQUIRE(view);
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["source"].setBinding(source->rhi());
                cursor["shown"].setBinding((*view).get());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["width"].setData(w);
                p["height"].setData(h);
                p["view"].setData(static_cast<uint32_t>(c.view));
                p["display"].setData(static_cast<uint32_t>(c.display));
                p["exposure"].setData(c.exposure);
                p["tolerance"].setData(2e-5F);
                p["backgroundR"].setData(settings.background[0]);
                p["backgroundG"].setData(settings.background[1]);
                p["backgroundB"].setData(settings.background[2]);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t n[2] = {};
        float error = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
        REQUIRE(worst->read(*gpu->device, 0, sizeof(error), &error));
        std::printf("  %s: %u of %u pixels beyond 2e-5, worst %.2e\n", c.name, n[0], n[1], double(error));
        CHECK(n[1] == w * h);
        CHECK(n[0] == 0);
    }
}
