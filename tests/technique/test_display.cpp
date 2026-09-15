// Copyright (c) 2026 lucabRTrender contributors.
//
// The display transform against its formulas evaluated again in another
// kernel: every view transform and display, exposure and a background.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <tuple>

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

// ACES 2.0 (aces2.slang, the Academy's reference ported function for
// function, its tables built by kernels) against what it must hold to, in
// aces2_check.slang: a neutral stays neutral and its luminance is the
// tonescale's -- written a second time in the check, from the published
// constants -- and rises with the input; the inverse undoes the forward
// over the display cube; nothing lands outside the cube. At 100 nits
// limited to Rec.709, and at 1000 nits limited to P3, which is what an
// extended-range display asks for.
TEST_CASE("ACES 2.0 keeps neutrals on its tonescale, inverts over the display cube and stays inside it",
          "[technique][display][aces]") {
    LRT_REQUIRE_GPU(gpu);
    auto tables = technique::Aces2Tables::create(*gpu->library);
    if (!tables) FAIL(tables.error().toString());
    auto neutral = gpu::ComputeKernel::create(*gpu->library, "lrt/test/aces2_check", "aces2Neutral");
    auto roundTrip = gpu::ComputeKernel::create(*gpu->library, "lrt/test/aces2_check", "aces2RoundTrip");
    auto gamut = gpu::ComputeKernel::create(*gpu->library, "lrt/test/aces2_check", "aces2Gamut");
    if (!neutral) FAIL(neutral.error().toString());
    if (!roundTrip) FAIL(roundTrip.error().toString());
    if (!gamut) FAIL(gamut.error().toString());
    struct Case {
        float                    peak;
        technique::Aces2Limiting limiting;
        const char*              name;
    };
    const Case cases[] = {{100.0F, technique::Aces2Limiting::Rec709, "100 nits, Rec.709"},
                          {1000.0F, technique::Aces2Limiting::P3D65, "1000 nits, P3-D65"}};
    for (const Case& c : cases) {
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(tables->prepare(batch, c.peak, c.limiting));
            REQUIRE(batch.submit(true));
        }
        const auto run = [&](gpu::ComputeKernel& kernel, uint32_t count, float tolerance, uint32_t* out, float* worstOut) {
            gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "aces.counts");
            gpu::Buffer worst = test::uintBuffer(*gpu->device, 4, "aces.worst");
            gpu::CommandBatch batch(*gpu->device);
            kernel.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["check"]["count"].setData(count);
                cursor["check"]["peakLuminance"].setData(c.peak);
                cursor["check"]["tolerance"].setData(tolerance);
                cursor["params"].setBinding(tables->params().rhi());
                cursor["tables"].setBinding(tables->tables().rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
            });
            REQUIRE(batch.submit(true));
            REQUIRE(counts.read(*gpu->device, 0, 16, out));
            REQUIRE(worst.read(*gpu->device, 0, 16, worstOut));
        };
        uint32_t n[4] = {};
        float nw[4] = {};
        run(*neutral, 256, 2.0e-3F, n, nw);
        std::printf("  %s: neutrals: %u of 256 not neutral (worst spread %.2e), %u off the tonescale (worst %.2e), %u not "
                    "rising; 18%% grey shows %.5f of reference white\n",
                    c.name, n[0], static_cast<double>(nw[0]), n[1], static_cast<double>(nw[2]), n[2],
                    static_cast<double>(nw[1]));
        CHECK(n[0] == 0);
        CHECK(n[1] == 0);
        CHECK(n[2] == 0);
        if (c.peak == 100.0F) {
            CHECK(std::abs(nw[1] - 0.0999993F) < 2.0e-4F);
        }
        uint32_t r[4] = {};
        float rw[4] = {};
        run(*roundTrip, 8192, 5.0e-3F, r, rw);
        std::printf("  %s: inverse then forward over %u display colours: %u beyond 5e-3 of the peak (worst %.2e)\n",
                    c.name, 8192u, r[0], static_cast<double>(rw[0]));
        CHECK(r[0] == 0);
        // The gamut compression approximates the cube's boundary (a smooth
        // cusp, a hull gamma fitted at five points), so the most saturated
        // and brightest inputs can land a little outside it; the reference
        // clamps at the display encoding, and so does the display kernel.
        // Measured: 4.7% of these colours at 100 nits, worst 0.11 over;
        // 0.45% at 1000 nits.
        uint32_t g[4] = {};
        float gw[4] = {};
        run(*gamut, 8192, 1.0e-3F, g, gw);
        std::printf("  %s: %u of 8192 scene colours of any saturation land outside the display cube (worst %.2e)\n",
                    c.name, g[0], static_cast<double>(gw[0]));
        CHECK(g[0] < 8192 / 16);
        CHECK(gw[0] < 0.7F);
    }
}

// OpenColorIO's ACES 2.0 against ours. The studio config's "ACES 2.0 - SDR
// 100 nits (Rec.709)" on its sRGB display, from linear Rec.709, is OCIO's own
// implementation of the output transform -- its fixed functions, its tables,
// its Rec.709 to ACES2065-1 matrix -- compiled by the display into a Slang
// kernel; aces2.slang is a port of the reference CTL. Two implementations of
// one transform, each run on the device, agreeing pixel for pixel over the
// same sixteen stops of hues and coverages.
TEST_CASE("OpenColorIO's ACES 2.0 view, compiled into a kernel, agrees with aces2.slang",
          "[technique][display][ocio]") {
    LRT_REQUIRE_GPU(gpu);
    if (!technique::ocioBuilt()) {
        SKIP("no OpenColorIO in this build");
    }
    const uint32_t w = 257;
    const uint32_t h = 64;
    auto generate = gpu::ComputeKernel::create(*gpu->library, "lrt/test/display_check", "displayGenerate");
    auto agree = gpu::ComputeKernel::create(*gpu->library, "lrt/test/display_check", "displayAgree");
    auto display = technique::DisplayTransform::create(*gpu->library);
    if (!generate) FAIL(generate.error().toString());
    if (!agree) FAIL(agree.error().toString());
    if (!display) FAIL(display.error().toString());
    technique::OcioView ocio;
    ocio.display = "sRGB - Display";
    ocio.view = "ACES 2.0 - SDR 100 nits (Rec.709)";
    if (auto set = display->setOcio(ocio); !set) FAIL(set.error().toString());
    std::printf("  %s\n", display->ocioDescription().c_str());
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
    target.label = "display.aces2";
    auto ours = gpu::Texture::create(*gpu->device, target);
    target.label = "display.ocio";
    auto theirs = gpu::Texture::create(*gpu->device, target);
    REQUIRE(ours);
    REQUIRE(theirs);
    technique::DisplaySource from;
    from.kind = technique::DisplaySource::Kind::Colour;
    from.buffer = &*source;
    from.width = w;
    from.height = h;
    technique::DisplaySettings settings;
    settings.display = technique::DisplayEncoding::Srgb;
    settings.background = {0.05F, 0.1F, 0.2F};
    const auto measure = [&](float tolerance) {
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "display.counts");
        gpu::BufferDesc one;
        one.bytes = 4;
        one.elementBytes = 4;
        auto worst = gpu::Buffer::create(*gpu->device, one);
        REQUIRE(worst);
        auto a = ours->view(0);
        auto b = theirs->view(0);
        REQUIRE(a);
        REQUIRE(b);
        {
            gpu::CommandBatch batch(*gpu->device);
            agree->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["shown"].setBinding((*a).get());
                cursor["other"].setBinding((*b).get());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst->rhi());
                cursor["params"]["width"].setData(w);
                cursor["params"]["height"].setData(h);
                cursor["params"]["tolerance"].setData(tolerance);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t n[2] = {};
        float error = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
        REQUIRE(worst->read(*gpu->device, 0, sizeof(error), &error));
        return std::make_tuple(n[0], n[1], error);
    };
    for (const float exposure : {0.0F, 2.5F, -3.0F}) {
        settings.exposure = exposure;
        {
            gpu::CommandBatch batch(*gpu->device);
            settings.view = technique::ViewTransform::Aces2;
            REQUIRE(display->run(batch, from, settings, ours->rhi()));
            settings.view = technique::ViewTransform::Ocio;
            REQUIRE(display->run(batch, from, settings, theirs->rhi()));
            REQUIRE(batch.submit(true));
        }
        const auto [over, pixels, worst] = measure(1.0F / 255.0F);
        std::printf("  exposure %+.1f: %u of %u pixels differ by more than 1/255, worst %.2e\n", double(exposure), over,
                    pixels, double(worst));
        CHECK(pixels == w * h);
        CHECK(over == 0);
        // Measured 5.5e-4 at worst (at +2.5 stops): the two differ in how
        // they reach ACES2065-1 from Rec.709 and in float evaluation, not in
        // the transform.
        CHECK(worst < 1e-3F);
    }
    // And the control: the OCIO view is not ACES's by accident -- its
    // un-tone-mapped view differs.
    ocio.view = "Un-tone-mapped";
    if (auto set = display->setOcio(ocio); !set) FAIL(set.error().toString());
    settings.exposure = 0.0F;
    {
        gpu::CommandBatch batch(*gpu->device);
        settings.view = technique::ViewTransform::Aces2;
        REQUIRE(display->run(batch, from, settings, ours->rhi()));
        settings.view = technique::ViewTransform::Ocio;
        REQUIRE(display->run(batch, from, settings, theirs->rhi()));
        REQUIRE(batch.submit(true));
    }
    const auto [over, pixels, worst] = measure(1.0F / 255.0F);
    std::printf("  control, the un-tone-mapped view: %u of %u pixels differ by more than 1/255, worst %.2e\n", over,
                pixels, double(worst));
    CHECK(over > pixels / 4);
}
