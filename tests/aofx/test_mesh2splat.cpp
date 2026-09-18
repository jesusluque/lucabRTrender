// Copyright (c) 2026 lucabRTrender contributors.
//
// Electronic Arts' conversion, as this repository runs it: a quad of known
// size in, and every gaussian that comes back where it should be, as wide as
// one cell of the projection grid, flat against the quad, and the colour of
// the texel it stands on.
//
// Everything the effect is fed is written by a kernel and everything it
// answers is checked by one (`shaders/lrt/test/mesh2splat_check.slang`); what
// crosses to the processor is five counters, which must all be zero.
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "aofx/Effect.h"
#include "lrt/aofx/EffectRegistry.h"
#include "lrt/aofx/EffectRender.h"
#include "lrt/gpu/Buffer.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/ComputeKernel.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/gpu_host/Context.h"
#include "lrt/gpu_host/ImageStorage.h"
#include "lrt/image/Image.h"

using namespace lrt;

namespace {

constexpr uint32_t kResolution = 16;
constexpr float    kSigma = 0.65F;
constexpr float    kFlatness = 0.1F;

/// A slang-rhi view of an image's own pixels, as the conversion's own host
/// takes one: the effect and the test write the same memory.
gpu::Buffer viewOf(gpu_host::Context& context, const image::ImagePtr& picture) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    REQUIRE(storage != nullptr);
    const uint64_t buffer = storage->bufferFor(picture->address());
    REQUIRE(buffer != 0);
    auto view = context.renderView(buffer, picture->sizeBytes(), 16, "test.image");
    REQUIRE(view);
    return std::move(*view);
}

image::ImagePtr pictureOf(int32_t width, int32_t height) {
    auto made = image::Image::create({0, 0, width, height});
    REQUIRE(made);
    return *made;
}

aofx::Effect* conversion(aofx_host::EffectRegistry& registry, gpu_host::Context* gpu) {
    registry.addSearchPath(LRT_AOFX_BUNDLES);
    registry.scan(gpu);
    return registry.find("tv.mediapro.aofx.mesh2splat");
}

void number(aofx_host::EffectJob& job, const char* name, double value) {
    job.params.push_back(aofx::ParamValue{name, {value}, {}});
}

}   // namespace

TEST_CASE("a quad becomes a gaussian a cell, flat, one cell wide", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sQuad");
        auto check = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sCheckQuad");
        REQUIRE(quad);
        REQUIRE(check);

        // The triangles: two of them, six entries each, in a picture 64
        // entries wide.
        constexpr uint32_t kMeshWidth = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const auto meshStride = static_cast<uint32_t>(mesh->stride());

        // Room for every cell of the grid and then some: the quad covers
        // `resolution` squared of them, and the cells the diagonal passes
        // through belong to both triangles.
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 4), 1);

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(meshStride);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();

        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "flatness", kFlatness);
        number(job, "opacity", 1.0);
        number(job, "writePbr", 0.0);
        job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
        job.params.push_back(aofx::ParamValue{"materialColour", {0.25, 0.5, 0.75}, {}});

        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        REQUIRE(counted->size() >= 4);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        // Every cell whose centre the quad covers, and no more: the grid is
        // `resolution` cells across the model's longest side, the quad is
        // that side, so that is `resolution` squared -- plus, at most, the
        // cells the diagonal runs through, which both triangles claim.
        CHECK(written >= kResolution * kResolution);
        CHECK(written <= kResolution * kResolution + kResolution);
        CHECK((*counted)[2] == 0.0F);   // no triangle without a frame
        CHECK((*counted)[3] == 0.0F);   // no cell left unwalked

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);

        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(uint32_t{4});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["resolution"].setData(kResolution);
            cursor["params"]["sigma"].setData(kSigma);
            cursor["params"]["flatness"].setData(kFlatness);
            cursor["params"]["tolerance"].setData(1.0e-3F);
            cursor["params"]["opacity"].setData(1.0F);
            const std::array<float, 4> colour{0.25F, 0.5F, 0.75F, 1.0F};
            const std::array<float, 4> axis{0.0F, 0.0F, 1.0F, 0.0F};
            cursor["params"]["colour"].setData(colour.data(), 16);
            cursor["params"]["axis"].setData(axis.data(), 16);
        });
        REQUIRE(second.submit(true));
        auto violations = counts->readAll<uint32_t>(library.device());
        REQUIRE(violations);
        CHECK((*violations)[0] == 0);   // every gaussian on the quad
        CHECK((*violations)[1] == 0);   // every one a cell wide, and flat
        CHECK((*violations)[2] == 0);   // every frame square to the quad
        CHECK((*violations)[3] == 0);   // every colour the material's
        CHECK((*violations)[4] == 0);   // every opacity the parameter's
    }));
}

TEST_CASE("a gaussian takes the colour of the texel it stands on", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sQuad");
        auto checker = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sChecker");
        auto check = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sCheckColour");
        REQUIRE(quad);
        REQUIRE(checker);
        REQUIRE(check);

        constexpr uint32_t kMeshWidth = 64;
        constexpr uint32_t kChecker = 8;
        constexpr int32_t  kMapSize = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const image::ImagePtr albedo = pictureOf(kMapSize, kMapSize);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const gpu::Buffer albedoView = viewOf(*gpu, albedo);
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 6), 1);

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        checker->dispatch(batch, {kMapSize, kMapSize, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["albedoWidth"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoHeight"].setData(static_cast<uint32_t>(kMapSize));
            cursor["params"]["albedoStride"].setData(static_cast<uint32_t>(albedo->stride()));
            cursor["params"]["checker"].setData(kChecker);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        albedo->deviceWrote();

        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        job.inputs.push_back({"Albedo", albedo});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "writePbr", 1.0);
        job.params.push_back(aofx::ParamValue{"materialColour", {1.0, 1.0, 1.0}, {}});

        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        REQUIRE(written >= kResolution * kResolution);
        // Six entries a record, since the texture coordinate the check reads
        // travels in the sixth.
        REQUIRE(counted->size() >= 5);
        REQUIRE((*counted)[4] == 6.0F);

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(albedoView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(uint32_t{6});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["tolerance"].setData(1.0e-2F);
            cursor["params"]["checker"].setData(kChecker);
        });
        REQUIRE(second.submit(true));
        auto violations = counts->readAll<uint32_t>(library.device());
        REQUIRE(violations);
        CHECK((*violations)[3] == 0);
    }));
}

TEST_CASE("glass keeps its material's opacity and takes its transmission colour", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sQuad");
        auto check = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sCheckQuad");
        REQUIRE(quad);
        REQUIRE(check);
        constexpr uint32_t kMeshWidth = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const uint32_t budget = kResolution * kResolution * 2;
        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 4), 1);

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();

        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "flatness", kFlatness);
        number(job, "opacity", 1.0);
        number(job, "writePbr", 0.0);
        // Glass: fully transmitting. What the conversion writes is what the
        // material says -- its opacity, its colour taken towards the
        // transmission colour, and the transmission in a channel of its own.
        // What a renderer makes of that is the renderer's.
        number(job, "transmission", 1.0);
        job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
        job.params.push_back(aofx::ParamValue{"materialColour", {1.0, 1.0, 1.0}, {}});
        job.params.push_back(aofx::ParamValue{"transmissionColour", {0.2, 0.5, 0.4}, {}});

        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        REQUIRE(written >= kResolution * kResolution);

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "test.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(uint32_t{4});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["resolution"].setData(kResolution);
            cursor["params"]["sigma"].setData(kSigma);
            cursor["params"]["flatness"].setData(kFlatness);
            cursor["params"]["tolerance"].setData(1.0e-3F);
            cursor["params"]["opacity"].setData(1.0F);
            const std::array<float, 4> colour{0.2F, 0.5F, 0.4F, 1.0F};
            const std::array<float, 4> axis{0.0F, 0.0F, 1.0F, 0.0F};
            cursor["params"]["colour"].setData(colour.data(), 16);
            cursor["params"]["axis"].setData(axis.data(), 16);
        });
        REQUIRE(second.submit(true));
        auto violations = counts->readAll<uint32_t>(library.device());
        REQUIRE(violations);
        CHECK((*violations)[3] == 0);   // the transmission colour, multiplied in
        CHECK((*violations)[4] == 0);   // and the opacity the material gave it
    }));
}

// THE SAME MESH MUST GIVE THE SAME ARRAY, not the same set in another order.
//
// The slot used to be taken with an atomic and the shader said so: "the order
// of the output is nobody's business". Two conversions of the chess pawn then
// wrote files that differed from the thousandth byte, which is fine for one
// still frame and impossible for a sequence -- a gaussian is followed from one
// pose to the next by being the same element. Counting each triangle's cells,
// settling where each triangle starts, and then writing at that offset makes
// the order the mesh's own.
TEST_CASE("two conversions of one mesh write the same array, bit for bit", "[aofx][mesh2splat]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sQuad");
        auto compare = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_compare", "m2sCompare");
        REQUIRE(quad);
        REQUIRE(compare);

        constexpr uint32_t kMeshWidth = 64;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const uint32_t budget = kResolution * kResolution * 2;
        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});

        const auto convert = [&]() -> image::ImagePtr {
            const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 4), 1);
            aofx_host::EffectJob job;
            job.bounds = records->bounds();
            job.inputs.push_back({"Mesh", mesh});
            number(job, "triangles", 2);
            number(job, "resolution", kResolution);
            number(job, "maxSplats", budget);
            number(job, "flatness", kFlatness);
            number(job, "opacity", 1.0);
            number(job, "writePbr", 0.0);
            job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
            auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
            return rendered ? *rendered : image::ImagePtr{};
        };
        const image::ImagePtr one = convert();
        const image::ImagePtr two = convert();
        REQUIRE(one);
        REQUIRE(two);

        const gpu::Buffer viewOne = viewOf(*gpu, one);
        const gpu::Buffer viewTwo = viewOf(*gpu, two);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "compare.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        const auto entries = static_cast<uint32_t>(kResolution * kResolution * 4);
        gpu::CommandBatch second(library.device());
        compare->dispatch(second, {entries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["first"].setBinding(viewOne.rhi());
            cursor["second"].setBinding(viewTwo.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["entries"].setData(entries);
            cursor["params"]["width"].setData(static_cast<uint32_t>(one->bounds().width()));
            cursor["params"]["stride"].setData(static_cast<uint32_t>(one->stride()));
        });
        REQUIRE(second.submit(true));
        uint32_t seen[2] = {0, 0};
        REQUIRE(counts->read(library.device(), 0, sizeof(seen), seen));
        std::printf("  %u entries compared, %u differ\n", seen[0], seen[1]);
        CHECK(seen[0] == entries);
        CHECK(seen[1] == 0);
    }));
}

// A GAUSSIAN IS CARRIED BY THE JOINTS ITS TRIANGLE IS CARRIED BY.
//
// The conversion blends the three corners' influences by the barycentric
// coordinates of the cell the gaussian stands in, which is what interpolating
// the skin means. Where every corner names one joint with all of its weight,
// every blend is that joint with all of its weight whatever the coordinates
// are -- and that is the case that catches the plumbing: a picture that did
// not arrive, an entry addressed wrong, a weight left unnormalised.
TEST_CASE("a gaussian keeps the joints of the triangle it stands on", "[aofx][mesh2splat][skinning]") {
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) {
        SKIP("no gpe device");
    }
    aofx_host::EffectRegistry registry;
    aofx::Effect* effect = conversion(registry, gpu);
    if (effect == nullptr) {
        SKIP("the Mesh2Splat bundle is not built here");
    }
    gpu::ShaderLibrary library(gpu->deviceShared());

    REQUIRE(gpu->run([&] {
        auto quad = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sQuad");
        auto joints = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sInfluences");
        auto check = gpu::ComputeKernel::create(library, "lrt/test/mesh2splat_check", "m2sCheckInfluences");
        REQUIRE(quad);
        REQUIRE(joints);
        REQUIRE(check);

        constexpr uint32_t kMeshWidth = 64;
        constexpr float    kJoint = 3.0F;
        const image::ImagePtr mesh = pictureOf(kMeshWidth, 1);
        const image::ImagePtr carried = pictureOf(kMeshWidth, 1);
        const gpu::Buffer meshView = viewOf(*gpu, mesh);
        const gpu::Buffer carriedView = viewOf(*gpu, carried);
        const uint32_t budget = kResolution * kResolution * 2;

        gpu::CommandBatch batch(library.device());
        quad->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(meshView.rhi());
            cursor["counts"].setBinding(meshView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(mesh->stride()));
        });
        joints->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(carriedView.rhi());
            cursor["albedo"].setBinding(carriedView.rhi());
            cursor["records"].setBinding(carriedView.rhi());
            cursor["counts"].setBinding(carriedView.rhi());
            cursor["params"]["meshWidth"].setData(kMeshWidth);
            cursor["params"]["meshStride"].setData(static_cast<uint32_t>(carried->stride()));
            cursor["params"]["joint"].setData(kJoint);
        });
        REQUIRE(batch.submit(true));
        mesh->deviceWrote();
        carried->deviceWrote();
        mesh->attach("bounds", {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F});

        const image::ImagePtr records = pictureOf(static_cast<int32_t>(budget * 8), 1);
        aofx_host::EffectJob job;
        job.bounds = records->bounds();
        job.inputs.push_back({"Mesh", mesh});
        job.inputs.push_back({"Influences", carried});
        number(job, "triangles", 2);
        number(job, "resolution", kResolution);
        number(job, "maxSplats", budget);
        number(job, "flatness", kFlatness);
        number(job, "opacity", 1.0);
        number(job, "writePbr", 1.0);
        number(job, "writeInfluences", 1.0);
        job.params.push_back(aofx::ParamValue{"sigma", {kSigma, kSigma}, {}});
        auto rendered = aofx_host::renderEffect(*gpu, *effect, job);
        REQUIRE(rendered);
        const std::vector<float>* counted = (*rendered)->attached("splats");
        REQUIRE(counted != nullptr);
        const auto written = static_cast<uint32_t>((*counted)[0]);
        REQUIRE(written >= kResolution * kResolution);
        // Eight entries a record, which is what the joints asked for.
        REQUIRE((*counted)[4] == 8.0F);

        const gpu::Buffer recordView = viewOf(*gpu, *rendered);
        gpu::BufferDesc desc;
        desc.bytes = 8 * 4;
        desc.elementBytes = 4;
        desc.label = "joints.counts";
        auto counts = gpu::Buffer::create(library.device(), desc);
        REQUIRE(counts);
        gpu::CommandBatch second(library.device());
        check->dispatch(second, {written, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["mesh"].setBinding(meshView.rhi());
            cursor["albedo"].setBinding(meshView.rhi());
            cursor["records"].setBinding(recordView.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["splats"].setData(written);
            cursor["params"]["recordPixels"].setData(uint32_t{8});
            cursor["params"]["dstWidth"].setData(static_cast<uint32_t>((*rendered)->bounds().width()));
            cursor["params"]["dstStride"].setData(static_cast<uint32_t>((*rendered)->stride()));
            cursor["params"]["tolerance"].setData(1.0e-5F);
            cursor["params"]["joint"].setData(kJoint);
        });
        REQUIRE(second.submit(true));
        uint32_t violations[4] = {0, 0, 0, 0};
        REQUIRE(counts->read(library.device(), 0, sizeof(violations), violations));
        std::printf("  %u gaussians carried; %u wrong joint, %u spilled, %u unnormalised\n", written,
                    violations[0], violations[1], violations[2]);
        CHECK(violations[0] == 0);   // every one on the joint its corners named
        CHECK(violations[1] == 0);   // and nothing in the other three slots
        CHECK(violations[2] == 0);   // and the weights a whole
    }));
}
