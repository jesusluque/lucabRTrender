// A body slab under a wing slab, one light from above. The per-part fields
// are baked once in that pose; then the wing's joint is moved aside without
// baking again, and the body must come out of shadow. Every input is built
// by the fixtures, every answer is counted by a kernel.
#include "../gpu/GpuTest.h"
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include "../render/SplatFixtures.h"
#include "lrt/light/LightTable.h"
#include "lrt/scene/GpuClouds.h"
#include "lrt/technique/SplatVisibility.h"

using namespace lrt;

namespace {

/// A slab of gaussians in the plane z = `z`, over x in [x0, x1), y in [-1, 1).
void slab(test::CloudBuilder& into, float z, float x0, float x1, uint32_t across, float size) {
    for (uint32_t i = 0; i < across; ++i) {
        for (uint32_t j = 0; j < across; ++j) {
            const float x = x0 + (x1 - x0) * (static_cast<float>(i) + 0.5F) / static_cast<float>(across);
            const float y = -1.0F + 2.0F * (static_cast<float>(j) + 0.5F) / static_cast<float>(across);
            into.add(x, y, z, 0.95F, size, size, size * 0.05F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
        }
    }
}

struct Counts {
    uint32_t underLit = 0, clearShadowed = 0, under = 0, clear = 0;
};

}   // namespace

TEST_CASE("a wing's field shadows the body under it, and moves with the wing's joint",
          "[technique][gpu][splatvisibility]") {
    LRT_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !caps.rayQuery) {
        SKIP("the bake traces inline: no ray queries here");
    }
    auto loader = scene::CloudLoader::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    auto visibility = technique::SplatVisibility::create(*gpu->library);
    if (!visibility) FAIL(visibility.error().toString());
    auto lightTable = light::LightTable::create(*gpu->library);
    if (!lightTable) FAIL(lightTable.error().toString());

    // The body: z = 0 over x in [-1, 1). The wing: z = 1 over x in [0, 1).
    constexpr uint32_t kAcross = 48;
    constexpr float kSize = 2.0F / kAcross * 0.9F;
    test::CloudBuilder built;
    slab(built, 0.0F, -1.0F, 1.0F, kAcross, kSize);
    const uint32_t bodyCount = kAcross * kAcross;
    slab(built, 1.0F, 0.0F, 1.0F, kAcross, kSize);
    auto cloud = loader->upload(built.raw, 0);
    REQUIRE(cloud);
    const uint32_t count = cloud->count;
    REQUIRE(count == bodyCount + kAcross * kAcross);

    // Joint 0 carries the body, joint 1 the wing: (joint, weight) pairs, four
    // a gaussian, which is the layout every skinner here reads.
    std::vector<float> pairs(size_t{count} * 8, 0.0F);
    for (uint32_t i = 0; i < count; ++i) {
        pairs[size_t{i} * 8] = i < bodyCount ? 0.0F : 1.0F;
        pairs[size_t{i} * 8 + 1] = 1.0F;
    }
    auto influences = gpu::Buffer::fromSpan<float>(*gpu->device, pairs, "test.influences");
    REQUIRE(influences);

    technique::VisibilityParts parts;
    parts.jointToPart = {0, 1};
    parts.partJoint = {0, 1};
    const uint32_t grid = 16;
    // Octave 8 is a map texel of some 22 degrees: its penumbra at the wing's
    // height is wider than the wing, and there is nothing left to count.
    // 16 and 32 leave a band of a cell plus tan(pitch) either side.
    const uint32_t octave = GENERATE(16u, 32u);
    technique::VisibilityBakeOptions options;
    options.grid = grid;
    options.octave = octave;
    INFO("grid " << grid << " octave " << octave);
    auto baked = visibility->bake(*cloud, *influences, 4, parts, options);
    if (!baked) FAIL(baked.error().toString());
    REQUIRE(cloud->hasVisibility());

    // One distant light straight down: UsdLux's distant light shines along
    // its own -z, and the identity frame keeps that pointing at the slabs.
    light::Light sun;
    sun.kind = light::LightKind::Distant;
    sun.intensity = 1.0F;
    REQUIRE(lightTable->set(std::span<const light::Light>(&sun, 1), 4.0F));

    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/visibility_parts_check", "vpCount");
    if (!check) FAIL(check.error().toString());
    gpu::BufferDesc factorDesc;
    factorDesc.bytes = uint64_t{count} * 4;
    factorDesc.elementBytes = 4;
    factorDesc.label = "test.factors";
    auto factorBuffer = gpu::Buffer::create(*gpu->device, factorDesc);
    REQUIRE(factorBuffer);
    gpu::Buffer& factors = *factorBuffer;

    const auto measure = [&](const gpu::Buffer* xforms) -> Counts {
        gpu::Buffer counters = test::uintBuffer(*gpu->device, 4, "test.counts");
        gpu::CommandBatch batch(*gpu->device);
        technique::VisibilityFactorsJob job;
        job.cloud = &*cloud;
        job.positions = &cloud->positions;
        job.skinningXforms = xforms;
        job.lights = &lightTable->records();
        job.lightCount = 1;
        job.base = 0;
        job.categories = ~uint64_t{0};
        job.objectToWorld = render::Mat4::identity().rows3x4();
        REQUIRE(visibility->factors(batch, job, factors));
        check->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["factors"].setBinding(factors.rhi());
            cursor["counts"].setBinding(counters.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["lights"].setData(1u);
            p["light"].setData(0u);
            p["cut"].setData(0.5F);
            // Either side of an edge is penumbra and not counted: a probe
            // cell of position blur, and the map's angular pitch times the
            // wing's height of direction blur. Inside those bands the answer
            // must be exact.
            const float cell = (2.0F + 2.0F * 0.05F * 2.0F) / static_cast<float>(grid);
            const float pitch = 3.14159265F / static_cast<float>(octave);   // radians a map texel, about
            p["wingFromX"].setData(0.0F);
            p["wingToX"].setData(1.0F);
            p["band"].setData(cell + 1.0F * std::tan(pitch));
            p["bodyZ"].setData(0.0F);
        });
        REQUIRE(batch.submit(true));
        Counts c;
        REQUIRE(counters.read(*gpu->device, 0, sizeof(c), &c));
        return c;
    };

    SECTION("in the pose it was baked in, the wing shadows exactly the body under it") {
        const Counts c = measure(nullptr);
        INFO("under " << c.under << " lit " << c.underLit << "; clear " << c.clear << " shadowed "
                      << c.clearShadowed);
        REQUIRE(c.under > 0);
        REQUIRE(c.clear > 0);
        CHECK(c.underLit == 0);
        CHECK(c.clearShadowed == 0);
    }

    SECTION("the wing's joint moved aside, its field moves with it and the body is lit again") {
        // Joint 0 stays; joint 1 is translated +3 in x, out over nothing.
        const std::vector<float> xforms{1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
                                        1, 0, 0, 3,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};
        auto moved = gpu::Buffer::fromSpan<float>(*gpu->device, xforms, "test.xforms");
        REQUIRE(moved);
        const Counts c = measure(&*moved);
        INFO("under " << c.under << " lit " << c.underLit << "; clear " << c.clear << " shadowed "
                      << c.clearShadowed);
        // Every body gaussian is clear of the wing now: none may stay shadowed.
        CHECK(c.underLit == c.under);
        CHECK(c.clearShadowed == 0);
    }
}

TEST_CASE("a rig becomes parts by its largest subtrees, small ones staying with their parent",
          "[technique][splatvisibility][partition]") {
    // root(12) -> { A(6) -> { A1(3) -> {A1a, A1b}, A2(2) -> {A2a} }, B(4) -> {B1, B2, B3}, C(1) }
    const std::vector<std::string> paths{"root",       "root/A",     "root/A/A1", "root/A/A1/A1a",
                                         "root/A/A1/A1b", "root/A/A2", "root/A/A2/A2a", "root/B",
                                         "root/B/B1",  "root/B/B2",  "root/B/B3", "root/C"};
    SECTION("one part is everything") {
        const technique::VisibilityParts p = technique::partitionJoints(paths, 1, 2);
        REQUIRE(p.partJoint.size() == 1);
        CHECK(p.partJoint[0] == 0);
        for (uint32_t part : p.jointToPart) CHECK(part == 0);
    }
    SECTION("three parts: the root, A and B; C is too small and stays with the root") {
        const technique::VisibilityParts p = technique::partitionJoints(paths, 3, 2);
        REQUIRE(p.partJoint.size() == 3);
        CHECK(p.partJoint[0] == 0);   // root
        CHECK(p.partJoint[1] == 1);   // A, the biggest subtree, split off first
        CHECK(p.partJoint[2] == 7);   // then B
        CHECK(p.jointToPart[0] == 0);
        CHECK(p.jointToPart[11] == 0);  // C with the root
        CHECK(p.jointToPart[1] == 1);
        CHECK(p.jointToPart[4] == 1);   // A1b under A
        CHECK(p.jointToPart[8] == 2);   // B1 under B
    }
    SECTION("five parts: A1 and A2 come out of A next, biggest first") {
        const technique::VisibilityParts p = technique::partitionJoints(paths, 5, 2);
        REQUIRE(p.partJoint.size() == 5);
        CHECK(p.partJoint[3] == 2);   // A1 (3 joints)
        CHECK(p.partJoint[4] == 5);   // A2 (2 joints)
        CHECK(p.jointToPart[3] == 3);   // A1a under A1
        CHECK(p.jointToPart[6] == 4);   // A2a under A2
        CHECK(p.jointToPart[1] == 1);   // A itself keeps its own part
    }
}
