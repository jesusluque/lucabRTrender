// Copyright (c) 2026 lucabRTrender contributors.
//
// Meshes through the visibility pass and their first shading, checked against
// what the geometry analytically is and against itself arranged another way.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <vector>

#include "lrt/geom/Mesh.h"
#include "lrt/render/ReferenceRenderer.h"
#include "lrt/technique/Visibility.h"
#include "lrt/world/GpuScene.h"
#include "lrt/world/Instancing.h"

#include <bit>
#include <cmath>

using namespace lrt;

namespace {

struct Renderer {
    geom::MeshBuilder           builder;
    world::GpuScene             scene;
    technique::VisibilityRaster raster;
    technique::HeadlightShading shading;
    technique::VisibilityTargets visibility;
};

std::unique_ptr<Renderer> renderer(test::Gpu& gpu) {
    auto builder = geom::MeshBuilder::create(*gpu.library);
    auto scene = world::GpuScene::create(*gpu.library);
    auto raster = technique::VisibilityRaster::create(*gpu.library);
    auto shading = technique::HeadlightShading::create(*gpu.library);
    if (!builder) FAIL(builder.error().toString());
    if (!scene) FAIL(scene.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!shading) FAIL(shading.error().toString());
    return std::unique_ptr<Renderer>(
        new Renderer{std::move(*builder), std::move(*scene), std::move(*raster), std::move(*shading), {}});
}

/// A unit square in z = 0, two triangles, facing +z.
std::shared_ptr<const geom::GpuMesh> square(Renderer& r, float half) {
    static std::vector<float> points;
    points = {-half, -half, 0, half, -half, 0, half, half, 0, -half, half, 0};
    static const std::vector<int32_t> counts{4};
    static const std::vector<int32_t> indices{0, 1, 2, 3};
    geom::MeshInput in;
    in.source = "square";
    in.points = {std::as_bytes(std::span<const float>(points)), false};
    in.faceVertexCounts = counts;
    in.faceVertexIndices = indices;
    in.smoothNormals = false;
    auto mesh = r.builder.build(in);
    if (!mesh) FAIL(mesh.error().toString());
    return std::make_shared<const geom::GpuMesh>(std::move(*mesh));
}

render::RenderTargets draw(test::Gpu& gpu, Renderer& r, std::span<const world::MeshInstance> instances,
                           const render::Projection& projection, uint32_t w, uint32_t h,
                           std::span<const world::InstanceSet> sets = {}) {
    REQUIRE(r.scene.update(instances, projection, sets));
    render::RenderTargets out;
    gpu::CommandBatch batch(*gpu.device);
    REQUIRE(r.raster.render(batch, r.scene, projection, w, h, r.visibility));
    REQUIRE(r.shading.shade(batch, r.scene, r.visibility, projection, out));
    REQUIRE(batch.submit(true));
    return out;
}

/// IEEE half bits of a float in the half's range: how a USD half array
/// stores its values (authoring input, as a file would).
uint16_t halfBits(float value) {
    const uint32_t f = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int exponent = static_cast<int>((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f & 0x7FFFFFu;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    mantissa = (mantissa + 0x1000u) >> 13;   // 23 bits to 10, rounded
    if (mantissa == 0x400u) {
        return static_cast<uint16_t>(sign | static_cast<uint32_t>(exponent + 1) << 10);
    }
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(exponent) << 10 | mantissa);
}

}   // namespace

TEST_CASE("a square drawn through visibility and shading is where, how deep and how lit it analytically is",
          "[technique][visibility]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    auto r = renderer(*gpu);
    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    world::MeshInstance instance;
    instance.mesh = square(*r, 1.0F);
    instance.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0});
    instance.displayColor = {0.8F, 0.4F, 0.2F};
    const render::RenderTargets out = draw(*gpu, *r, std::span<const world::MeshInstance>(&instance, 1), projection, w, h);

    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/visibility_check", "planeCheck");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "counts");
    gpu::BufferDesc one;
    one.bytes = 4;
    one.elementBytes = 4;
    auto worst = gpu::Buffer::create(*gpu->device, one);
    REQUIRE(worst);
    gpu::CommandBatch batch(*gpu->device);
    check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst->rhi());
        technique::setCamera(cursor["camera"], projection, w, h);
        cursor["plane"]["z"].setData(5.0F);
        cursor["plane"]["half"].setData(1.0F);
        cursor["plane"]["colourR"].setData(0.8F);
        cursor["plane"]["colourG"].setData(0.4F);
        cursor["plane"]["colourB"].setData(0.2F);
    });
    REQUIRE(batch.submit(true));
    uint32_t c[3] = {0, 0, 0};
    float depthError = 0.0F;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(depthError), &depthError));
    std::printf("  square: %u pixels covered, %u coverage and %u colour mismatches, depth off by %.2e\n", c[2], c[0],
                c[1], static_cast<double>(depthError));
    CHECK(c[2] > 1000);
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
    CHECK(depthError < 1e-4F);
}

TEST_CASE("instances of one mesh draw as the same meshes built apart", "[technique][visibility][instancing]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    auto r = renderer(*gpu);
    const uint32_t w = 200;
    const uint32_t h = 150;
    render::Camera camera = render::Camera::lookingAt({1.0, 2.0, 9.0}, {0.0, 0.0, 0.0});
    const render::Projection projection = render::projectionFor(camera, w, h);
    const auto shared = square(*r, 0.7F);
    std::vector<world::MeshInstance> instanced;
    std::vector<world::MeshInstance> apart;
    for (int k = 0; k < 6; ++k) {
        world::MeshInstance i;
        i.objectToWorld = aofx::xform::translation({-3.0 + k * 1.2, 0.3 * k, -0.5 * k}) *
                          aofx::xform::rotationY(20.0 * k) * aofx::xform::rotationX(-10.0 * k);
        i.displayColor = {0.2F + 0.1F * k, 0.5F, 0.9F - 0.1F * k};
        i.mesh = shared;
        instanced.push_back(i);
        i.mesh = square(*r, 0.7F);   // the same data, a mesh of its own
        apart.push_back(i);
    }
    const render::RenderTargets a = draw(*gpu, *r, instanced, projection, w, h);
    const render::RenderTargets b = draw(*gpu, *r, apart, projection, w, h);
    CHECK(r->scene.meshCount() == 6);
    auto colour = render::compareHdr(*gpu->library, a.colour, b.colour, w, h);
    REQUIRE(colour);
    auto depth = render::countDifferent(*gpu->library, a.depth, b.depth, w * h);   // float bits compared as words
    REQUIRE(depth);
    std::printf("  six instances against six meshes: colour max relative %.2e, %llu depths differ\n",
                colour->maxRelative, static_cast<unsigned long long>(*depth));
    CHECK(colour->maxRelative == 0.0);
    CHECK(*depth == 0);
}

TEST_CASE("instancer levels composed on the device draw as the instances authored one by one",
          "[technique][visibility][instancing]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    auto r = renderer(*gpu);
    auto instancing = world::Instancing::create(*gpu->library);
    if (!instancing) FAIL(instancing.error().toString());
    const uint32_t w = 240;
    const uint32_t h = 180;
    const render::Projection projection =
        render::projectionFor(render::Camera::lookingAt({2.0, 6.0, 14.0}, {0.0, 0.0, 0.0}), w, h);
    const auto mesh = square(*r, 0.6F);

    // The inner instancer's primvars, as USD would hold them: translations
    // float, rotations half (GfQuath: ix iy iz real), scales float, transforms
    // double row-major -- four elements, of which the prototype takes 3, 0, 2.
    const std::vector<float> translations{-2, 0, 0, 0, 1, 0, 2, 0, 0, 0, -1, 1};
    const auto quat = [](double degrees, int axis) {
        const double half = degrees * 3.14159265358979 / 360.0;
        std::array<double, 4> q{0, 0, 0, std::cos(half)};
        q[static_cast<size_t>(axis)] = std::sin(half);
        return q;
    };
    const std::array<std::array<double, 4>, 4> quats{quat(90, 1), quat(45, 0), quat(-30, 2), quat(180, 1)};
    std::vector<uint16_t> rotations;
    for (const auto& q : quats) {
        for (const double v : q) {
            rotations.push_back(halfBits(static_cast<float>(v)));
        }
    }
    const std::vector<float> scales{1, 1, 1, 2, 1, 1, 1, 0.5F, 1, 1, 1, 1.5F};
    std::vector<double> transforms;
    for (int e = 0; e < 4; ++e) {
        // Identity with a translation in the last row (row-major, as GfMatrix4d).
        std::array<double, 16> m{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.25 * e, -0.5, 0.1 * e, 1};
        transforms.insert(transforms.end(), m.begin(), m.end());
    }
    const std::vector<int32_t> innerIndices{3, 0, 2};
    world::InstancerLevel inner;
    inner.indices = innerIndices;
    inner.translations = {std::as_bytes(std::span<const float>(translations)), false};
    inner.rotations = {std::as_bytes(std::span<const uint16_t>(rotations)), true};
    inner.scales = {std::as_bytes(std::span<const float>(scales)), false};
    inner.transforms = {std::as_bytes(std::span<const double>(transforms)), false, true};
    inner.instancerTransform = aofx::xform::translation({0.0, 0.5, 0.0});
    // The outer instancer places the inner one twice.
    const std::vector<float> outerTranslations{-3, 0, -2, 3, 1, 1};
    const std::vector<int32_t> outerIndices{1, 0};
    world::InstancerLevel outer;
    outer.indices = outerIndices;
    outer.translations = {std::as_bytes(std::span<const float>(outerTranslations)), false};
    outer.instancerTransform = aofx::xform::rotationY(15.0);

    const std::array<world::InstancerLevel, 2> levels{inner, outer};
    auto chain = instancing->compose(levels);
    if (!chain) FAIL(chain.error().toString());
    CHECK(chain->count == 6);
    world::InstanceSet set;
    set.mesh = mesh;
    set.chainRows = chain->rows;
    set.count = chain->count;
    set.prototype = aofx::xform::rotationX(-20.0);
    set.displayColor = {0.7F, 0.6F, 0.3F};

    // The same six, authored one by one: outer * inner * prototype.
    std::vector<world::MeshInstance> authored;
    for (const int32_t o : outerIndices) {
        const render::Mat4 outerM = outer.instancerTransform *
                                    aofx::xform::translation({outerTranslations[size_t(o) * 3],
                                                              outerTranslations[size_t(o) * 3 + 1],
                                                              outerTranslations[size_t(o) * 3 + 2]});
        for (const int32_t e : innerIndices) {
            const auto& q = quats[size_t(e)];
            render::Mat4 rotation = render::Mat4::identity();
            const double x = q[0], y = q[1], z = q[2], wq = q[3];
            const double rm[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * wq), 2 * (x * z + y * wq)},
                                     {2 * (x * y + z * wq), 1 - 2 * (x * x + z * z), 2 * (y * z - x * wq)},
                                     {2 * (x * z - y * wq), 2 * (y * z + x * wq), 1 - 2 * (x * x + y * y)}};
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    rotation.at(a, b) = rm[a][b];
                }
            }
            const render::Mat4 innerM =
                inner.instancerTransform *
                aofx::xform::translation({translations[size_t(e) * 3], translations[size_t(e) * 3 + 1],
                                          translations[size_t(e) * 3 + 2]}) *
                rotation *
                aofx::xform::scaling({scales[size_t(e) * 3], scales[size_t(e) * 3 + 1], scales[size_t(e) * 3 + 2]}) *
                aofx::xform::translation({0.25 * e, -0.5, 0.1 * e});
            world::MeshInstance i;
            i.mesh = mesh;
            i.objectToWorld = outerM * innerM * set.prototype;
            i.displayColor = set.displayColor;
            authored.push_back(i);
        }
    }
    const render::RenderTargets a = draw(*gpu, *r, authored, projection, w, h);
    const render::RenderTargets b = draw(*gpu, *r, {}, projection, w, h, std::span<const world::InstanceSet>(&set, 1));
    auto diff = render::compareHdr(*gpu->library, a.colour, b.colour, w, h);
    REQUIRE(diff);
    std::printf("  six nested instances against six authored: relMSE %.2e, p99 relative %.2e\n", diff->relMse,
                diff->p99Relative);
    // Half-precision rotations and float composition against double: the
    // same squares to the last few edge pixels (measured relMSE 1.2e-9).
    CHECK(diff->p99Relative <= 1e-3);
    CHECK(diff->relMse < 1e-6);
}

TEST_CASE("rays see the triangles the rasteriser sees", "[technique][visibility][raytracing]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization || !gpu->device->caps().rayQuery) {
        SKIP("needs both rasterisation and ray queries");
    }
    auto r = renderer(*gpu);
    auto rt = world::RayTracingScene::create(*gpu->library);
    auto trace = technique::VisibilityTrace::create(*gpu->library);
    if (!rt) FAIL(rt.error().toString());
    if (!trace) FAIL(trace.error().toString());
    const uint32_t w = 241;
    const uint32_t h = 181;
    const render::Projection projection =
        render::projectionFor(render::Camera::lookingAt({3.0, 4.0, 11.0}, {0.0, 0.0, 0.0}), w, h);
    // A bumpy grid and a scatter of squares, overlapping in depth.
    static std::vector<float> points;
    static std::vector<int32_t> counts;
    static std::vector<int32_t> indices;
    const int n = 32;
    for (int y = 0; y <= n; ++y) {
        for (int x = 0; x <= n; ++x) {
            const float fx = static_cast<float>(x) / n * 6.0F - 3.0F;
            const float fy = static_cast<float>(y) / n * 6.0F - 3.0F;
            points.insert(points.end(), {fx, 0.4F * std::sin(fx * 1.7F) * std::cos(fy * 1.3F), fy});
        }
    }
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int p = y * (n + 1) + x;
            counts.push_back(4);
            indices.insert(indices.end(), {p, p + n + 1, p + n + 2, p + 1});
        }
    }
    geom::MeshInput in;
    in.source = "grid";
    in.points = {std::as_bytes(std::span<const float>(points)), false};
    in.faceVertexCounts = counts;
    in.faceVertexIndices = indices;
    auto grid = r->builder.build(in);
    if (!grid) FAIL(grid.error().toString());
    std::vector<world::MeshInstance> instances;
    world::MeshInstance ground;
    ground.mesh = std::make_shared<const geom::GpuMesh>(std::move(*grid));
    instances.push_back(ground);
    const auto squareMesh = square(*r, 0.5F);
    for (int k = 0; k < 12; ++k) {
        world::MeshInstance i;
        i.mesh = squareMesh;
        i.objectToWorld = aofx::xform::translation({-2.5 + 0.45 * k, 0.8 + 0.1 * (k % 3), -2.0 + 0.35 * k}) *
                          aofx::xform::rotationY(37.0 * k) * aofx::xform::rotationX(25.0 * k) *
                          aofx::xform::scaling({k % 4 == 3 ? -1.0 : 1.0, 1.0, 1.0});
        i.doubleSided = k % 2 == 0;   // half single sided, some of those mirrored
        instances.push_back(i);
    }
    REQUIRE(r->scene.update(instances, projection));
    technique::VisibilityTargets rastered;
    technique::VisibilityTargets traced;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(r->raster.render(batch, r->scene, projection, w, h, rastered));
        REQUIRE(batch.submit(true));
    }
    REQUIRE(rt->build(r->scene));
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(trace->render(batch, *rt, projection, w, h, traced));
        REQUIRE(batch.submit(true));
    }
    // And through the compute BVHs, the route for devices without the hardware.
    auto bvh = world::BvhScene::create(*gpu->library);
    auto walk = technique::VisibilityBvh::create(*gpu->library);
    if (!bvh) FAIL(bvh.error().toString());
    if (!walk) FAIL(walk.error().toString());
    technique::VisibilityTargets walked;
    REQUIRE(bvh->build(r->scene));
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(walk->render(batch, r->scene, *bvh, projection, w, h, walked));
        REQUIRE(batch.submit(true));
    }
    auto compare = gpu::ComputeKernel::create(*gpu->library, "lrt/test/ids_compare", "idsCompare");
    if (!compare) FAIL(compare.error().toString());
    for (const auto& [name, other] : {std::pair{"rays", &traced}, std::pair{"compute BVH", &walked}}) {
        gpu::Buffer out = test::uintBuffer(*gpu->device, 4, "ids.counts");
        auto viewA = rastered.ids.view(0);
        auto viewB = other->ids.view(0);
        REQUIRE(viewA);
        REQUIRE(viewB);
        gpu::CommandBatch batch(*gpu->device);
        compare->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding((*viewA).get());
            cursor["b"].setBinding((*viewB).get());
            cursor["counts"].setBinding(out.rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
        });
        REQUIRE(batch.submit(true));
        uint32_t c[4] = {0, 0, 0, 0};
        REQUIRE(out.read(*gpu->device, 0, sizeof(c), c));
        std::printf("  raster against %s: %u of %u interior pixels differ; covered %u / %u\n", name, c[0], c[1],
                    c[2], c[3]);
        CHECK(c[1] > 1000);
        CHECK(c[0] == 0);
        CHECK(std::abs(static_cast<int>(c[2]) - static_cast<int>(c[3])) <= static_cast<int>(c[2] / 50));
    }
}

TEST_CASE("a single-sided mesh shows its front only, mirrored or not", "[technique][visibility][culling]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    auto r = renderer(*gpu);
    const uint32_t w = 101;
    const uint32_t h = 81;
    const render::Projection projection =
        render::projectionFor(render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}), w, h);
    const auto mesh = square(*r, 1.0F);   // counter-clockwise seen from +z
    const auto covered = [&](const render::Mat4& transform, bool doubleSided) {
        world::MeshInstance i;
        i.mesh = mesh;
        i.objectToWorld = aofx::xform::translation({0.0, 0.0, -5.0}) * transform;
        i.doubleSided = doubleSided;
        const render::RenderTargets out = draw(*gpu, *r, std::span<const world::MeshInstance>(&i, 1), projection, w, h);
        gpu::BufferDesc zeros;
        zeros.bytes = uint64_t{w} * h * 4;
        zeros.elementBytes = 4;
        auto empty = gpu::Buffer::create(*gpu->device, zeros);
        REQUIRE(empty);
        auto drawn = render::countDifferent(*gpu->library, out.depth, *empty, w * h);
        REQUIRE(drawn);
        return *drawn;
    };
    const uint64_t front = covered(render::Mat4::identity(), false);
    const uint64_t back = covered(aofx::xform::rotationY(180.0), false);
    const uint64_t backTwoSided = covered(aofx::xform::rotationY(180.0), true);
    // Mirrored: winding on screen turns round, the side the object's normal
    // is on does not -- that side is still the front.
    const uint64_t mirroredFront = covered(aofx::xform::scaling({-1.0, 1.0, 1.0}), false);
    const uint64_t mirroredBack = covered(aofx::xform::scaling({-1.0, 1.0, 1.0}) * aofx::xform::rotationY(180.0), false);
    std::printf("  single sided: front %llu, back %llu (two sided %llu); mirrored front %llu, back %llu\n",
                (unsigned long long)front, (unsigned long long)back, (unsigned long long)backTwoSided,
                (unsigned long long)mirroredFront, (unsigned long long)mirroredBack);
    CHECK(front > 1000);
    CHECK(back == 0);
    CHECK(backTwoSided == front);
    CHECK(mirroredFront == front);
    CHECK(mirroredBack == 0);
}

// A mesh deformed in place: rebuilt with the same topology key and layout,
// the scene copies its positions over the old ones -- the pools stand, no
// repack -- and what was built on them is refit, not rebuilt: the hardware
// bottom level in place, the compute LBVH by settling its boxes over the
// same tree. Checked by what the deformation changes: rays and the walker
// must see the triangles the rasteriser sees (it reads the pool directly),
// the LBVH's nodes must each hold their children, and with the refit
// skipped on purpose the same comparisons fail by the thousand.
TEST_CASE("a deformed mesh refits its acceleration structures and the routes still agree",
          "[technique][visibility][deformation][refit]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization || !gpu->device->caps().rayQuery) {
        SKIP("needs both rasterisation and ray queries");
    }
    auto r = renderer(*gpu);
    auto rt = world::RayTracingScene::create(*gpu->library);
    auto trace = technique::VisibilityTrace::create(*gpu->library);
    auto bvh = world::BvhScene::create(*gpu->library);
    auto walk = technique::VisibilityBvh::create(*gpu->library);
    auto compare = gpu::ComputeKernel::create(*gpu->library, "lrt/test/ids_compare", "idsCompare");
    auto check = gpu::ComputeKernel::create(*gpu->library, "lrt/test/bvh_check", "bvhCheck");
    if (!rt) FAIL(rt.error().toString());
    if (!trace) FAIL(trace.error().toString());
    if (!bvh) FAIL(bvh.error().toString());
    if (!walk) FAIL(walk.error().toString());
    if (!compare) FAIL(compare.error().toString());
    if (!check) FAIL(check.error().toString());
    const uint32_t w = 241;
    const uint32_t h = 181;
    const render::Projection projection =
        render::projectionFor(render::Camera::lookingAt({3.0, 4.0, 11.0}, {0.0, 0.0, 0.0}), w, h);
    // The bumpy grid, built as many times as it is deformed: one topology.
    const int n = 32;
    std::vector<int32_t> counts;
    std::vector<int32_t> indices;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int p = y * (n + 1) + x;
            counts.push_back(4);
            indices.insert(indices.end(), {p, p + n + 1, p + n + 2, p + 1});
        }
    }
    const auto grid = [&](float phase, float amplitude) {
        std::vector<float> points;
        for (int y = 0; y <= n; ++y) {
            for (int x = 0; x <= n; ++x) {
                const float fx = static_cast<float>(x) / n * 6.0F - 3.0F;
                const float fy = static_cast<float>(y) / n * 6.0F - 3.0F;
                points.insert(points.end(),
                              {fx, amplitude * std::sin(fx * 1.7F + phase) * std::cos(fy * 1.3F - phase), fy});
            }
        }
        geom::MeshInput in;
        in.source = "grid";
        in.topology = 7;
        in.points = {std::as_bytes(std::span<const float>(points)), false};
        in.faceVertexCounts = counts;
        in.faceVertexIndices = indices;
        auto built = r->builder.build(in);
        if (!built) FAIL(built.error().toString());
        return std::make_shared<const geom::GpuMesh>(std::move(*built));
    };
    const auto idsAgainstRaster = [&](const technique::VisibilityTargets& rastered,
                                      const technique::VisibilityTargets& other) {
        gpu::Buffer out = test::uintBuffer(*gpu->device, 4, "ids.counts");
        auto viewA = rastered.ids.view(0);
        auto viewB = other.ids.view(0);
        REQUIRE(viewA);
        REQUIRE(viewB);
        gpu::CommandBatch batch(*gpu->device);
        compare->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding((*viewA).get());
            cursor["b"].setBinding((*viewB).get());
            cursor["counts"].setBinding(out.rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
        });
        REQUIRE(batch.submit(true));
        uint32_t c[4] = {0, 0, 0, 0};
        REQUIRE(out.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(c[1] > 1000);
        return c[0];
    };
    const auto bvhViolations = [&]() {
        gpu::Buffer out = test::uintBuffer(*gpu->device, 2, "bvh.violations");
        const uint32_t triangles = r->scene.mesh(0).triangles;
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(r->scene.positions().rhi());
            cursor["indices"].setBinding(r->scene.indices().rhi());
            cursor["meshBoxes"].setBinding(bvh->meshBoxes().rhi());
            cursor["meshChildren"].setBinding(bvh->meshChildren().rhi());
            cursor["meshLeaves"].setBinding(bvh->meshLeaves().rhi());
            cursor["violations"].setBinding(out.rhi());
            cursor["params"]["count"].setData(triangles);
            cursor["params"]["nodeBase"].setData(uint32_t{0});
            cursor["params"]["leafBase"].setData(r->scene.firstTriangle(0));
            cursor["mesh"]["firstPoint"].setData(r->scene.firstPoint(0));
            cursor["mesh"]["firstTriangle"].setData(r->scene.firstTriangle(0));
        });
        REQUIRE(batch.submit(true));
        uint32_t c[2] = {0, 0};
        REQUIRE(out.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(c[1] == triangles - 1);
        return c[0];
    };
    const auto routes = [&](bool refit, uint32_t* raysDiffer, uint32_t* walkDiffer) {
        technique::VisibilityTargets rastered;
        technique::VisibilityTargets traced;
        technique::VisibilityTargets walked;
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(r->raster.render(batch, r->scene, projection, w, h, rastered));
            REQUIRE(batch.submit(true));
        }
        REQUIRE(rt->build(r->scene, refit));
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(trace->render(batch, *rt, projection, w, h, traced));
            REQUIRE(batch.submit(true));
        }
        REQUIRE(bvh->build(r->scene, refit));
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(walk->render(batch, r->scene, *bvh, projection, w, h, walked));
            REQUIRE(batch.submit(true));
        }
        *raysDiffer = idsAgainstRaster(rastered, traced);
        *walkDiffer = idsAgainstRaster(rastered, walked);
    };

    world::MeshInstance ground;
    ground.mesh = grid(0.0F, 0.4F);
    REQUIRE(r->scene.update(std::span<const world::MeshInstance>(&ground, 1), projection));
    const uint64_t generation = r->scene.generation();
    uint32_t rays = 0;
    uint32_t walked = 0;
    routes(true, &rays, &walked);
    std::printf("  built: %u / %u interior pixels differ (rays / walker), %u LBVH violations\n", rays, walked,
                bvhViolations());
    CHECK(rays == 0);
    CHECK(walked == 0);

    // Deformed: the same topology, other heights. No repack, a refit.
    ground.mesh = grid(1.3F, 0.9F);
    REQUIRE(r->scene.update(std::span<const world::MeshInstance>(&ground, 1), projection));
    CHECK(r->scene.generation() == generation);
    CHECK(r->scene.positionsRevision() == 1);
    routes(true, &rays, &walked);
    const uint32_t violationsAfterRefit = bvhViolations();
    std::printf("  deformed and refit: %u / %u differ, %u LBVH violations\n", rays, walked, violationsAfterRefit);
    CHECK(rays == 0);
    CHECK(walked == 0);
    CHECK(violationsAfterRefit == 0);

    // Deformed again with the refit skipped: the structures are the last
    // shape's, and the comparisons say so.
    ground.mesh = grid(2.6F, -0.7F);
    REQUIRE(r->scene.update(std::span<const world::MeshInstance>(&ground, 1), projection));
    CHECK(r->scene.generation() == generation);
    routes(false, &rays, &walked);
    const uint32_t violationsStale = bvhViolations();
    std::printf("  deformed, refit skipped: %u / %u differ, %u LBVH violations\n", rays, walked, violationsStale);
    CHECK(rays > 1000);
    CHECK(walked > 1000);
    CHECK(violationsStale > 100);
    // And caught up.
    routes(true, &rays, &walked);
    std::printf("  then refit: %u / %u differ, %u LBVH violations\n", rays, walked, bvhViolations());
    CHECK(rays == 0);
    CHECK(walked == 0);
}
