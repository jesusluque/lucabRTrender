// Copyright (c) 2026 lucabRTrender contributors.
//
// Every lobe's sampling against its density (chi-square), its sample weights
// against its eval (two albedo estimators that must agree), and the lobes that
// lose no light against 1 (the furnace). All of it in lobe_check.slang.
#include "../gpu/GpuTest.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

using namespace lrt;

namespace {

struct LobeCase {
    const char*          name;
    uint32_t             kind;
    uint32_t             flags = 0;
    uint32_t             scatter = 0;
    std::array<float, 3> colour0{1.0F, 1.0F, 1.0F};
    std::array<float, 3> colour1{0.0F, 0.0F, 0.0F};
    std::array<float, 3> colour2{1.0F, 1.0F, 1.0F};
    float                ior = 1.5F;
    float                exponent = 5.0F;
    float                roughness = 0.0F;
    float                alphaX = 0.3F;
    float                alphaY = 0.3F;
    float                cosThetaO = 0.766F;   // 40 degrees
    bool                 chiSquare = true;
    float                furnace = -1.0F;       // the albedo it must return, where it is known
    float                furnaceTolerance = 0.01F;
    bool                 atMostOne = false;
    /// How far the two albedo estimators may part. The uniform one is heavy
    /// tailed under a narrow transmission lobe: two seeds gave 0.970 and
    /// 0.977 against a sampled 0.987.
    float                consistency = 0.02F;
    /// How far the pdf's integral over the bins may miss what was drawn: the
    /// bins' quadrature under a lobe as narrow as hair's R (v 0.1).
    float                integralTolerance = 2e-3F;
    std::array<float, 4> hairA{0.0F, 0.0F, 0.0F, 0.05F};   ///< hair: absorption, TT longitudinal variance
    std::array<float, 4> hairB{0.05F, 0.2F, 0.2F, 0.0F};   ///< hair: TT azimuthal s, TRT variance and s
};

struct Outcome {
    float statistic = 0.0F;
    float dof = 0.0F;
    float drawn = 0.0F;
    float integral = 0.0F;
    std::array<float, 3> sampled{};
    std::array<float, 3> uniform{};
};

Outcome run(test::Gpu& gpu, const LobeCase& c) {
    static const auto kernel = [&](const char* entry) {
        auto made = gpu::ComputeKernel::create(*gpu.library, "lrt/test/lobe_check", entry);
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    static gpu::ComputeKernel kDraw = kernel("lobeDraw");
    static gpu::ComputeKernel kCount = kernel("lobeCount");
    static gpu::ComputeKernel kExpect = kernel("lobeExpect");
    static gpu::ComputeKernel kStatistic = kernel("lobeStatistic");
    constexpr uint32_t kSamples = 1u << 20;
    constexpr uint32_t kTheta = 20;
    constexpr uint32_t kPhi = 40;
    const auto floats = [&](uint64_t count, uint32_t element, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = count * element;
        desc.elementBytes = element;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu.device, desc);
        if (!made) FAIL(made.error().toString());
        return *made;
    };
    gpu::Buffer bins = test::uintBuffer(*gpu.device, kSamples, "lobe.bins");
    gpu::Buffer weights = floats(kSamples, 16, "lobe.weights");
    gpu::Buffer uniforms = floats(kSamples, 16, "lobe.uniforms");
    gpu::Buffer observed = floats(kTheta * kPhi + 1, 4, "lobe.observed");
    gpu::Buffer expected = floats(kTheta * kPhi, 4, "lobe.expected");
    gpu::Buffer result = floats(4, 16, "lobe.result");
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["bins"].setBinding(bins.rhi());
        cursor["weights"].setBinding(weights.rhi());
        cursor["uniforms"].setBinding(uniforms.rhi());
        cursor["observed"].setBinding(observed.rhi());
        cursor["expected"].setBinding(expected.rhi());
        cursor["result"].setBinding(result.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["kind"].setData(c.kind);
        p["flags"].setData(c.flags);
        p["scatter"].setData(c.scatter);
        p["samples"].setData(kSamples);
        p["colour0"].setData(c.colour0.data(), sizeof(float) * 3);
        p["colour1"].setData(c.colour1.data(), sizeof(float) * 3);
        p["colour2"].setData(c.colour2.data(), sizeof(float) * 3);
        p["ior"].setData(c.ior);
        p["exponent"].setData(c.exponent);
        p["roughness"].setData(c.roughness);
        p["alphaX"].setData(c.alphaX);
        p["alphaY"].setData(c.alphaY);
        p["cosThetaO"].setData(c.cosThetaO);
        p["seed"].setData(uint32_t{1234});
        p["thetaBins"].setData(kTheta);
        p["phiBins"].setData(kPhi);
        p["hairA"].setData(c.hairA.data(), sizeof(float) * 4);
        p["hairB"].setData(c.hairB.data(), sizeof(float) * 4);
    };
    {
        gpu::CommandBatch batch(*gpu.device);
        kDraw.dispatch(batch, {kSamples, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    for (gpu::ComputeKernel* k : {&kCount, &kExpect, &kStatistic}) {
        gpu::CommandBatch batch(*gpu.device);
        k->dispatch(batch, {1, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    float r[16] = {};
    REQUIRE(result.read(*gpu.device, 0, sizeof(r), r));
    Outcome out;
    out.sampled = {r[0], r[1], r[2]};
    out.uniform = {r[4], r[5], r[6]};
    out.integral = r[8];
    out.statistic = r[12];
    out.dof = r[13];
    out.drawn = r[14];
    return out;
}

/// Wilson-Hilferty: how many standard deviations above its mean a chi-square
/// statistic with `dof` degrees of freedom lies.
double chiSquareZ(double statistic, double dof) {
    const double k = 2.0 / (9.0 * dof);
    return (std::cbrt(statistic / dof) - (1.0 - k)) / std::sqrt(k);
}

}   // namespace

TEST_CASE("each lobe samples its density, weighs its samples by its eval, and keeps the light it should",
          "[material][lobes]") {
    LRT_REQUIRE_GPU(gpu);
    constexpr uint32_t kOrenNayar = 1, kBurley = 2, kTranslucent = 3, kDielectric = 4, kConductor = 5, kSchlick = 6,
                       kSheen = 7, kHair = 8;
    std::vector<LobeCase> cases;
    {
        LobeCase c{"Lambert (Oren-Nayar, roughness 0), white", kOrenNayar};
        c.furnace = 1.0F;
        cases.push_back(c);
    }
    {
        LobeCase c{"Oren-Nayar compensated (EON), roughness 0.8, white", kOrenNayar, 1};
        c.roughness = 0.8F;
        c.furnace = 1.0F;
        c.furnaceTolerance = 0.02F;
        cases.push_back(c);
    }
    {
        LobeCase c{"Oren-Nayar, roughness 1, white", kOrenNayar};
        c.roughness = 1.0F;
        c.atMostOne = true;
        cases.push_back(c);
    }
    {
        LobeCase c{"Burley, roughness 0.5", kBurley};
        c.roughness = 0.5F;
        c.colour0 = {0.8F, 0.6F, 0.4F};
        cases.push_back(c);
    }
    {
        LobeCase c{"translucent, white", kTranslucent};
        c.furnace = 1.0F;
        cases.push_back(c);
    }
    {
        LobeCase c{"conductor, gold, alpha 0.3", kConductor};
        c.colour0 = {0.18F, 0.42F, 1.37F};
        c.colour1 = {3.42F, 2.35F, 1.77F};
        cases.push_back(c);
    }
    {
        LobeCase c{"conductor, k 100, alpha 0.6, 80 degrees", kConductor};
        c.colour0 = {1.0F, 1.0F, 1.0F};
        c.colour1 = {100.0F, 100.0F, 100.0F};
        c.alphaX = 0.6F;
        c.alphaY = 0.6F;
        c.cosThetaO = 0.174F;
        c.furnace = 1.0F;
        c.furnaceTolerance = 0.05F;
        cases.push_back(c);
    }
    {
        LobeCase c{"conductor, anisotropic alpha 0.15 x 0.5", kConductor};
        c.colour0 = {0.2F, 0.2F, 0.2F};
        c.colour1 = {3.0F, 3.0F, 3.0F};
        c.alphaX = 0.15F;
        c.alphaY = 0.5F;
        cases.push_back(c);
    }
    {
        LobeCase c{"dielectric R+T, ior 1.5, alpha 0.25", kDielectric, 0, 2};
        c.consistency = 0.03F;
        c.alphaX = 0.25F;
        c.alphaY = 0.25F;
        c.atMostOne = true;
        cases.push_back(c);
    }
    {
        LobeCase c{"dielectric R+T from inside, ior 1.5, alpha 0.3 (total internal reflection)", kDielectric, 2, 2};
        c.cosThetaO = 0.5F;
        cases.push_back(c);
    }
    {
        LobeCase c{"dielectric R only, ior 1.33, alpha 0.5", kDielectric, 0, 0};
        c.ior = 1.33F;
        c.alphaX = 0.5F;
        c.alphaY = 0.5F;
        cases.push_back(c);
    }
    {
        LobeCase c{"dielectric R+T, smooth", kDielectric, 0, 2};
        c.alphaX = 0.0F;
        c.alphaY = 0.0F;
        c.chiSquare = false;
        c.furnace = 1.0F;
        c.furnaceTolerance = 1e-3F;
        cases.push_back(c);
    }
    {
        LobeCase c{"generalized Schlick R+T, F0 0.04, F82 0.9, alpha 0.25", kSchlick, 0, 2};
        c.consistency = 0.03F;
        c.colour0 = {0.04F, 0.04F, 0.04F};
        c.colour1 = {1.0F, 1.0F, 1.0F};
        c.colour2 = {0.9F, 0.9F, 0.9F};
        c.alphaX = 0.25F;
        c.alphaY = 0.25F;
        cases.push_back(c);
    }
    {
        LobeCase c{"sheen (Imageworks), roughness 0.3", kSheen};
        c.roughness = 0.3F;
        c.atMostOne = true;
        cases.push_back(c);
    }
    {
        LobeCase c{"sheen (Zeltner), roughness 0.5", kSheen, 4};
        c.roughness = 0.5F;
        c.atMostOne = true;
        cases.push_back(c);
    }
    // Hair: with no absorption and white tints the attenuations sum to 1 and
    // so does the albedo -- the white furnace, exactly. The cuticle tilts
    // the lobes; at 0.5 there is no tilt.
    {
        LobeCase c{"hair, no absorption, roughness R 0.2 / 0.3", kHair};
        c.colour1 = {1.0F, 1.0F, 1.0F};
        c.colour2 = {1.0F, 1.0F, 1.0F};
        c.alphaX = 0.2F;
        c.alphaY = 0.3F;
        c.roughness = 0.5F;
        c.ior = 1.55F;
        c.hairA = {0.0F, 0.0F, 0.0F, 0.1F};
        c.hairB = {0.3F, 0.4F, 0.3F, 0.0F};
        c.furnace = 1.0F;
        c.furnaceTolerance = 0.02F;
        c.consistency = 0.03F;
        c.cosThetaO = 0.9F;
        cases.push_back(c);
    }
    {
        LobeCase c{"hair, absorbing, no tilt, 60 degrees", kHair};
        c.colour0 = {1.0F, 1.0F, 1.0F};
        c.colour1 = {0.9F, 0.7F, 0.5F};
        c.colour2 = {0.8F, 0.8F, 0.8F};
        c.alphaX = 0.1F;
        c.alphaY = 0.2F;
        c.roughness = 0.5F;
        c.ior = 1.55F;
        c.hairA = {0.3F, 0.5F, 0.9F, 0.05F};
        c.hairB = {0.2F, 0.2F, 0.2F, 0.0F};
        c.atMostOne = true;
        c.consistency = 0.03F;
        c.integralTolerance = 0.01F;
        c.cosThetaO = 0.5F;
        cases.push_back(c);
    }
    {
        LobeCase c{"hair, absorbing, tilted cuticle, 60 degrees", kHair};
        c.colour0 = {1.0F, 1.0F, 1.0F};
        c.colour1 = {0.9F, 0.7F, 0.5F};
        c.colour2 = {0.8F, 0.8F, 0.8F};
        c.alphaX = 0.1F;
        c.alphaY = 0.2F;
        c.roughness = 0.52F;
        c.ior = 1.55F;
        c.hairA = {0.3F, 0.5F, 0.9F, 0.05F};
        c.hairB = {0.2F, 0.2F, 0.2F, 0.0F};
        c.atMostOne = true;
        c.consistency = 0.03F;
        c.cosThetaO = 0.5F;
        c.integralTolerance = 0.01F;
        cases.push_back(c);
    }
    for (const LobeCase& c : cases) {
        SECTION(c.name) {
            const Outcome o = run(*gpu, c);
            const double z = chiSquareZ(o.statistic, o.dof);
            const auto relative = [](float a, float b) {
                return std::abs(a - b) / std::max(std::max(std::abs(a), std::abs(b)), 1e-3F);
            };
            float worst = 0.0F;
            for (int k = 0; k < 3; ++k) {
                worst = std::max(worst, relative(o.sampled[size_t(k)], o.uniform[size_t(k)]));
            }
            std::printf("  %s: chi2 %.1f on %.0f dof (z %.2f), pdf integral %.4f against %.4f drawn; albedo %.4f "
                        "sampled, %.4f uniform (green)\n",
                        c.name, double(o.statistic), double(o.dof), z, double(o.integral), double(o.drawn),
                        double(o.sampled[1]), double(o.uniform[1]));
            if (c.chiSquare) {
                CHECK(z < 3.7);   // p > 1e-4
                CHECK(std::abs(o.integral - o.drawn) < c.integralTolerance);
                CHECK(worst < c.consistency);
            }
            if (c.furnace >= 0.0F) {
                for (int k = 0; k < 3; ++k) {
                    CHECK(std::abs(o.sampled[size_t(k)] - c.furnace) <= c.furnaceTolerance);
                }
            }
            if (c.atMostOne) {
                for (int k = 0; k < 3; ++k) {
                    CHECK(o.sampled[size_t(k)] <= 1.0F + 5e-3F);
                }
            }
        }
    }
}
