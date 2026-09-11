// Copyright (c) 2026 lucabRTrender contributors.
//
// MaterialX documents through MaterialCompiler into Slang the device compiles:
// the surface shaders USD brings (UsdPreviewSurface, standard_surface,
// OpenPBR, glTF PBR) and an unlit texture graph.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "lrt/material/MaterialCompiler.h"
#include "lrt/material/TextureStore.h"

using namespace lrt;

namespace {

std::unique_ptr<material::MaterialCompiler> compiler(test::Gpu& gpu) {
    std::vector<std::filesystem::path> shaders;
    for (const std::string& path : gpu.device->shaderSearchPaths()) {
        shaders.emplace_back(path);
    }
    auto made = material::MaterialCompiler::create({LRT_MATERIALX_ROOT}, shaders);
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

std::string surface(const std::string& node, const std::string& inputs) {
    return "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
           "  <" + node + " name=\"shader\" type=\"surfaceshader\">\n" + inputs +
           "  </" + node + ">\n"
           "  <surfacematerial name=\"material\" type=\"material\">\n"
           "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
           "  </surfacematerial>\n</materialx>\n";
}

}   // namespace

TEST_CASE("MaterialX surface shaders compile to Slang modules the device loads", "[material][materialx]") {
    LRT_REQUIRE_GPU(gpu);
    auto mx = compiler(*gpu);
    struct Case {
        const char* name;
        std::string xml;
    };
    const std::vector<Case> cases{
        {"UsdPreviewSurface",
         surface("UsdPreviewSurface", "    <input name=\"diffuseColor\" type=\"color3\" value=\"0.8, 0.2, 0.1\" />\n"
                                      "    <input name=\"roughness\" type=\"float\" value=\"0.4\" />\n")},
        {"standard_surface", surface("standard_surface", "    <input name=\"coat\" type=\"float\" value=\"0.5\" />\n")},
        {"open_pbr_surface", surface("open_pbr_surface", "")},
        {"gltf_pbr", surface("gltf_pbr", "    <input name=\"metallic\" type=\"float\" value=\"1.0\" />\n")},
        {"unlit texture",
         "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
         "  <image name=\"tex\" type=\"color3\">\n"
         "    <input name=\"file\" type=\"filename\" value=\"wood.png\" colorspace=\"srgb_texture\" />\n"
         "    <input name=\"uaddressmode\" type=\"string\" value=\"clamp\" />\n"
         "  </image>\n"
         "  <surface_unlit name=\"shader\" type=\"surfaceshader\">\n"
         "    <input name=\"emission_color\" type=\"color3\" nodename=\"tex\" />\n"
         "  </surface_unlit>\n"
         "  <surfacematerial name=\"material\" type=\"material\">\n"
         "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
         "  </surfacematerial>\n</materialx>\n"},
    };
    for (const Case& c : cases) {
        SECTION(c.name) {
            auto compiled = mx->compileXml(c.xml);
            if (!compiled) FAIL(compiled.error().toString());
            size_t textures = 0;
            size_t primvars = 0;
            for (const auto& slot : compiled->slots) {
                textures += slot.kind == material::MaterialSlot::Kind::Texture ? 1 : 0;
                primvars += slot.kind == material::MaterialSlot::Kind::Primvar ? 1 : 0;
            }
            std::printf("  %s: %s, %zu lines, %u blob words, %zu textures, %zu primvars\n", c.name,
                        compiled->module.c_str(), static_cast<size_t>(std::count(compiled->source.begin(),
                                                                                 compiled->source.end(), '\n')),
                        compiled->words, textures, primvars);
            if (const char* dump = std::getenv("LRT_TEST_DUMP"); dump != nullptr) {
                std::FILE* f = std::fopen((std::string(dump) + "/" + compiled->module + ".slang").c_str(), "w");
                if (f != nullptr) {
                    std::fwrite(compiled->source.data(), 1, compiled->source.size(), f);
                    std::fclose(f);
                }
            }
            auto loaded = gpu->library->loadSource(compiled->module, compiled->source, {});
            if (!loaded) FAIL(loaded.error().toString());
        }
    }
}

namespace {

std::string replaceAll(std::string text, const std::string& from, const std::string& to) {
    for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
    return text;
}

const char* kProbe = R"(
import lrt.material.material_runtime;
import LOBES_MODULE;
import REFERENCE_MODULE;

struct ProbeParams {
    uint count;
    uint referenceBlob;
    uint seed;
    uint pad0;
};

RWStructuredBuffer<float4>  ours;
RWStructuredBuffer<float4>  theirs;
ConstantBuffer<ProbeParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void materialProbe(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    Rng rng = Rng(i, params.seed);
    const float3 n = normalize(sampleUniformSphere(float2(rng.next(), rng.next())));
    const Frame frame = Frame(n, float3(0.3, 0.9, 0.1));
    const float3 v = frame.toWorld(sampleCosineHemisphere(float2(rng.next(), rng.next())));
    const float3 l = frame.toWorld(sampleCosineHemisphere(float2(rng.next(), rng.next())));
    MaterialInputs inputs;
    inputs.positionWorld = float3(0.0);
    inputs.normalWorld = n;
    inputs.tangentWorld = frame.x;
    inputs.bitangentWorld = frame.y;
    inputs.positionObject = float3(0.0);
    inputs.normalObject = n;
    inputs.tangentObject = frame.x;
    inputs.bitangentObject = frame.y;
    inputs.viewPosition = v * 4.0;
    inputs.uvDx = float2(0.0);
    inputs.uvDy = float2(0.0);
    inputs.frame = 0.0;
    inputs.time = 0.0;
    inputs.inside = false;
    inputs.mesh.firstPoint = 0;
    inputs.mesh.points = 0;
    inputs.mesh.firstTriangle = 0;
    inputs.mesh.triangles = 0;
    inputs.mesh.hasNormals = 0;
    inputs.mesh.nodeBase = 0;
    inputs.mesh.slotBase = 0;
    inputs.mesh.pad2 = 0;
    inputs.mesh.boundsLo = float4(0.0);
    inputs.mesh.boundsHi = float4(0.0);
    inputs.triangle = 0;
    inputs.points = uint3(0);
    inputs.weights = float3(1.0, 0.0, 0.0);
    inputs.displayColor = float4(0.18, 0.18, 0.18, 1.0);
    LOBES_FUNCTION(inputs, 0);
    const LobeStack stack = gLrtResult;
    ours[i] = float4(stackEval(stack, v, l), float(stack.count));
    gLrtReferenceL = l;
    gLrtReferenceResponse = float3(0.0);
    REFERENCE_FUNCTION(inputs, params.referenceBlob);
    theirs[i] = float4(gLrtReferenceResponse, 0.0);
}
)";

}   // namespace

TEST_CASE("a surface shader's lobes evaluate as MaterialX's own genglsl closures respond", "[material][materialx]") {
    LRT_REQUIRE_GPU(gpu);
    auto mx = compiler(*gpu);
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    struct Case {
        const char* name;
        std::string xml;
    };
    const auto closure = [](const std::string& node, const std::string& inputs) {
        return "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n  <" + node + " name=\"b\" type=\"BSDF\">\n" +
               inputs + "  </" + node + ">\n"
               "  <surface name=\"shader\" type=\"surfaceshader\">\n"
               "    <input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" />\n  </surface>\n"
               "  <surfacematerial name=\"material\" type=\"material\">\n"
               "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" />\n"
               "  </surfacematerial>\n</materialx>\n";
    };
    const std::vector<Case> cases{
        {"oren_nayar_diffuse_bsdf, roughness 0.5",
         closure("oren_nayar_diffuse_bsdf", "    <input name=\"color\" type=\"color3\" value=\"0.6, 0.3, 0.2\" />\n"
                                            "    <input name=\"roughness\" type=\"float\" value=\"0.5\" />\n")},
        {"generalized_schlick_bsdf, alpha 0.0625",
         closure("generalized_schlick_bsdf", "    <input name=\"color0\" type=\"color3\" value=\"0.3, 0.5, 0.2\" />\n"
                                             "    <input name=\"roughness\" type=\"vector2\" value=\"0.0625, 0.0625\" />\n")},
        {"conductor_bsdf, alpha 0.2",
         closure("conductor_bsdf", "    <input name=\"roughness\" type=\"vector2\" value=\"0.2, 0.2\" />\n")},
        {"dielectric_bsdf R, alpha 0.3",
         closure("dielectric_bsdf", "    <input name=\"roughness\" type=\"vector2\" value=\"0.3, 0.3\" />\n")},
        {"sheen_bsdf",
         closure("sheen_bsdf", "    <input name=\"roughness\" type=\"float\" value=\"0.4\" />\n")},
        {"layer: schlick over oren_nayar",
         "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
         "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\"><input name=\"color\" type=\"color3\" value=\"0.6, 0.3, 0.2\" /></oren_nayar_diffuse_bsdf>\n"
         "  <generalized_schlick_bsdf name=\"s\" type=\"BSDF\"><input name=\"color0\" type=\"color3\" value=\"0.04, 0.04, 0.04\" /><input name=\"roughness\" type=\"vector2\" value=\"0.1, 0.1\" /></generalized_schlick_bsdf>\n"
         "  <layer name=\"b\" type=\"BSDF\"><input name=\"top\" type=\"BSDF\" nodename=\"s\" /><input name=\"base\" type=\"BSDF\" nodename=\"d\" /></layer>\n"
         "  <surface name=\"shader\" type=\"surfaceshader\"><input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" /></surface>\n"
         "  <surfacematerial name=\"material\" type=\"material\"><input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" /></surfacematerial>\n"
         "</materialx>\n"},
        {"mix and multiply: 0.3 schlick, 0.7 of 0.5 oren_nayar",
         "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
         "  <oren_nayar_diffuse_bsdf name=\"d\" type=\"BSDF\"><input name=\"color\" type=\"color3\" value=\"0.6, 0.3, 0.2\" /></oren_nayar_diffuse_bsdf>\n"
         "  <multiply name=\"m\" type=\"BSDF\"><input name=\"in1\" type=\"BSDF\" nodename=\"d\" /><input name=\"in2\" type=\"float\" value=\"0.5\" /></multiply>\n"
         "  <generalized_schlick_bsdf name=\"s\" type=\"BSDF\"><input name=\"color0\" type=\"color3\" value=\"0.04, 0.04, 0.04\" /><input name=\"roughness\" type=\"vector2\" value=\"0.1, 0.1\" /></generalized_schlick_bsdf>\n"
         "  <mix name=\"b\" type=\"BSDF\"><input name=\"fg\" type=\"BSDF\" nodename=\"s\" /><input name=\"bg\" type=\"BSDF\" nodename=\"m\" /><input name=\"mix\" type=\"float\" value=\"0.3\" /></mix>\n"
         "  <surface name=\"shader\" type=\"surfaceshader\"><input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" /></surface>\n"
         "  <surfacematerial name=\"material\" type=\"material\"><input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"shader\" /></surfacematerial>\n"
         "</materialx>\n"},
        {"UsdPreviewSurface, metallic 0.3, clearcoat 0.6",
         surface("UsdPreviewSurface", "    <input name=\"diffuseColor\" type=\"color3\" value=\"0.8, 0.2, 0.1\" />\n"
                                      "    <input name=\"roughness\" type=\"float\" value=\"0.4\" />\n"
                                      "    <input name=\"metallic\" type=\"float\" value=\"0.3\" />\n"
                                      "    <input name=\"clearcoat\" type=\"float\" value=\"0.6\" />\n"
                                      "    <input name=\"clearcoatRoughness\" type=\"float\" value=\"0.2\" />\n")},
        {"UsdPreviewSurface, specular workflow",
         surface("UsdPreviewSurface", "    <input name=\"useSpecularWorkflow\" type=\"integer\" value=\"1\" />\n"
                                      "    <input name=\"specularColor\" type=\"color3\" value=\"0.3, 0.5, 0.2\" />\n"
                                      "    <input name=\"roughness\" type=\"float\" value=\"0.25\" />\n")},
        {"standard_surface, metalness 0.2, coat 0.5, sheen 0.3",
         surface("standard_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.6, 0.4, 0.3\" />\n"
                                     "    <input name=\"specular_roughness\" type=\"float\" value=\"0.35\" />\n"
                                     "    <input name=\"metalness\" type=\"float\" value=\"0.2\" />\n"
                                     "    <input name=\"coat\" type=\"float\" value=\"0.5\" />\n"
                                     "    <input name=\"sheen\" type=\"float\" value=\"0.3\" />\n")},
        {"open_pbr_surface, coat 0.4, fuzz 0.3",
         surface("open_pbr_surface", "    <input name=\"base_color\" type=\"color3\" value=\"0.7, 0.5, 0.2\" />\n"
                                     "    <input name=\"specular_roughness\" type=\"float\" value=\"0.3\" />\n"
                                     "    <input name=\"coat_weight\" type=\"float\" value=\"0.4\" />\n"
                                     "    <input name=\"fuzz_weight\" type=\"float\" value=\"0.3\" />\n")},
        {"gltf_pbr, metallic 0.5, clearcoat 0.4",
         surface("gltf_pbr", "    <input name=\"base_color\" type=\"color3\" value=\"0.9, 0.6, 0.3\" />\n"
                             "    <input name=\"metallic\" type=\"float\" value=\"0.5\" />\n"
                             "    <input name=\"roughness\" type=\"float\" value=\"0.3\" />\n"
                             "    <input name=\"clearcoat\" type=\"float\" value=\"0.4\" />\n")},
    };
    constexpr uint32_t kSamples = 1u << 16;
    for (const Case& c : cases) {
        SECTION(c.name) {
            auto lobes = mx->compileXml(c.xml);
            auto reference = mx->compileXml(c.xml, {}, material::ClosureVariant::GenglslReference);
            if (!lobes) FAIL(lobes.error().toString());
            if (!reference) FAIL(reference.error().toString());
            if (const char* dump = std::getenv("LRT_TEST_DUMP"); dump != nullptr) {
                for (const auto* m : {&*lobes, &*reference}) {
                    if (std::FILE* f = std::fopen((std::string(dump) + "/" + m->module + ".slang").c_str(), "w")) {
                        std::fwrite(m->source.data(), 1, m->source.size(), f);
                        std::fclose(f);
                    }
                }
            }
            REQUIRE(gpu->library->loadSource(lobes->module, lobes->source, {}));
            // genglsl's reflection pass scales what lies beneath a specular
            // layer by an albedo compensated with the Fresnel at the light's
            // half vector, so its throughput depends on the light; its
            // indirect pass uses the Fresnel at the view, which is what a
            // lobe stack can carry. The reference takes the view's here.
            std::string referenceSource = reference->source;
            referenceSource = replaceAll(
                referenceSource, "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;",
                "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * "
                "mx_ggx_energy_compensation(NdotV, avgAlpha, mx_compute_fresnel(NdotV, fd));");
            referenceSource = replaceAll(
                referenceSource, "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;",
                "float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * "
                "mx_ggx_energy_compensation(NdotV, avgAlpha, mx_compute_fresnel(NdotV, fd));");
            // genglsl's subsurface approximation reads screen-space curvature
            // (fwidth), which compute has not; these cases carry no subsurface.
            referenceSource = replaceAll(referenceSource, "float curvature = length(fwidth(N)) / length(fwidth(P));",
                                         "float curvature = 0.0;");
            const std::string referenceModule = reference->module + "_view_throughput";
            if (auto loaded = gpu->library->loadSource(referenceModule, referenceSource, {}); !loaded) {
                FAIL(loaded.error().toString());
            }
            const auto noSlot = [](const std::string&) { return uint32_t{0xFFFFFFFF}; };
            std::vector<float> blob = material::MaterialCompiler::parameters(*lobes, **textures, noSlot);
            const std::vector<float> theirsBlob = material::MaterialCompiler::parameters(*reference, **textures, noSlot);
            const uint32_t referenceBlob = static_cast<uint32_t>(blob.size());
            blob.insert(blob.end(), theirsBlob.begin(), theirsBlob.end());
            REQUIRE((*textures)->commit());
            std::string source = replaceAll(kProbe, "LOBES_MODULE", lobes->module);
            source = replaceAll(source, "REFERENCE_MODULE", referenceModule);
            source = replaceAll(source, "LOBES_FUNCTION", lobes->function);
            source = replaceAll(source, "REFERENCE_FUNCTION", reference->function);
            const std::string probeName = "lrt_probe_" + lobes->module.substr(8) + "_" + reference->module.substr(8);
            if (auto probe = gpu->library->loadSource(probeName, source, {"materialProbe"}); !probe) {
                FAIL(probe.error().toString());
            }
            auto probe = gpu::ComputeKernel::create(*gpu->library, probeName, "materialProbe");
            auto compare = gpu::ComputeKernel::create(*gpu->library, "lrt/test/material_compare", "materialCompare");
            if (!probe) FAIL(probe.error().toString());
            REQUIRE(compare);
            const auto floats = [&](uint64_t count, uint32_t element, const float* data, const char* label) {
                gpu::BufferDesc desc;
                desc.bytes = std::max<uint64_t>(count, 1) * element;
                desc.elementBytes = element;
                desc.label = label;
                auto made = gpu::Buffer::create(*gpu->device, desc, data);
                REQUIRE(made);
                return *made;
            };
            gpu::Buffer blobBuffer = floats(blob.size(), 4, blob.data(), "material.blob");
            gpu::Buffer ours = floats(kSamples, 16, nullptr, "probe.ours");
            gpu::Buffer theirs = floats(kSamples, 16, nullptr, "probe.theirs");
            gpu::Buffer placeholder4 = floats(1, 16, nullptr, "probe.placeholder4");
            gpu::Buffer placeholder = test::uintBuffer(*gpu->device, 4, "probe.placeholder");
            {
                gpu::CommandBatch batch(*gpu->device);
                probe->dispatch(batch, {kSamples, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["gMaterialBlob"].setBinding(blobBuffer.rhi());
                    (*textures)->bind(cursor["gTextures"]);
                    cursor["primvarRecords"].setBinding(placeholder.rhi());
                    cursor["primvarValues"].setBinding(placeholder4.rhi());
                    cursor["primvarSlots"].setBinding(placeholder.rhi());
                    cursor["triangleCorners"].setBinding(placeholder.rhi());
                    cursor["triangleFaces"].setBinding(placeholder.rhi());
                    cursor["ours"].setBinding(ours.rhi());
                    cursor["theirs"].setBinding(theirs.rhi());
                    cursor["params"]["count"].setData(kSamples);
                    cursor["params"]["referenceBlob"].setData(referenceBlob);
                    cursor["params"]["seed"].setData(uint32_t{77});
                });
                REQUIRE(batch.submit(true));
            }
            gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "compare.counts");
            gpu::Buffer worst = floats(1, 16, nullptr, "compare.worst");
            {
                gpu::CommandBatch batch(*gpu->device);
                compare->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["ours"].setBinding(ours.rhi());
                    cursor["theirs"].setBinding(theirs.rhi());
                    cursor["counts"].setBinding(counts.rhi());
                    cursor["worst"].setBinding(worst.rhi());
                    cursor["params"]["count"].setData(kSamples);
                    cursor["params"]["relative"].setData(1e-3F);
                    cursor["params"]["floor"].setData(1e-4F);
                });
                REQUIRE(batch.submit(true));
            }
            uint32_t n[3] = {};
            float w[4] = {};
            REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
            REQUIRE(worst.read(*gpu->device, 0, sizeof(w), w));
            std::printf("  %s: %u lobes; %u of %u samples lit, mean response %.4f; %u components beyond 1e-3, "
                        "worst %.2e (%.5f against %.5f)\n",
                        c.name, n[2], n[1], kSamples, double(w[3]), n[0], double(w[0]), double(w[1]), double(w[2]));
            CHECK(n[1] > kSamples / 2);
            CHECK(n[0] == 0);
        }
    }
}
