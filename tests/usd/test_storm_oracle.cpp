// Copyright (c) 2026 lucabRTrender contributors.
//
// Storm -- OpenUSD's own GPU renderer, on Metal here -- as the oracle for what
// geometry the engine draws: the same stage through the same scene indices,
// the same camera, compared on primId, depth and Neye by a kernel. Not colour:
// Storm lights and shades by its own rules. Its own executable, so Storm's
// device and plugins live in a process of their own.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/range1f.h>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/glf/simpleLight.h>
#include <pxr/imaging/glf/simpleLightingContext.h>
#include <pxr/imaging/hd/driver.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/tokens.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>

#include "lrt/core/Platform.h"
#include "lrt/usd/StageRenderer.h"


using namespace lrt;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

struct Outputs {
    std::vector<uint8_t> primId, depth, eye, colour;
};

GfMatrix4d viewOf(const render::Camera& camera) {
    const render::Mat4 toCamera = aofx::xform::inverseAffine(camera.cameraToWorld);
    GfMatrix4d view;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            view[c][r] = toCamera.at(r, c);
        }
    }
    return view;
}

GfMatrix4d projectionOf(const render::Camera& camera, uint32_t w, uint32_t h) {
    GfCamera lens;
    lens.SetFocalLength(static_cast<float>(camera.lens.focal));
    lens.SetHorizontalAperture(static_cast<float>(camera.lens.haperture));
    lens.SetVerticalAperture(static_cast<float>(camera.lens.haperture * h / w));
    lens.SetClippingRange(GfRange1f(static_cast<float>(camera.lens.nearZ), static_cast<float>(camera.lens.farZ)));
    return lens.GetFrustum().ComputeProjectionMatrix();
}

std::vector<uint8_t> mapped(HdRenderBuffer* buffer) {
    buffer->Resolve();
    const size_t bytes = size_t{buffer->GetWidth()} * buffer->GetHeight() * HdDataSizeOfFormat(buffer->GetFormat());
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), buffer->Map(), bytes);
    buffer->Unmap();
    return out;
}

Outputs storm(const fs::path& stagePath, const render::Camera& camera, uint32_t w, uint32_t h, bool colour = false) {
#if defined(__linux__)
    // Storm draws through OpenGL here, and HgiGL expects a context current.
    if (!platform::makeHeadlessGlContextCurrent()) {
        SKIP("no OpenGL context could be made for Storm (EGL on a GPU device)");
    }
#endif
    HgiUniquePtr hgi = Hgi::CreatePlatformDefaultHgi();
    REQUIRE(hgi);
    HdDriver driver{HgiTokens->renderDriver, VtValue(hgi.get())};
    HdPluginRenderDelegateUniqueHandle delegate =
        HdRendererPluginRegistry::GetInstance().CreateRenderDelegate(TfToken("HdStormRendererPlugin"));
    REQUIRE(delegate);
    std::unique_ptr<HdRenderIndex> index(HdRenderIndex::New(delegate.Get(), {&driver}));
    REQUIRE(index);
    UsdStageRefPtr stage = UsdStage::Open(stagePath.string());
    REQUIRE(stage);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices sceneIndices = UsdImagingCreateSceneIndices(info);
    index->InsertSceneIndex(sceneIndices.finalSceneIndex, SdfPath::AbsoluteRootPath());
    HdxTaskController controller(index.get(), SdfPath("/__stormOracle"), /*gpuEnabled=*/true);
    controller.SetEnableSelection(false);
    TfTokenVector outputs{HdAovTokens->depth, HdAovTokens->primId, HdAovTokens->Neye};
    if (colour) {
        outputs.push_back(HdAovTokens->color);
    }
    controller.SetRenderOutputs(outputs);
    if (colour) {
        // Storm's MaterialX shaders read environment uniforms that exist only
        // under a lighting state, as usdview gives it: one light, a dome.
        GlfSimpleLightingContextRefPtr lighting = GlfSimpleLightingContext::New();
        GlfSimpleLight light;
        light.SetIsDomeLight(true);
        lighting->SetLights({light});
        lighting->SetUseLighting(true);
        controller.SetLightingState(lighting);
    }
    HdRprimCollection collection(HdTokens->geometry, HdReprSelector(HdReprTokens->smoothHull));
    collection.SetRootPath(SdfPath::AbsoluteRootPath());
    controller.SetCollection(collection);
    sceneIndices.stageSceneIndex->SetTime(UsdTimeCode(0.0));
    sceneIndices.stageSceneIndex->ApplyPendingUpdates();
    controller.SetFreeCameraMatrices(viewOf(camera), projectionOf(camera, w, h));
    controller.SetRenderBufferSize(GfVec2i(static_cast<int>(w), static_cast<int>(h)));
    controller.SetFraming(CameraUtilFraming(GfRect2i(GfVec2i(0), static_cast<int>(w), static_cast<int>(h))));
    HdEngine engine;
    HdTaskSharedPtrVector tasks = controller.GetRenderingTasks();
    engine.Execute(index.get(), &tasks);
    Outputs out;
    out.depth = mapped(controller.GetRenderOutput(HdAovTokens->depth));
    out.eye = mapped(controller.GetRenderOutput(HdAovTokens->Neye));
    out.primId = mapped(controller.GetRenderOutput(HdAovTokens->primId));
    if (colour) {
        HdRenderBuffer* buffer = controller.GetRenderOutput(HdAovTokens->color);
        REQUIRE(buffer->GetFormat() == HdFormatFloat16Vec4);
        out.colour = mapped(buffer);
    }
    return out;
}

Outputs engineOutputs(usd::StageRenderer& renderer, const render::Camera& camera, uint32_t w, uint32_t h) {
    auto image = renderer.render(camera, 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    Outputs out;
    auto primId = renderer.mappedOutput("primId");
    auto depth = renderer.mappedOutput("depth");
    auto eye = renderer.mappedOutput("Neye");
    REQUIRE(primId);
    REQUIRE(depth);
    REQUIRE(eye);
    out.primId = std::move(*primId);
    out.depth = std::move(*depth);
    out.eye = std::move(*eye);
    return out;
}

}   // namespace

TEST_CASE("Kitchen_set's geometry matches Storm's on every visibility route: prim ids, depth and eye normals",
          "[usd][gpu][oracle][storm]") {
    LRT_REQUIRE_GPU(gpu);
    const fs::path kitchen = fs::path(std::getenv("HOME")) / "tools/assets/Kitchen_set/Kitchen_set.usd";
    if (!fs::exists(kitchen)) {
        SKIP("Pixar's Kitchen_set is not at " << kitchen.string());
    }
    // One sample a pixel, as the engine's visibility. Storm multisamples by
    // default, and on Metal an R32Sint buffer cannot be a resolve target: its
    // id buffers come back with their upper bits unwritten. OpenUSD reads its
    // environment settings as its libraries load, so ctest sets this one.
    if (platform::env("HDX_MSAA_SAMPLE_COUNT") != "1") {
        FAIL("Storm must render single-sampled: run with HDX_MSAA_SAMPLE_COUNT=1 in the environment (ctest sets it)");
    }
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(LRT_HYDRA_PLUGIN_DIR).string());
    const uint32_t w = 480;
    const uint32_t h = 270;
    render::Camera camera =
        render::Camera::lookingAt({500.0, -350.0, 350.0}, {0.0, 50.0, 60.0}, {0.0, 0.0, 1.0});
    camera.lens.focal = 20.0;
    camera.lens.nearZ = 1.0;
    camera.lens.farZ = 10000.0;
    auto renderer = usd::StageRenderer::open(kitchen);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->requestOutputs({"primId", "Neye"});
    const Outputs theirs = storm(kitchen, camera, w, h);
    for (const char* route : {"raster", "rays", "bvh"}) {
        if (auto set = (*renderer)->setMeshVisibility(route); !set) FAIL(set.error().toString());
        const Outputs ours = engineOutputs(**renderer, camera, w, h);
        REQUIRE(ours.primId.size() == theirs.primId.size());
        REQUIRE(ours.depth.size() == theirs.depth.size());
        REQUIRE(ours.eye.size() == theirs.eye.size() * 3);   // float3 against RGBA8

        const auto upload = [&](const std::vector<uint8_t>& bytes, const char* label) {
            gpu::BufferDesc desc;
            desc.bytes = bytes.size();
            desc.elementBytes = 4;
            desc.label = label;
            auto made = gpu::Buffer::create(*gpu->device, desc, bytes.data());
            REQUIRE(made);
            return *made;
        };
        gpu::Buffer primA = upload(ours.primId, "ours.primId");
        gpu::Buffer primB = upload(theirs.primId, "storm.primId");
        gpu::Buffer depthA = upload(ours.depth, "ours.depth");
        gpu::Buffer depthB = upload(theirs.depth, "storm.depth");
        gpu::Buffer eyeA = upload(ours.eye, "ours.Neye");
        gpu::Buffer eyeB = upload(theirs.eye, "storm.Neye");
        auto compare = gpu::ComputeKernel::create(*gpu->library, "lrt/test/oracle_compare", "oracleCompare");
        if (!compare) FAIL(compare.error().toString());
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 9, "oracle.counts");
        gpu::BufferDesc one;
        one.bytes = 4;
        one.elementBytes = 4;
        auto worst = gpu::Buffer::create(*gpu->device, one);
        REQUIRE(worst);
        gpu::CommandBatch batch(*gpu->device);
        compare->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["primA"].setBinding(primA.rhi());
            cursor["primB"].setBinding(primB.rhi());
            cursor["depthA"].setBinding(depthA.rhi());
            cursor["depthB"].setBinding(depthB.rhi());
            cursor["eyeA"].setBinding(eyeA.rhi());
            cursor["eyeB"].setBinding(eyeB.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst->rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
            cursor["params"]["depthEpsilon"].setData(1e-5F);
            cursor["params"]["byteTolerance"].setData(uint32_t{6});
        });
        REQUIRE(batch.submit(true));
        uint32_t c[9] = {};
        float depthWorst = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst->read(*gpu->device, 0, sizeof(depthWorst), &depthWorst));
        const uint32_t segmentation = c[1] + c[8];
        std::printf("  Kitchen_set by %s against Storm: covered %u / %u (%u differ); %u of %u pixels one prim across 3x3 "
                    "in only one image; %u interior pixels: %u depths > 1e-5 (worst %.2e), %u normals > 6/255\n",
                    route, c[4], c[5], c[6], segmentation, c[7], c[0], c[2], double(depthWorst), c[3]);
        CHECK(c[0] > 50000);
        CHECK(uint64_t{c[6]} * 1000 <= uint64_t{w} * h);   // coverage: under 0.1% of the image
        CHECK(uint64_t{segmentation} * 1000 <= c[7]);
        CHECK(uint64_t{c[2]} * 1000 <= c[0]);
        CHECK(uint64_t{c[3]} * 1000 <= c[0]);
    }
}

namespace {

struct SuiteMaterial {
    std::string file;     // the TestSuite file, relative
    std::string output;   // nodegraph/output, or output
    std::string path;     // the material's prim on the stage
};

}   // namespace

// Hidden ([.]): Storm here (OpenUSD 26.08, MaterialX 1.39.5, MSL on Metal)
// compiles none of these shaders -- its MaterialX code reads u_envRadianceMips,
// u_envMatrix and u_envLightIntensity it never declares, with or without a
// lighting state and a dome -- and draws its fallback. Run by name where
// Storm's MaterialX shaders build: [storm-materialx].
TEST_CASE("MaterialX pattern graphs from MaterialX's TestSuite shade as Storm shades them",
          "[.][usd][gpu][oracle][storm-materialx]") {
    LRT_REQUIRE_GPU(gpu);
    const fs::path suite = fs::path(LRT_USD_ROOT) / "src/MaterialX-1.39.5/resources/Materials/TestSuite/stdlib";
    if (!fs::exists(suite)) {
        SKIP("MaterialX's TestSuite is not at " << suite.string());
    }
    if (platform::env("HDX_MSAA_SAMPLE_COUNT") != "1") {
        FAIL("Storm must render single-sampled: run with HDX_MSAA_SAMPLE_COUNT=1 in the environment (ctest sets it)");
    }
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(LRT_HYDRA_PLUGIN_DIR).string());
    namespace mx = MaterialX;
    mx::DocumentPtr libraries = mx::createDocument();
    mx::FileSearchPath libraryPath;
    libraryPath.append(mx::FilePath(LRT_USD_ROOT));
    mx::loadLibraries({"libraries"}, libraryPath, libraries);

    // Every output of the files' graphs as an unlit material's emission.
    const std::vector<std::string> files{"math/math.mtlx",       "math/trig.mtlx",          "math/vector_math.mtlx",
                                         "math/math_operators.mtlx", "noise/noise.mtlx",    "noise/procedural.mtlx",
                                         "compositing/compositing.mtlx", "adjustment/remap.mtlx",
                                         "adjustment/smoothstep.mtlx", "adjustment/luminance.mtlx",
                                         "adjustment/hsvtorgb.mtlx"};
    const fs::path dir = fs::temp_directory_path() / "lrt-tests" / "oracle-materialx";
    fs::create_directories(dir);
    std::vector<SuiteMaterial> materials;
    std::string referencesUsd;
    for (size_t f = 0; f < files.size(); ++f) {
        mx::DocumentPtr doc = mx::createDocument();
        mx::readFromXmlFile(doc, mx::FilePath((suite / files[f]).string()));
        std::vector<std::pair<mx::NodeGraphPtr, mx::OutputPtr>> outputs;
        for (const mx::OutputPtr& out : doc->getOutputs()) {
            outputs.emplace_back(nullptr, out);
        }
        for (const mx::NodeGraphPtr& graph : doc->getNodeGraphs()) {
            if (graph->getNodeDefString().empty()) {
                for (const mx::OutputPtr& out : graph->getOutputs()) {
                    outputs.emplace_back(graph, out);
                }
            }
        }
        size_t made = 0;
        for (const auto& [graph, out] : outputs) {
            const std::string type = out->getType();
            const std::string index = std::to_string(made);
            mx::NodePtr unlit = doc->addNode("surface_unlit", "lrt_unlit_" + index, "surfaceshader");
            mx::InputPtr emission = unlit->addInput("emission_color", "color3");
            const auto connect = [&](const mx::InputPtr& input) {
                if (graph) {
                    input->setNodeGraphString(graph->getName());
                }
                input->setOutputString(out->getName());
            };
            if (type == "color3") {
                connect(emission);
            } else if (libraries->getNodeDef("ND_convert_" + type + "_color3")) {
                mx::NodePtr convert = doc->addNode("convert", "lrt_convert_" + index, "color3");
                connect(convert->addInput("in", type));
                emission->setNodeName(convert->getName());
            } else {
                doc->removeNode(unlit->getName());
                continue;
            }
            doc->addMaterialNode("lrt_material_" + index, unlit);
            materials.push_back({files[f], (graph ? graph->getName() + "/" : std::string()) + out->getName(),
                                 "/Materials/file" + std::to_string(f) + "/Materials/lrt_material_" + index});
            ++made;
        }
        const fs::path written = dir / ("suite" + std::to_string(f) + ".mtlx");
        mx::writeToXmlFile(doc, mx::FilePath(written.string()));
        referencesUsd += "    def \"file" + std::to_string(f) + "\" (\n        prepend references = @" + written.string() +
                         "@</MaterialX>\n    )\n    {\n    }\n";
    }
    const uint32_t columns = static_cast<uint32_t>(std::ceil(std::sqrt(double(materials.size()))));
    const fs::path stagePath = dir / "suite.usda";
    {
        std::ofstream out(stagePath);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n";
        for (size_t k = 0; k < materials.size(); ++k) {
            const double x = double(k % columns) * 1.25;
            const double y = double(k / columns) * 1.25;
            out << "def Mesh \"Q" << k << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                << "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                << "    point3f[] points = [(" << x << ", " << y << ", 0), (" << x + 1 << ", " << y << ", 0), ("
                << x + 1 << ", " << y + 1 << ", 0), (" << x << ", " << y + 1 << ", 0)]\n"
                << "    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] ( interpolation = \"vertex\" )\n"
                << "    uniform token subdivisionScheme = \"none\"\n"
                << "    rel material:binding = <" << materials[k].path << ">\n}\n";
        }
        out << "def Scope \"Materials\"\n{\n" << referencesUsd << "}\n";
        // Storm's MaterialX shaders declare their environment lighting only
        // where there is a dome; unlit materials do not read it.
        out << "def DomeLight \"Dome\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    const uint32_t w = 960;
    const uint32_t h = 960;
    const double extent = double(columns) * 1.25;
    render::Camera camera = render::Camera::lookingAt({extent / 2, extent / 2, extent * 1.3},
                                                      {extent / 2, extent / 2, 0.0}, {0.0, 1.0, 0.0});
    camera.lens.focal = 40.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = extent * 10.0;

    auto renderer = usd::StageRenderer::open(stagePath);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->requestOutputs({"primId"});
    auto image = (*renderer)->render(camera, 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    auto primIds = (*renderer)->mappedOutput("primId");
    auto depth = (*renderer)->mappedOutput("depth");
    REQUIRE(primIds);
    REQUIRE(depth);
    const Outputs theirs = storm(stagePath, camera, w, h, /*colour=*/true);

    const auto upload = [&](const uint8_t* bytes, size_t size, uint32_t element, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<size_t>(size, element);
        desc.elementBytes = element;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, bytes);
        REQUIRE(made);
        return *made;
    };
    const uint32_t prims = static_cast<uint32_t>(materials.size()) + 64;
    gpu::Buffer ours = upload(reinterpret_cast<const uint8_t*>(image->rgba.data()), image->rgba.size() * 4, 16, "ours");
    gpu::Buffer storms = upload(theirs.colour.data(), theirs.colour.size(), 4, "storm");
    gpu::Buffer ourDepth = upload(depth->data(), depth->size(), 4, "ours.depth");
    gpu::Buffer stormDepth = upload(theirs.depth.data(), theirs.depth.size(), 4, "storm.depth");
    gpu::Buffer prim = upload(primIds->data(), primIds->size(), 4, "ours.prim");
    gpu::Buffer counts = test::uintBuffer(*gpu->device, uint64_t{prims} * 3, "colour.counts");
    gpu::Buffer worst = upload(nullptr, uint64_t{prims} * 16, 16, "colour.worst");
    auto kernel = gpu::ComputeKernel::create(*gpu->library, "lrt/test/oracle_colour", "oracleColour");
    if (!kernel) FAIL(kernel.error().toString());
    {
        gpu::CommandBatch batch(*gpu->device);
        kernel->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["ours"].setBinding(ours.rhi());
            cursor["storm"].setBinding(storms.rhi());
            cursor["ourDepth"].setBinding(ourDepth.rhi());
            cursor["stormDepth"].setBinding(stormDepth.rhi());
            cursor["prim"].setBinding(prim.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["height"].setData(h);
            cursor["params"]["prims"].setData(prims);
            cursor["params"]["tolerance"].setData(2e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    std::vector<uint32_t> c(uint64_t{prims} * 3);
    std::vector<float> wv(uint64_t{prims} * 4);
    REQUIRE(counts.read(*gpu->device, 0, c.size() * 4, c.data()));
    REQUIRE(worst.read(*gpu->device, 0, wv.size() * 4, wv.data()));
    size_t compared = 0;
    size_t failing = 0;
    for (uint32_t p = 0; p < prims; ++p) {
        if (c[p * 3 + 1] == 0) {
            continue;
        }
        ++compared;
        if (c[p * 3] == 0) {
            continue;
        }
        ++failing;
        const uint32_t pixel = c[p * 3 + 2] - 1;
        auto picked = (*renderer)->pick(pixel % w, h - 1 - pixel / w);
        std::string name = "?";
        if (picked && picked->has_value()) {
            const std::string& path = (*picked)->prim;
            const size_t k = std::stoul(path.substr(path.rfind('Q') + 1));
            name = materials[k].file + " " + materials[k].output;
        }
        std::printf("  differs: %s -- %u of %u pixels, worst %.3g (green %.5f against %.5f)\n", name.c_str(),
                    c[p * 3], c[p * 3 + 1], double(wv[p * 4]), double(wv[p * 4 + 1]), double(wv[p * 4 + 2]));
    }
    std::printf("  %zu TestSuite outputs as materials; %zu quads compared with Storm, %zu differ\n", materials.size(),
                compared, failing);
    CHECK(compared * 10 >= materials.size() * 9);
    CHECK(failing == 0);
}
