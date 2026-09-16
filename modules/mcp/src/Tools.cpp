// Copyright (c) 2026 lucabRTrender contributors.
//
// Every tool the renderer offers, in one table. What a caller can do here is
// what `lrt` can do from a shell -- open a stage, look at it, set what a frame
// costs, render it, read a pixel, convert a capture -- with the device and the
// stage kept warm between calls, which a shell cannot do.
//
// A render answers with the image as well as the numbers. A model that cannot
// see what it rendered is guessing, and the display transform that makes those
// pixels is the same kernel `lrt view` shows through, so what comes back is
// what the viewer would show.
#include "Tools.h"

#include <array>
#include <cstdio>
#include <sstream>

#include "lrt/core/Platform.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/Texture.h"
#include "lrt/io/Exr.h"
#include "lrt/io/Png.h"
#include "lrt/render/Camera.h"
#include "lrt/technique/Denoiser.h"
#include "lrt/technique/DisplayTransform.h"
#include "lrt/usd/Export.h"

namespace lrt::mcp {
namespace {

using nlohmann::json;

json object(std::initializer_list<std::pair<const std::string, json>> properties, std::vector<std::string> required = {}) {
    json schema{{"type", "object"}, {"properties", json::object()}};
    for (const auto& [name, property] : properties) {
        schema["properties"][name] = property;
    }
    if (!required.empty()) {
        schema["required"] = required;
    }
    return schema;
}

json field(const char* type, const char* description) {
    return json{{"type", type}, {"description", description}};
}

std::string errorText(const Error& error) {
    return error.toString();
}

/// Opens the stage if a path is given and none is open, or if a different one
/// is asked for. The renderer is what costs: keeping it is the point.
Result<void> ensureStage(Session& session, const json& args) {
    const std::string path = args.value("stage", std::string{});
    if (!path.empty() && (session.stage != std::filesystem::path(path) || !session.open())) {
        auto opened = usd::StageRenderer::open(path);
        if (!opened) return std::move(opened).error();
        session.renderer = std::move(*opened);
        session.stage = path;
        return ok();
    }
    if (!session.open()) {
        return Error(ErrorCode::InvalidArgument, "no stage is open: call open_stage, or pass 'stage'");
    }
    return ok();
}

/// The settings a frame is drawn with, all optional, all remembered.
void applySettings(Session& session, const json& args) {
    usd::StageRenderer& r = *session.renderer;
    if (args.contains("lightSamples")) r.setLightSamples(args["lightSamples"].get<uint32_t>());
    if (args.contains("chooseLights")) r.setChooseLights(args["chooseLights"].get<bool>());
    if (args.contains("splatShadows")) r.setSplatShadows(args["splatShadows"].get<bool>());
    if (args.contains("pathSamples")) r.setPathSamples(args["pathSamples"].get<uint32_t>());
    if (args.contains("pathBounces")) r.setPathBounces(args["pathBounces"].get<uint32_t>());
    if (args.contains("pathTotal")) r.setPathTotal(args["pathTotal"].get<uint32_t>());
    if (args.contains("denoise")) r.setDenoise(args["denoise"].get<bool>());
    if (args.contains("motionBuckets")) r.setMotionBuckets(args["motionBuckets"].get<uint32_t>());
    if (args.contains("refine")) r.setRefineLevel(args["refine"].get<uint32_t>());
    if (args.contains("visibility")) {
        (void)r.setMeshVisibility(args["visibility"].get<std::string>());
    }
}

/// The frame as the viewer would show it: the display transform on the device,
/// read back as eight-bit rows and packed into a PNG.
Result<std::vector<uint8_t>> preview(usd::StageRenderer& renderer, const std::string& aov, uint32_t width,
                                     uint32_t height, const std::string& view, float exposure) {
    auto source = renderer.displaySource(aov);
    if (!source) return std::move(source).error();
    auto display = technique::DisplayTransform::create(renderer.library());
    if (!display) return std::move(display).error();
    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = rhi::Format::RGBA8Unorm;
    desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource |
                 rhi::TextureUsage::CopySource;
    desc.label = "mcp.preview";
    auto texture = gpu::Texture::create(renderer.device(), desc);
    if (!texture) return std::move(texture).error();
    technique::DisplaySettings settings;
    settings.view = view == "standard"  ? technique::ViewTransform::Standard
                    : view == "aces"    ? technique::ViewTransform::Aces2
                                        : technique::ViewTransform::AgX;
    settings.display = technique::DisplayEncoding::Srgb;
    settings.exposure = exposure;
    {
        gpu::CommandBatch batch(renderer.device());
        LRT_TRY(display->run(batch, *source, settings, texture->rhi(), width, height));
        LRT_TRY(batch.submit(true));
    }
    auto texels = texture->read(renderer.device(), 0, 0);
    if (!texels) return std::move(texels).error();
    std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(texels->data()), texels->size());
    return io::encodePng(width, height, bytes);
}

json imageAnswer(std::string text, const std::vector<uint8_t>& png) {
    static constexpr char kBase64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((png.size() + 2) / 3 * 4);
    for (size_t at = 0; at < png.size(); at += 3) {
        const uint32_t a = png[at];
        const uint32_t b = at + 1 < png.size() ? png[at + 1] : 0;
        const uint32_t c = at + 2 < png.size() ? png[at + 2] : 0;
        const uint32_t word = (a << 16) | (b << 8) | c;
        encoded += kBase64[(word >> 18) & 63];
        encoded += kBase64[(word >> 12) & 63];
        encoded += at + 1 < png.size() ? kBase64[(word >> 6) & 63] : '=';
        encoded += at + 2 < png.size() ? kBase64[word & 63] : '=';
    }
    return json{{"content", json::array({json{{"type", "text"}, {"text", std::move(text)}},
                                         json{{"type", "image"}, {"data", std::move(encoded)},
                                              {"mimeType", "image/png"}}})}};
}

// ---- the tools ---------------------------------------------------------------

json openStage(Session& session, const json& args) {
    const std::string path = args.value("stage", std::string{});
    if (path.empty()) {
        return textAnswer("open_stage wants 'stage': a .usd, .usda or .usdc path.", true);
    }
    auto opened = usd::StageRenderer::open(path);
    if (!opened) return textAnswer(errorText(opened.error()), true);
    session.renderer = std::move(*opened);
    session.stage = path;
    std::ostringstream out;
    out << "Opened " << path << "\n";
    out << "up axis: " << session.renderer->upAxis() << ", time codes to " << session.renderer->endTimeCode() << "\n";
    const std::vector<std::string> cameras = session.renderer->cameras();
    out << "cameras: " << (cameras.empty() ? std::string("none (render frames one of its own)") : std::string());
    for (const std::string& camera : cameras) {
        out << "\n  " << camera;
    }
    return textAnswer(out.str());
}

json stageTree(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    const std::string path = args.value("path", std::string("/"));
    std::ostringstream out;
    out << path << "\n";
    for (const usd::StagePrim& child : session.renderer->children(path)) {
        out << "  " << child.name << "  " << child.type << (child.hasChildren ? "  [+]" : "")
            << (child.instance ? "  (instance)" : "") << "\n";
    }
    return textAnswer(out.str());
}

json deviceInfo(Session& session, const json& args) {
    (void)args;
    std::ostringstream out;
    if (!session.open()) {
        out << "no stage open, so no device yet: open_stage first, or ask about a stage.\n";
        return textAnswer(out.str());
    }
    const gpu::Caps& caps = session.renderer->device().caps();
    out << "backend        " << caps.apiName << " on " << caps.adapterName << "\n";
    out << "rasterisation  " << (caps.rasterization ? "yes" : "no") << "\n";
    out << "ray queries    " << (caps.rayQuery ? "yes (inline)" : "no") << "\n";
    out << "ray pipelines  " << (caps.rayTracing ? "yes" : "no") << "\n";
    out << "structures     " << (caps.accelerationStructure ? "yes" : "no") << "\n";
    if (caps.optixVersion != 0) {
        out << "optix          " << caps.optixVersion << "\n";
    }
    out << "half           " << (caps.half ? "yes" : "no") << "\n";
    out << "converting stores " << (caps.convertingStores ? "yes" : "no") << "\n";
    out << "denoiser       " << (technique::denoiserBuilt() ? "OIDN" : "none") << "\n";
    out << "ocio           " << (technique::ocioBuilt() ? "yes" : "no") << "\n";
    return textAnswer(out.str());
}

json renderTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    applySettings(session, args);
    usd::StageRenderer& renderer = *session.renderer;
    const uint32_t width = args.value("width", 640u);
    const uint32_t height = args.value("height", 360u);
    const double time = args.value("time", 0.0);
    const std::string technique = args.value("technique", std::string("raster"));
    const std::string camera = args.value("camera", std::string{});
    const std::vector<std::string> outputs =
        args.contains("aovs") ? args["aovs"].get<std::vector<std::string>>() : std::vector<std::string>{};
    if (!outputs.empty()) {
        renderer.requestOutputs(outputs);
    } else {
        renderer.requestOutputs({"primId", "instanceId"});
    }
    const auto start = std::chrono::steady_clock::now();
    Result<usd::StageImage> image = Error(ErrorCode::InternalError, "not rendered");
    // A camera of the caller's own beats anything on the stage: it is how you
    // look at something in particular rather than at everything at once.
    if (args.contains("eye") && args.contains("target")) {
        const std::vector<double> eye = args["eye"].get<std::vector<double>>();
        const std::vector<double> at = args["target"].get<std::vector<double>>();
        if (eye.size() != 3 || at.size() != 3) {
            return textAnswer("'eye' and 'target' want three numbers each.", true);
        }
        const std::vector<double> up =
            args.contains("up") ? args["up"].get<std::vector<double>>() : std::vector<double>{0.0, 1.0, 0.0};
        render::Camera own = render::Camera::lookingAt({eye[0], eye[1], eye[2]}, {at[0], at[1], at[2]},
                                                       {up[0], up[1], up[2]});
        own.lens.focal = args.value("focal", 35.0);
        image = renderer.render(own, time, width, height, technique);
    } else if (camera.empty() && renderer.cameras().empty()) {
        auto framed = renderer.framingCamera(time, args.value("focal", 35.0), technique);
        if (!framed) return textAnswer(errorText(framed.error()), true);
        image = renderer.render(*framed, time, width, height, technique);
    } else {
        image = renderer.render(camera, time, width, height, technique);
    }
    if (!image) return textAnswer(errorText(image.error()), true);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    session.lastWidth = width;
    session.lastHeight = height;
    session.lastTime = time;
    session.lastCamera = camera;

    std::ostringstream out;
    out << "Rendered " << session.stage.filename().string() << " at " << width << "x" << height << ", " << technique
        << ", time " << time << " in " << std::fixed << ms << " ms";
    if (const std::string exr = args.value("output", std::string{}); !exr.empty()) {
        if (auto written = io::writeExr(exr, width, height, image->rgba, image->depth, false); !written) {
            out << "\nEXR not written: " << errorText(written.error());
        } else {
            out << "\nwrote " << exr;
        }
    }
    const std::string aov = args.value("aov", std::string("color"));
    auto png = preview(renderer, aov, width, height, args.value("view", std::string("agx")),
                       static_cast<float>(args.value("exposure", 0.0)));
    if (!png) {
        out << "\nno preview: " << errorText(png.error());
        return textAnswer(out.str());
    }
    return imageAnswer(out.str(), *png);
}

json pickTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    if (session.lastWidth == 0) {
        return textAnswer("nothing has been rendered yet: render first, then pick a pixel of that frame.", true);
    }
    const uint32_t x = args.value("x", 0u);
    const uint32_t y = args.value("y", 0u);
    auto picked = session.renderer->pick(x, y);
    if (!picked) return textAnswer(errorText(picked.error()), true);
    if (!picked->has_value()) {
        return textAnswer("nothing at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
    }
    std::ostringstream out;
    out << "prim " << (*picked)->prim << "\nrprim " << (*picked)->rprim;
    if ((*picked)->instance >= 0) {
        out << "\ninstance " << (*picked)->instance;
    }
    return textAnswer(out.str());
}

json boundsTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    auto bounds = session.renderer->bounds();
    if (!bounds) return textAnswer(errorText(bounds.error()), true);
    if (!bounds->has_value()) {
        return textAnswer("nothing drawn yet, so nothing has bounds: render first.");
    }
    const scene::Bounds& b = **bounds;
    std::ostringstream out;
    out << "min " << b.min[0] << ", " << b.min[1] << ", " << b.min[2] << "\n";
    out << "max " << b.max[0] << ", " << b.max[1] << ", " << b.max[2];
    return textAnswer(out.str());
}

json productsTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    const std::string prim = args.value("settings", std::string{});
    if (prim.empty()) {
        return textAnswer("render_products wants 'settings': a UsdRenderSettings prim path.", true);
    }
    auto written = session.renderer->renderProducts(prim, args.value("time", 0.0),
                                                    args.value("directory", std::string{}));
    if (!written) return textAnswer(errorText(written.error()), true);
    std::ostringstream out;
    out << "wrote " << written->size() << " file(s)";
    for (const std::filesystem::path& file : *written) {
        out << "\n  " << file.string();
    }
    return textAnswer(out.str());
}

json convertTool(Session& session, const json& args) {
    (void)session;
    const std::string input = args.value("input", std::string{});
    const std::string output = args.value("output", std::string{});
    if (input.empty() || output.empty()) {
        return textAnswer("convert wants 'input' (a .ply, .splat, .spz or .sog) and 'output' (a .usdc or .lrtc).",
                          true);
    }
    return textAnswer("convert is the CLI's for now: run `lrt convert " + input + " " + output +
                      "`. The MCP server holds a stage, not a converter.");
}

}   // namespace

json textAnswer(std::string text, bool isError) {
    json answer{{"content", json::array({json{{"type", "text"}, {"text", std::move(text)}}})}};
    if (isError) {
        answer["isError"] = true;
    }
    return answer;
}

const std::vector<ToolEntry>& toolTable() {
    static const std::vector<ToolEntry> table = [] {
        std::vector<ToolEntry> tools;
        tools.push_back({"open_stage",
                         "Open a USD stage and keep it open. Answers with its cameras, up axis and time range.",
                         object({{"stage", field("string", "path to a .usd, .usda or .usdc")}}, {"stage"}),
                         openStage});
        tools.push_back({"stage_tree", "What is under a prim path ('/' for the root): name, type, whether it has children.",
                         object({{"path", field("string", "prim path, '/' by default")},
                                 {"stage", field("string", "open this stage first")}}),
                         stageTree});
        tools.push_back({"device_info", "The GPU the engine opened and what it can do: rasterisation, inline rays, ray tracing pipelines, OptiX, the denoiser, OCIO.",
                         object({}), deviceInfo});
        tools.push_back(
            {"render",
             "Render the open stage and answer with the image and its timing. 'technique' is raster or rt (path "
             "traced); path tracing draws until it holds 'pathTotal' paths. Every setting is remembered for the "
             "next call.",
             object({{"stage", field("string", "open this stage first")},
                     {"camera", field("string", "a camera prim; omitted, the first, or a framing camera")},
                     {"width", field("integer", "pixels across, 640 by default")},
                     {"height", field("integer", "pixels down, 360 by default")},
                     {"time", field("number", "USD time code")},
                     {"technique", field("string", "raster | rt")},
                     {"aov", field("string", "which AOV the image shows: color, depth, primId, Neye, normal, albedo")},
                     {"aovs", json{{"type", "array"}, {"items", {{"type", "string"}}},
                                   {"description", "AOVs to compute this frame"}}},
                     {"view", field("string", "view transform for the image: agx (default), aces, standard")},
                     {"exposure", field("number", "stops, for the image only")},
                     {"output", field("string", "also write the frame to this EXR")},
                     {"eye", json{{"type", "array"}, {"items", {{"type", "number"}}},
                                  {"description", "a camera of your own: where it is (x y z), with 'target'"}}},
                     {"target", json{{"type", "array"}, {"items", {{"type", "number"}}},
                                     {"description", "where that camera looks"}}},
                     {"up", json{{"type", "array"}, {"items", {{"type", "number"}}}, {"description", "its up vector"}}},
                     {"focal", field("number", "focal length, mm, of a camera of your own or a framing one")},
                     {"lightSamples", field("integer", "samples per light per pixel")},
                     {"chooseLights", field("boolean", "one light a sample, chosen by power")},
                     {"splatShadows", field("boolean", "a relit splat cloud shadows itself")},
                     {"pathSamples", field("integer", "rt: paths a pixel each pass")},
                     {"pathBounces", field("integer", "rt: bounces after the first hit")},
                     {"pathTotal", field("integer", "rt: paths a pixel the image is drawn until it holds")},
                     {"denoise", field("boolean", "rt: denoise once the total is reached (OIDN)")},
                     {"motionBuckets", field("integer", "rt: shutter slices")},
                     {"refine", field("integer", "subdivision levels")},
                     {"visibility", field("string", "how meshes are found: automatic | raster | rays | bvh")}}),
             renderTool});
        tools.push_back({"pick", "What the last frame drew at a pixel: the prim, and which instance of it.",
                         object({{"x", field("integer", "from the left")}, {"y", field("integer", "from the top")}},
                                {"x", "y"}),
                         pickTool});
        tools.push_back({"bounds", "Where what the last frame drew is, in world space.", object({}), boundsTool});
        tools.push_back({"render_products",
                         "Render a UsdRenderSettings prim's products: every render var a layer of its EXR.",
                         object({{"settings", field("string", "the settings prim path")},
                                 {"time", field("number", "USD time code")},
                                 {"directory", field("string", "where the files go")}},
                                {"settings"}),
                         productsTool});
        tools.push_back({"convert", "How to turn a splat capture into a stage this server can open.",
                         object({{"input", field("string", "a .ply, .splat, .spz or .sog")},
                                 {"output", field("string", "a .usdc or .lrtc")}}),
                         convertTool});
        return tools;
    }();
    return table;
}

}   // namespace lrt::mcp
