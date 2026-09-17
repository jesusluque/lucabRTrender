// Copyright (c) 2026 lucabRTrender contributors.
#include "lrt/view/Viewer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <imgui.h>
#include <imgui_impl_glfw.h>

#include "lrt/core/Log.h"
#include "lrt/gpu/Texture.h"
#include "lrt/io/Exr.h"
#include "lrt/gpu/CommandBatch.h"
#include "lrt/gpu/Device.h"
#include "lrt/gpu/ShaderLibrary.h"
#include "lrt/technique/DisplayTransform.h"
#include "lrt/usd/StageRenderer.h"
#include "lrt/view/ImGuiRenderer.h"
#include "lrt/view/Window.h"

namespace lrt::view {

namespace {

constexpr rhi::Format kSurfaceFormat = rhi::Format::BGRA8Unorm;   // encoded by the display transform
constexpr std::array<float, 3> kBackground{0.0F, 0.0F, 0.0F};

/// The frame as shown, panels included, into a float texture, read back and
/// written: output, the one image this viewer reads. Returns how many pixels
/// the display made brighter than the background, counted by a kernel.
Result<uint64_t> snapshot(usd::StageRenderer& stage, technique::DisplayTransform& display, ImGuiRenderer& ui,
                          const technique::DisplaySettings& settings, const std::string& aov, uint32_t width,
                          uint32_t height, const std::filesystem::path& path) {
    gpu::Device& device = stage.device();
    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = rhi::Format::RGBA32Float;
    desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::RenderTarget |
                 rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySource;
    desc.label = "view.snapshot";
    auto texture = gpu::Texture::create(device, desc);
    if (!texture) return std::move(texture).error();
    auto source = stage.displaySource(aov);
    if (!source) return std::move(source).error();
    auto view = texture->view(0);
    if (!view) return std::move(view).error();
    auto count = gpu::ComputeKernel::create(stage.library(), "lrt/view/snapshot_count", "snapshotCount");
    if (!count) return std::move(count).error();
    gpu::BufferDesc one;
    one.bytes = 4;
    one.elementBytes = 4;
    one.label = "view.snapshot.lit";
    auto lit = gpu::Buffer::create(device, one);
    if (!lit) return std::move(lit).error();
    gpu::CommandBatch batch(device);
    LRT_TRY(display.run(batch, *source, settings, texture->rhi(), width, height));
    LRT_TRY(ui.render(batch, ImGui::GetDrawData(), (*view).get(), rhi::Format::RGBA32Float, width, height));
    LRT_TRY(batch.submit(true));
    {
        gpu::CommandBatch counting(device);
        count->dispatch(counting, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["shown"].setBinding((*view).get());
            cursor["lit"].setBinding(lit->rhi());
            cursor["params"]["width"].setData(width);
            cursor["params"]["height"].setData(height);
            cursor["params"]["threshold"].setData(0.25F);
        });
        LRT_TRY(counting.submit(true));
    }
    uint32_t litPixels = 0;
    LRT_TRY(lit->read(device, 0, sizeof(litPixels), &litPixels));
    auto bytes = texture->read(device, 0, 0);
    if (!bytes) return std::move(bytes).error();
    // Top row first on the device; bottom row first for the EXR writer.
    std::vector<float> rgba(size_t{width} * height * 4);
    const size_t row = size_t{width} * 4 * sizeof(float);
    for (size_t y = 0; y < height; ++y) {
        std::memcpy(rgba.data() + (height - 1 - y) * size_t{width} * 4, bytes->data() + y * row, row);
    }
    LRT_TRY(io::writeExr(path, width, height, rgba, {}, false));
    return uint64_t{litPixels};
}

/// A camera turning about a point: the free camera.
struct Orbit {
    std::array<double, 3> target{0.0, 0.0, 0.0};
    double                distance = 10.0;
    double                yaw = 0.6;
    double                pitch = 0.35;
    double                radius = 5.0;   ///< the framed scene's, for clipping planes
};

std::array<double, 3> upOf(char axis) {
    return axis == 'Z' ? std::array<double, 3>{0.0, 0.0, 1.0} : std::array<double, 3>{0.0, 1.0, 0.0};
}

render::Camera cameraOf(const Orbit& orbit, char axis, double focal) {
    const double cp = std::cos(orbit.pitch);
    const std::array<double, 3> away = axis == 'Z'
                                           ? std::array<double, 3>{cp * std::sin(orbit.yaw), -cp * std::cos(orbit.yaw),
                                                                   std::sin(orbit.pitch)}
                                           : std::array<double, 3>{cp * std::sin(orbit.yaw), std::sin(orbit.pitch),
                                                                   cp * std::cos(orbit.yaw)};
    const std::array<double, 3> eye{orbit.target[0] + away[0] * orbit.distance,
                                    orbit.target[1] + away[1] * orbit.distance,
                                    orbit.target[2] + away[2] * orbit.distance};
    const std::array<double, 3> up = upOf(axis);
    render::Camera camera = render::Camera::lookingAt({eye[0], eye[1], eye[2]},
                                                      {orbit.target[0], orbit.target[1], orbit.target[2]},
                                                      {up[0], up[1], up[2]});
    camera.lens.focal = focal;
    camera.lens.nearZ = std::max(orbit.distance * 1e-3, 1e-4);
    camera.lens.farZ = orbit.distance + orbit.radius * 8.0 + 1.0;
    return camera;
}

void frameBounds(Orbit& orbit, const scene::Bounds& bounds, double focal) {
    for (size_t k = 0; k < 3; ++k) {
        orbit.target[k] = 0.5 * (double(bounds.min[k]) + double(bounds.max[k]));
    }
    const double dx = double(bounds.max[0]) - double(bounds.min[0]);
    const double dy = double(bounds.max[1]) - double(bounds.min[1]);
    const double dz = double(bounds.max[2]) - double(bounds.min[2]);
    orbit.radius = std::max(0.5 * std::sqrt(dx * dx + dy * dy + dz * dz), 1e-3);
    const double halfFov = std::atan(0.5 * 18.672 / focal);
    orbit.distance = orbit.radius / std::sin(halfFov) * 1.05;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Choice {
    const char* label;
    const char* value;
};

constexpr std::array<Choice, 2> kTechniques{{{"Raster", "raster"}, {"Path traced", "rt"}}};
constexpr std::array<Choice, 4> kVisibility{
    {{"Automatic", "automatic"}, {"Raster", "raster"}, {"Rays", "rays"}, {"Compute BVH", "bvh"}}};
constexpr std::array<Choice, 7> kAovs{{{"Colour", "color"},
                                       {"Depth", "depth"},
                                       {"Prim id", "primId"},
                                       {"Instance id", "instanceId"},
                                       {"Element id", "elementId"},
                                       {"Eye normal", "Neye"},
                                       {"World normal", "normal"}}};

bool combo(const char* label, int& index, std::span<const Choice> choices) {
    bool changed = false;
    if (ImGui::BeginCombo(label, choices[static_cast<size_t>(index)].label)) {
        for (size_t k = 0; k < choices.size(); ++k) {
            if (ImGui::Selectable(choices[k].label, static_cast<size_t>(index) == k)) {
                changed = changed || static_cast<size_t>(index) != k;
                index = static_cast<int>(k);
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

int indexOf(std::span<const Choice> choices, const std::string& value) {
    for (size_t k = 0; k < choices.size(); ++k) {
        if (value == choices[k].value) {
            return static_cast<int>(k);
        }
    }
    return 0;
}

}   // namespace

Result<ViewStats> runViewer(const ViewOptions& options) {
    auto opened = usd::StageRenderer::open(options.stage);
    if (!opened) return std::move(opened).error();
    usd::StageRenderer& stage = **opened;
    gpu::Device& device = stage.device();
    gpu::ShaderLibrary& library = stage.library();

    auto window = Window::open("lrt view - " + options.stage.filename().string(), options.width, options.height,
                               options.visible);
    if (!window) return std::move(window).error();
    rhi::ComPtr<rhi::ISurface> surface = device.rhi()->createSurface((*window)->handle());
    if (!surface) {
        return Error(ErrorCode::DeviceFailure, "cannot make a surface for the window");
    }
    (*window)->matchSurfaceToBacking();
    // Extended range: the surface in floats, linear P3 with 1.0 at the
    // reference white, and ACES 2.0 filling the headroom the screen has.
    rhi::Format surfaceFormat = kSurfaceFormat;
    double headroom = 1.0;
    if (options.edr && (*window)->enableExtendedRange()) {
        surfaceFormat = rhi::Format::RGBA16Float;
        headroom = (*window)->extendedRangeHeadroom();
        lrt::log::info("lrt view: extended range on, headroom {:.2f}", headroom);
    }
    uint32_t surfaceWidth = 0;
    uint32_t surfaceHeight = 0;
    const auto configure = [&](uint32_t w, uint32_t h) -> Result<void> {
        rhi::SurfaceConfig config;
        config.format = surfaceFormat;
        config.usage = rhi::TextureUsage::Present | rhi::TextureUsage::RenderTarget |
                       rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        config.width = w;
        config.height = h;
        config.vsync = true;
        if (SLANG_FAILED(surface->configure(config))) {
            return Error::make(ErrorCode::DeviceFailure, "cannot configure a {}x{} surface", w, h);
        }
        surfaceWidth = w;
        surfaceHeight = h;
        return ok();
    };

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    struct ContextGuard {
        ~ContextGuard() {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
    };
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui_ImplGlfw_InitForOther((*window)->glfw(), true);
    const ContextGuard contextGuard;
    auto ui = ImGuiRenderer::create(library);
    if (!ui) return std::move(ui).error();
    auto display = technique::DisplayTransform::create(library);
    if (!display) return std::move(display).error();
    const bool ocio = !options.ocioConfig.empty() || !options.ocioDisplay.empty() || !options.ocioView.empty();
    if (ocio) {
        technique::OcioView chosen;
        if (!options.ocioConfig.empty()) {
            chosen.config = options.ocioConfig;
        }
        chosen.display = options.ocioDisplay;
        chosen.view = options.ocioView;
        LRT_TRY(display->setOcio(chosen));
        lrt::log::info("lrt view: {}", display->ocioDescription());
    }

    // What the panels set.
    const std::vector<std::string> cameras = stage.cameras();
    int cameraIndex = 0;   // 0: the free camera
    for (size_t k = 0; k < cameras.size(); ++k) {
        if (cameras[k] == options.camera) {
            cameraIndex = static_cast<int>(k) + 1;
        }
    }
    int technique = indexOf(kTechniques, options.technique);
    int visibility = indexOf(kVisibility, options.visibility);
    int aov = 0;
    // OCIO when one was given; ACES 2.0 for extended range; AgX otherwise.
    int viewTransform = ocio ? 3 : surfaceFormat == rhi::Format::RGBA16Float ? 2 : 1;
    int displayEncoding = surfaceFormat == rhi::Format::RGBA16Float ? 3 : 0;
    float exposure = 0.0F;
    float renderScale = 1.0F;
    double time = stage.startTimeCode();
    // The timeline: playing advances the time by the wall clock at the
    // stage's timeCodesPerSecond and wraps at the end. What frame N shows is
    // the stage's business (SetTime); when it is drawn is the clock's.
    bool playing = options.play && stage.endTimeCode() > stage.startTimeCode();
    double drawnTime = time;
    uint32_t distinctTimes = 0;
    auto lastTick = std::chrono::steady_clock::now();
    double focal = 35.0;
    std::optional<usd::StagePick> picked;
    std::string status;
    Orbit orbit;
    bool framed = false;
    const char up = stage.upAxis();
    const auto request = [&] {
        std::vector<std::string> outputs{"primId", "instanceId"};
        const std::string chosen = kAovs[static_cast<size_t>(aov)].value;
        if (chosen != "color" && chosen != "depth" && chosen != "primId" && chosen != "instanceId") {
            outputs.push_back(chosen);
        }
        stage.requestOutputs(outputs);
    };
    request();
    LRT_TRY(stage.setMeshVisibility(kVisibility[static_cast<size_t>(visibility)].value));
    stage.setLightSamples(options.lightSamples);
    stage.setChooseLights(options.chooseLights);
    int pathSamples = static_cast<int>(std::max(options.pathSamples, 1u));
    int pathBounces = static_cast<int>(options.pathBounces);
    int pathTotal = static_cast<int>(std::max(options.pathTotal, 1u));
    bool denoise = options.denoise;
    stage.setPathSamples(static_cast<uint32_t>(pathSamples));
    stage.setPathBounces(static_cast<uint32_t>(pathBounces));
    stage.setPathTotal(static_cast<uint32_t>(pathTotal));
    stage.setDenoise(denoise);
    const bool stageLit = stage.hasLights();
    bool defaultLights = !stageLit && options.defaultLights;
    LRT_TRY(stage.setDefaultLights(defaultLights));
    // The first frame of a technique compiles its kernels for every material
    // on the stage -- minutes, the first time, for a stage like the chess set
    // -- and the window cannot draw while it does. One frame says so first.
    std::array<bool, kTechniques.size()> techniqueDrawn{};
    int announcedFor = -1;

    const auto displaySettings = [&] {
        technique::DisplaySettings settings;
        settings.view = static_cast<technique::ViewTransform>(viewTransform);
        settings.display = static_cast<technique::DisplayEncoding>(displayEncoding);
        settings.peakLuminance = static_cast<float>(100.0 * headroom);
        settings.exposure = exposure;
        settings.background = kBackground;
        settings.nearZ = static_cast<float>(std::max(orbit.distance - orbit.radius, orbit.distance * 1e-2));
        settings.farZ = static_cast<float>(orbit.distance + orbit.radius);
        return settings;
    };
    uint64_t snapshotLit = 0;
    std::vector<double> drawMs;
    std::vector<double> frameMs;
    ImVec2 pressedAt{0.0F, 0.0F};
    uint32_t frames = 0;
    while (!(*window)->shouldClose() && (options.frames == 0 || frames < options.frames)) {
        const auto frameStart = std::chrono::steady_clock::now();
        (*window)->pollEvents();
        const auto [fbw, fbh] = (*window)->framebufferSize();
        if (fbw == 0 || fbh == 0) {
            continue;   // minimised
        }
        if (fbw != surfaceWidth || fbh != surfaceHeight) {
            LRT_TRY(configure(fbw, fbh));
        }
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuiIO& io = ImGui::GetIO();
        const uint32_t rw = std::max<uint32_t>(1, static_cast<uint32_t>(float(fbw) * renderScale));
        const uint32_t rh = std::max<uint32_t>(1, static_cast<uint32_t>(float(fbh) * renderScale));

        // The free camera, from the mouse the panels do not want.
        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                pressedAt = io.MousePos;
            }
            const ImVec2 delta = io.MouseDelta;
            const bool pan = ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                             (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && io.KeyShift);
            if (cameraIndex == 0) {
                if (pan) {
                    const render::Camera camera = cameraOf(orbit, up, focal);
                    const double step = orbit.distance * 0.0015;
                    for (size_t k = 0; k < 3; ++k) {
                        orbit.target[k] += (-camera.cameraToWorld.at(int(k), 0) * delta.x +
                                            camera.cameraToWorld.at(int(k), 1) * delta.y) *
                                           step;
                    }
                } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    orbit.yaw -= delta.x * 0.005;
                    orbit.pitch = std::clamp(orbit.pitch + delta.y * 0.005, -1.55, 1.55);
                } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                    orbit.distance *= std::exp(delta.y * 0.005);
                }
                if (io.MouseWheel != 0.0F) {
                    orbit.distance *= std::pow(0.9, double(io.MouseWheel));
                }
            }
            // A click that did not drag picks.
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !io.KeyShift) {
                const float moved = std::hypot(io.MousePos.x - pressedAt.x, io.MousePos.y - pressedAt.y);
                if (moved < 3.0F) {
                    const float px = io.MousePos.x * io.DisplayFramebufferScale.x * float(rw) / float(fbw);
                    const float py = io.MousePos.y * io.DisplayFramebufferScale.y * float(rh) / float(fbh);
                    if (px >= 0.0F && py >= 0.0F) {
                        auto hit = stage.pick(static_cast<uint32_t>(px), static_cast<uint32_t>(py));
                        if (hit) {
                            picked = *hit;
                        } else {
                            status = hit.error().toString();
                        }
                    }
                }
            }
        }
        if (!io.WantCaptureKeyboard) {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                (*window)->requestClose();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_F)) {
                framed = false;
                cameraIndex = 0;
            }
        }

        // Panels.
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 420), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("View")) {
            const char* cameraLabel = cameraIndex == 0 ? "Free" : cameras[static_cast<size_t>(cameraIndex - 1)].c_str();
            if (ImGui::BeginCombo("Camera", cameraLabel)) {
                if (ImGui::Selectable("Free", cameraIndex == 0)) {
                    cameraIndex = 0;
                }
                for (size_t k = 0; k < cameras.size(); ++k) {
                    if (ImGui::Selectable(cameras[k].c_str(), cameraIndex == static_cast<int>(k) + 1)) {
                        cameraIndex = static_cast<int>(k) + 1;
                    }
                }
                ImGui::EndCombo();
            }
            if (cameraIndex == 0) {
                float f = static_cast<float>(focal);
                if (ImGui::SliderFloat("Focal (mm)", &f, 8.0F, 200.0F, "%.0f")) {
                    focal = f;
                }
                if (ImGui::Button("Frame all (F)")) {
                    framed = false;
                }
            }
            combo("Technique", technique, kTechniques);
            if (!stageLit && ImGui::Checkbox("Default lights (the stage has none)", &defaultLights)) {
                LRT_TRY(stage.setDefaultLights(defaultLights));
            }
            if (kTechniques[static_cast<size_t>(technique)].value == std::string_view("rt")) {
                // The path tracer's own settings: the delegate's defaults are
                // one bounce, which lights a room little more than the raster.
                ImGui::Indent();
                if (ImGui::SliderInt("Paths per frame", &pathSamples, 1, 64)) {
                    stage.setPathSamples(static_cast<uint32_t>(pathSamples));
                }
                if (ImGui::SliderInt("Bounces", &pathBounces, 0, 16)) {
                    stage.setPathBounces(static_cast<uint32_t>(pathBounces));
                }
                if (ImGui::Checkbox("Denoise", &denoise)) {
                    stage.setDenoise(denoise);
                }
                // A viewport keeps gathering while the camera is still and
                // starts again when it moves; the count says which.
                ImGui::Text("%u paths a pixel", stage.pathAccumulated());
                ImGui::Unindent();
            }
            if (combo("Mesh visibility", visibility, kVisibility)) {
                LRT_TRY(stage.setMeshVisibility(kVisibility[static_cast<size_t>(visibility)].value));
            }
            if (combo("Output", aov, kAovs)) {
                request();
            }
            ImGui::Separator();
            const char* views[] = {"Standard", "AgX", "ACES 2.0", "OCIO"};
            ImGui::Combo("View transform", &viewTransform, views, ocio ? 4 : 3);
            if (viewTransform == 3) {
                ImGui::TextUnformatted(display->ocioDescription().c_str());
            }
            const char* displays[] = {"sRGB", "Rec.709 (BT.1886)", "Display P3", "Linear P3 (extended range)"};
            ImGui::Combo("Display", &displayEncoding, displays, surfaceFormat == rhi::Format::RGBA16Float ? 4 : 3);
            ImGui::SliderFloat("Exposure", &exposure, -8.0F, 8.0F, "%.1f stops");
            ImGui::SliderFloat("Render scale", &renderScale, 0.25F, 1.0F, "%.2f");
            const double start = stage.startTimeCode();
            const double end = stage.endTimeCode();
            if (end > start) {
                float t = static_cast<float>(time);
                if (ImGui::SliderFloat("Time", &t, float(start), float(end), "%.1f")) {
                    time = t;
                    playing = false;
                }
                if (ImGui::Button(playing ? "Pause" : "Play")) {
                    playing = !playing;
                    lastTick = std::chrono::steady_clock::now();
                }
                ImGui::SameLine();
                if (ImGui::Button("|<")) {
                    time = start;
                    playing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("<")) {
                    time = std::max(start, std::floor(time - 1.0 + 0.5));
                    playing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button(">")) {
                    time = std::min(end, std::floor(time + 1.0 + 0.5));
                    playing = false;
                }
                ImGui::SameLine();
                ImGui::Text("%.1f fps", stage.timeCodesPerSecond());
            }
            if (playing) {
                const auto now = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(now - lastTick).count();
                lastTick = now;
                time += seconds * stage.timeCodesPerSecond();
                if (time > end) {
                    time = start + std::fmod(time - start, std::max(end - start, 1e-9));
                }
            }
            ImGui::Separator();
            ImGui::Text("%s on %s", device.caps().apiName.c_str(), device.caps().adapterName.c_str());
            ImGui::Text("%u x %u, draw %.2f ms, frame %.2f ms", rw, rh, drawMs.empty() ? 0.0 : drawMs.back(),
                        frameMs.empty() ? 0.0 : frameMs.back());
            if (picked) {
                ImGui::TextWrapped("Picked %s (instance %d)", picked->prim.c_str(), picked->instance);
            }
            if (!status.empty()) {
                ImGui::TextWrapped("%s", status.c_str());
            }
        }
        ImGui::End();
        ImGui::SetNextWindowPos(ImVec2(10, 440), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 420), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Stage")) {
            const std::function<void(const std::string&)> tree = [&](const std::string& path) {
                for (const usd::StagePrim& prim : stage.children(path)) {
                    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                    if (!prim.hasChildren) {
                        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                    }
                    if (picked && picked->prim == prim.path) {
                        flags |= ImGuiTreeNodeFlags_Selected;
                    }
                    const bool open = ImGui::TreeNodeEx(prim.path.c_str(), flags, "%s  %s%s", prim.name.c_str(),
                                                        prim.type.c_str(), prim.instance ? " (instance)" : "");
                    if (open && prim.hasChildren) {
                        tree(prim.path);
                        ImGui::TreePop();
                    }
                }
            };
            tree("/");
        }
        ImGui::End();

        // The frame.
        const auto drawStart = std::chrono::steady_clock::now();
        const std::string techniqueName = kTechniques[static_cast<size_t>(technique)].value;
        Result<void> drawn = ok();
        const size_t techniqueAt = static_cast<size_t>(technique);
        const bool announce = !techniqueDrawn[techniqueAt] && frames > 0 && announcedFor != technique;
        if (announce) {
            // Shown over the last frame; the frame that compiles is the next.
            announcedFor = technique;
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(ImVec2(viewport->Size.x * 0.5F, viewport->Size.y * 0.5F), ImGuiCond_Always,
                                    ImVec2(0.5F, 0.5F));
            ImGui::Begin("##preparing", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);
            ImGui::Text("Preparing %s: its kernels compile for this stage's materials.",
                        kTechniques[techniqueAt].label);
            ImGui::Text("The window waits until they have; next time they come from the cache.");
            ImGui::End();
        } else if (cameraIndex == 0) {
            drawn = stage.draw(cameraOf(orbit, up, focal), time, rw, rh, techniqueName);
            if (drawn && !framed) {
                auto bounds = stage.bounds();
                if (bounds && bounds->has_value()) {
                    frameBounds(orbit, **bounds, focal);
                    framed = true;
                    drawn = stage.draw(cameraOf(orbit, up, focal), time, rw, rh, techniqueName);
                }
            }
        } else {
            drawn = stage.draw(cameras[static_cast<size_t>(cameraIndex - 1)], time, rw, rh, techniqueName);
        }
        drawMs.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - drawStart).count());
        status = drawn ? std::string() : drawn.error().toString();
        if (drawn && !announce) {
            techniqueDrawn[techniqueAt] = true;
        }
        if (drawn) {
            if (distinctTimes == 0 || time != drawnTime) {
                ++distinctTimes;
            }
            drawnTime = time;
        }

        rhi::ComPtr<rhi::ITexture> image = surface->acquireNextImage();
        ImGui::Render();
        if (!image) {
            continue;
        }
        gpu::CommandBatch batch(device);
        if (drawn) {
            auto source = stage.displaySource(kAovs[static_cast<size_t>(aov)].value);
            if (source) {
                LRT_TRY(display->run(batch, *source, displaySettings(), image.get(), fbw, fbh));
            }
        }
        LRT_TRY((*ui)->render(batch, ImGui::GetDrawData(), image->getDefaultView(), surfaceFormat, fbw, fbh));
        LRT_TRY(batch.submit(false));
        const bool last = options.frames != 0 && frames + 1 == options.frames;
        if (last && !options.snapshot.empty() && drawn) {
            auto shot = snapshot(stage, *display, **ui, displaySettings(), kAovs[static_cast<size_t>(aov)].value,
                                 fbw, fbh, options.snapshot);
            if (!shot) return std::move(shot).error();
            snapshotLit = *shot;
        }
        if (SLANG_FAILED(surface->present())) {
            return Error(ErrorCode::DeviceFailure, "cannot present to the window");
        }
        ++frames;
        frameMs.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count());
    }
    {
        gpu::CommandBatch finish(device);
        LRT_TRY(finish.submit(true));
    }
    ui->reset();
    surface.setNull();
    ViewStats stats;
    stats.frames = frames;
    stats.lastTime = drawnTime;
    stats.distinctTimes = distinctTimes;
    stats.snapshotLitPixels = snapshotLit;
    stats.medianDrawMs = median(drawMs);
    stats.medianFrameMs = median(frameMs);
    return stats;
}

}   // namespace lrt::view
