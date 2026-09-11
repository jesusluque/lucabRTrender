// Copyright (c) 2026 lucabRTrender contributors.
//
// lrt view opens, draws a stage through its own display and panels, and closes:
// a smoke test, skipped where the session cannot open windows.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "lrt/view/Viewer.h"
#include "lrt/view/Window.h"

using namespace lrt;
namespace fs = std::filesystem;

TEST_CASE("lrt view draws a stage and its panels in a window", "[view][window]") {
    LRT_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization || !view::Window::available()) {
        SKIP("no window session or no rasterisation");
    }
    const fs::path dir = fs::temp_directory_path() / "lrt-tests" / "view";
    fs::create_directories(dir);
    const fs::path stage = dir / "square.usda";
    {
        std::ofstream out(stage);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    uniform bool doubleSided = 1\n"
               "    color3f[] primvars:displayColor = [(1, 0.6, 0.2)] ( interpolation = \"constant\" )\n"
               "}\n";
    }
    view::ViewOptions options;
    options.stage = stage;
    options.width = 480;
    options.height = 320;
    options.frames = 4;
    options.visible = false;
    options.snapshot = dir / "square_view.exr";
    auto stats = view::runViewer(options);
    if (!stats && stats.error().code() == ErrorCode::Unsupported) {
        SKIP(stats.error().toString());
    }
    if (!stats) FAIL(stats.error().toString());
    std::printf("  %u frames, draw %.2f ms, frame %.2f ms; %llu pixels lit\n", stats->frames, stats->medianDrawMs,
                stats->medianFrameMs, static_cast<unsigned long long>(stats->snapshotLitPixels));
    CHECK(stats->frames == 4);
    // The framed square fills a good part of the window, panels besides.
    CHECK(stats->snapshotLitPixels > 10000);
}
