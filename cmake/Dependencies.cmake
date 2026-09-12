# Third-party dependencies, fetched at configure time and pinned.
#
# Two are not fetched, on purpose:
#   - Slang itself: one version for the whole process, the one gpe's slangc
#     compiles blobs with (~/tools/slang, 2026.14.1) -- slang-rhi links that
#     same install. Two Slang runtimes in one process is not a thing to debug.
#   - OpenUSD: tens of minutes to build and its own oneTBB, so it is built once
#     per machine by scripts/build-usd.sh and found here (cmake/Usd.cmake).

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# --- Slang --------------------------------------------------------------------

set(SLANG_ROOT "$ENV{SLANG_ROOT}" CACHE PATH "Slang release install (bin/, lib/, include/)")
if(NOT SLANG_ROOT)
    set(SLANG_ROOT "$ENV{HOME}/tools/slang" CACHE PATH "" FORCE)
endif()
if(NOT EXISTS "${SLANG_ROOT}/include/slang.h")
    message(FATAL_ERROR
        "Slang not found at ${SLANG_ROOT}. Unpack the 2026.14.1 release from "
        "github.com/shader-slang/slang there, or set SLANG_ROOT.")
endif()
# gpe searches for slangc itself; pointing its search at the same install keeps
# build-time blobs and run-time compilation on one compiler.
set(ENV{SLANG_ROOT} "${SLANG_ROOT}")
find_program(GPE_SLANGC slangc HINTS "${SLANG_ROOT}/bin" NO_DEFAULT_PATH REQUIRED)

# --- slang-rhi -----------------------------------------------------------------
#
# Pinned to a commit rather than a tag: slang-rhi does not cut releases, and
# its API moves. e17f6d7 is 2026-09-03 ("Add opacity micromap support").

set(SLANG_RHI_FETCH_SLANG OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_SLANG_INCLUDE_DIR "${SLANG_ROOT}/include" CACHE STRING "" FORCE)
set(SLANG_RHI_SLANG_BINARY_DIR "${SLANG_ROOT}" CACHE STRING "" FORCE)
set(SLANG_RHI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_TESTS_WITH_GLFW OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_INSTALL OFF CACHE BOOL "" FORCE)
# No CPU backend: nothing in this engine may run its numbers on the CPU, and a
# backend that exists gets used as a fallback by someone. No WebGPU: Dawn is a
# large download for a target this engine does not have.
set(SLANG_RHI_ENABLE_CPU OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_WGPU OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
if(APPLE)
    # Metal only. Vulkan on macOS is MoltenVK over Metal, which is the device
    # we already have minus a translation layer.
    set(SLANG_RHI_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
endif()

# Patched: see cmake/patches/. Each patch is a bug found here, small, and
# worth sending upstream; the list should shrink when the pin moves.
#   - slang-rhi-metal-render-target-array-length.patch: every Metal render
#     pass with more than one colour target was invalid.
#   - slang-rhi-metal-texture-view-format.patch: a view of a whole texture in
#     another format (sRGB over linear) came back as the texture itself, in
#     the texture's format.
#   - slang-rhi-cuda-driver-symbols.patch: slang-rhi loads the CUDA driver by
#     dlopen and holds its entry points in variables carrying the driver's own
#     names. At global scope those are the definitions the rest of the program
#     binds to, so gpe and OIDN called through slang-rhi's pointers instead of
#     libcuda and crashed on a null one. A namespace keeps them to slang-rhi.
#   - slang-rhi-metal-acceleration-structures.patch: once any
#     acceleration structure had been freed, the next build threw inside
#     Metal (a nil in the device's structure array) and aborted the process;
#     and an indexed triangle build took max(vertices, indices) / 3 triangles,
#     reading past its index window.
FetchContent_Declare(slang_rhi
    GIT_REPOSITORY https://github.com/shader-slang/slang-rhi.git
    GIT_TAG        e17f6d75f858f9b7cb91bc102a7b8c6fda0435dc
    GIT_SHALLOW    FALSE
    GIT_SUBMODULES ""
    PATCH_COMMAND  ${CMAKE_COMMAND}
                   "-DPATCHES=${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-render-target-array-length.patch$<SEMICOLON>${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-acceleration-structures.patch$<SEMICOLON>${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-texture-view-format.patch$<SEMICOLON>${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-cuda-driver-symbols.patch"
                   -P "${CMAKE_CURRENT_LIST_DIR}/patches/apply.cmake"
    UPDATE_DISCONNECTED TRUE
    SYSTEM)
FetchContent_MakeAvailable(slang_rhi)

# --- small libraries ------------------------------------------------------------

FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.5.0
    GIT_SHALLOW    TRUE
    SYSTEM)
set(CLI11_PRECOMPILED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cli11)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    SYSTEM)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# tinyexr writes the renders. Its own miniz, compiled here, so EXR's zip
# compression needs nothing from the system.
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        v1.0.12
    GIT_SHALLOW    TRUE
    SYSTEM)
FetchContent_GetProperties(tinyexr)
if(NOT tinyexr_POPULATED)
    FetchContent_Populate(tinyexr)
endif()
add_library(lrt_tinyexr STATIC
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c)
target_include_directories(lrt_tinyexr SYSTEM PUBLIC
    ${tinyexr_SOURCE_DIR} ${tinyexr_SOURCE_DIR}/deps/miniz)
set_target_properties(lrt_tinyexr PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- GLFW and Dear ImGui: lrt view ---------------------------------------------
#
# The viewer's window and its panels. GLFW with no client API: slang-rhi makes
# the surface. Dear ImGui tessellates its panels on the CPU -- chrome, not
# scene data -- and draws through the engine's own slang-rhi backend
# (modules/view/src/ImGuiRenderer.cpp); only its GLFW input backend is used.

option(LRT_BUILD_VIEW "lrt view: a window onto a stage (GLFW, Dear ImGui)" ON)
if(LRT_BUILD_VIEW)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_MakeAvailable(glfw)

    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.92.9
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_GetProperties(imgui)
    if(NOT imgui_POPULATED)
        FetchContent_Populate(imgui)
    endif()
    add_library(lrt_imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)
    target_include_directories(lrt_imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    # 32-bit indices: the renderer pulls vertices through them from a buffer.
    target_compile_definitions(lrt_imgui PUBLIC "ImDrawIdx=unsigned int" IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
    target_link_libraries(lrt_imgui PUBLIC glfw)
    set_target_properties(lrt_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING)
    FetchContent_Declare(catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_MakeAvailable(catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()
