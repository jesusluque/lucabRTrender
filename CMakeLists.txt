cmake_minimum_required(VERSION 3.24)
project(lucabRTrender VERSION 0.1.0 LANGUAGES C CXX)
if(APPLE)
    # gpe's Metal backend is one .mm. A language enabled only inside a
    # subdirectory leaves CMake without its compile rule at generate time, so
    # it is enabled here, at the top, where every directory can see it.
    enable_language(OBJC OBJCXX)
endif()

# lucabRTrender: a Slang renderer for Gaussian splats and point clouds, and the
# engine openFXplayer will stand on. Linux and macOS first; Windows later, which
# is why nothing here names a POSIX call outside modules/core/Platform.
#
# The design, the decisions and why each was taken live in docs/decisions.md.
# The short version a build needs:
#
#   - slang-rhi is the render device (Metal on Apple, CUDA+OptiX or Vulkan on
#     Linux), with Slang compiled at run time and bound by reflection.
#   - gpe (submodule) is the compute/video engine openFXplayer already uses. It
#     does not make a second device: it adopts slang-rhi's, so a buffer written
#     by one is read by the other with nothing in between.
#   - aofx is openFXplayer's plugin SDK at its current ABI (22), unchanged in shape, so the
#     bundles openFXplayer builds load here.
#   - OpenUSD is where entities live; a Hydra 2.0 render delegate feeds the
#     engine's flat RenderScene.
#   - genlock (submodule) is time: PTP and SMPTE ST 2059-1 alignment.
#
# Nothing numeric runs on the CPU, reference renders and test oracles
# included: the CPU reads files and parses headers, and the GPU does the rest.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Warnings)
# Before Dependencies: it fetches Catch2 only when BUILD_TESTING is on, and in a
# fresh build directory nothing defines BUILD_TESTING until CTest does.
include(CTest)
include(Dependencies)
include(Shaders)
lrt_shader_copy_target()

# --- submodules -----------------------------------------------------------------
#
# Pinned, like openFXplayer pins them, and overridable for the same reason: a
# person working on gpe and this engine at once points LRT_GPE_DIR at their
# own checkout and edits in place.

set(LRT_GPE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gpe"
    CACHE PATH "Path to the gpe compute engine")
set(LRT_GENLOCK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/genlock"
    CACHE PATH "Path to the genlock timing library")

if(NOT EXISTS "${LRT_GPE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "gpe not found at ${LRT_GPE_DIR}: "
                        "git submodule update --init --recursive")
endif()
# gpe's own tests are built only when asked for: they are gpe's to run, and a
# consumer that builds them has two test suites fighting over one GPU.
set(LRT_BUILD_GPE_TESTS OFF CACHE BOOL "Build and register gpe's own tests")
# gpe's QRhi viewer is a windowed presenter for gpe's own player. This engine
# is headless and has no use for Qt.
set(CMAKE_DISABLE_FIND_PACKAGE_Qt6 ON)
# slang-rhi already defines metal-cpp's selector tables; one definition per
# program.
set(GPE_METAL_CPP_IMPLEMENTATION OFF CACHE BOOL "" FORCE)
set(_lrt_testing ${BUILD_TESTING})
set(BUILD_TESTING ${LRT_BUILD_GPE_TESTS})
add_subdirectory(${LRT_GPE_DIR} ${CMAKE_BINARY_DIR}/gpe)
set(BUILD_TESTING ${_lrt_testing})
unset(CMAKE_DISABLE_FIND_PACKAGE_Qt6)
message(STATUS "gpe: ${LRT_GPE_DIR} (backend ${GPE_BACKEND})")

if(NOT EXISTS "${LRT_GENLOCK_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "genlock not found at ${LRT_GENLOCK_DIR}: "
                        "git submodule update --init --recursive")
endif()
set(GENLOCK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GENLOCK_BUILD_TOOLS ON CACHE BOOL "" FORCE)
add_subdirectory(${LRT_GENLOCK_DIR} ${CMAKE_BINARY_DIR}/genlock EXCLUDE_FROM_ALL)
if(TARGET genlock-cli)
    # Out of EXCLUDE_FROM_ALL and into bin/: `genlock-cli master` is what a
    # render node's --ptp points at, and a machine with one computer needs to
    # be able to serve the time it follows.
    set_target_properties(genlock-cli PROPERTIES EXCLUDE_FROM_ALL FALSE)
endif()

# --- the engine -----------------------------------------------------------------

add_subdirectory(third_party/spz)
# OpenUSD and MaterialX, seen by the modules that read them, the apps and tests.
include(Usd)
add_subdirectory(modules)
add_subdirectory(plugins)
add_subdirectory(apps)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
