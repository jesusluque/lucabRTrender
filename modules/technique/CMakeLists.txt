# Techniques: how a frame is drawn from the scene -- visibility, shading, path
# tracing, denoising, display.
lrt_add_module(technique
    SOURCES
        src/Denoiser.cpp
        src/DisplayTransform.cpp
        src/Ocio.cpp
        src/Aces2.cpp
        src/MaterialPrograms.cpp
        src/MaterialShading.cpp
        src/EmissiveTable.cpp
        src/PathTracer.cpp
        src/Visibility.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::gpu
        lrt::world
        lrt::material
        lrt::light
        lrt::render)

# OpenColorIO (scripts/build-ocio.sh), as a compiler of shader text and LUTs
# for the display. Without it ViewTransform::Ocio says so.
set(LRT_OCIO_ROOT "$ENV{HOME}/tools/ocio-2.5.2" CACHE PATH "OpenColorIO install")
find_package(OpenColorIO 2.5 CONFIG QUIET HINTS "${LRT_OCIO_ROOT}" NO_DEFAULT_PATH)
if(OpenColorIO_FOUND)
    target_link_libraries(lrt_technique PRIVATE OpenColorIO::OpenColorIO)
    target_compile_definitions(lrt_technique PRIVATE LRT_HAVE_OCIO=1)
    message(STATUS "technique: OpenColorIO ${OpenColorIO_VERSION} (${LRT_OCIO_ROOT})")
else()
    target_compile_definitions(lrt_technique PRIVATE LRT_HAVE_OCIO=0)
    message(STATUS "technique: no OpenColorIO at ${LRT_OCIO_ROOT}; no OCIO view")
endif()

# OIDN, GPU devices only (scripts/build-oidn.sh). Without it the engine has no
# denoiser, and says so.
set(LRT_OIDN_ROOT "$ENV{HOME}/tools/oidn-2.5.1" CACHE PATH "Open Image Denoise install (GPU devices only)")
find_package(OpenImageDenoise 2.5 CONFIG QUIET HINTS "${LRT_OIDN_ROOT}" NO_DEFAULT_PATH)
if(OpenImageDenoise_FOUND)
    target_link_libraries(lrt_technique PRIVATE OpenImageDenoise)
    target_compile_definitions(lrt_technique PRIVATE LRT_HAVE_OIDN=1)
    message(STATUS "technique: OIDN ${OpenImageDenoise_VERSION} (${LRT_OIDN_ROOT})")
else()
    target_compile_definitions(lrt_technique PRIVATE LRT_HAVE_OIDN=0)
    message(STATUS "technique: no OIDN at ${LRT_OIDN_ROOT}; no denoiser")
endif()
