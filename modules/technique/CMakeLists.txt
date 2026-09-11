# Techniques: how a frame is drawn from the scene -- visibility, shading, path
# tracing, denoising, display.
lrt_add_module(technique
    SOURCES
        src/Denoiser.cpp
        src/DisplayTransform.cpp
        src/MaterialShading.cpp
        src/Visibility.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::gpu
        lrt::world
        lrt::material
        lrt::render)

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
