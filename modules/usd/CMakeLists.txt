set(LRT_USD_SOURCES
    src/HydraCamera.cpp
    src/PrimData.cpp
    src/Engine.cpp
    src/RenderBuffer.cpp
    src/ParticleField.cpp
    src/Mesh.cpp
    src/Instancer.cpp
    src/Material.cpp
    src/Points.cpp
    src/RenderPass.cpp
    src/RenderDelegate.cpp)
set(LRT_USD_LIBS usd usdGeom usdVol usdShade usdLux usdRender usdSkel usdImaging usdSkelImaging
    hd hdx hdsi hdMtlx hio cameraUtil gf tf vt sdf plug lrt::gpu)

# The engine side of Hydra, linked into the CLI and tests.
lrt_add_module(usd
    SOURCES ${LRT_USD_SOURCES} src/StageRenderer.cpp src/Export.cpp
    PUBLIC_LIBS lrt::render lrt::io lrt::scene lrt::lod lrt::geom lrt::world lrt::technique ${LRT_USD_LIBS})
# USD's headers predate the warnings this engine asks for.
target_compile_options(lrt_usd PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-cast-align
                       -Wno-deprecated-declarations -Wno-double-promotion)
target_compile_definitions(lrt_usd PUBLIC PXR_STATIC=0 TBB_SUPPRESS_DEPRECATED_MESSAGES=1)
# Default visibility, unlike every other engine module. USD identifies value
# types by std::type_info (TfType), and a VtArray<GfVec3f> instantiated here
# with hidden visibility has its own type_info the USD libraries do not
# recognise: saving a stage failed with "Attempted to pack unsupported type
# 'GfVec3f'" on an array of GfVec3f.
set_target_properties(lrt_usd PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)

# The same, as a Hydra renderer plugin any USD application can load
# (usdview, Houdini, Nuke): PXR_PLUGINPATH_NAME=<build>/plugin/usd.
set(LRT_HYDRA_PLUGIN_DIR "${CMAKE_BINARY_DIR}/plugin/usd" CACHE INTERNAL "Where hdLrt and its plugInfo land")
# RendererPlugin.cpp only here: it registers a TfType, and a process holding
# both the static library and the loaded plugin must not register it twice.
add_library(hdLrt MODULE ${LRT_USD_SOURCES} src/RendererPlugin.cpp)
target_include_directories(hdLrt PRIVATE include src)
target_link_libraries(hdLrt PRIVATE lrt::render lrt::io lrt::scene lrt::lod lrt::geom lrt::world lrt::technique
                      ${LRT_USD_LIBS})
target_compile_options(hdLrt PRIVATE -Wno-deprecated-declarations)
set_target_properties(hdLrt PROPERTIES
    PREFIX "" OUTPUT_NAME "hdLrt"
    CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
    LIBRARY_OUTPUT_DIRECTORY "${LRT_HYDRA_PLUGIN_DIR}/hdLrt"
    BUILD_RPATH "${SLANG_ROOT}/lib;${LRT_USD_ROOT}/lib")
file(MAKE_DIRECTORY "${LRT_HYDRA_PLUGIN_DIR}/hdLrt/resources")
configure_file(plugin/plugInfo.json.in "${LRT_HYDRA_PLUGIN_DIR}/hdLrt/resources/plugInfo.json" @ONLY)
file(WRITE "${LRT_HYDRA_PLUGIN_DIR}/plugInfo.json" "{ \"Includes\": [ \"*/resources/\" ] }\n")

# The codeless schemas (LrtSplatEditAPI, LrtPointStyleAPI), beside hdLrt so
# the same PXR_PLUGINPATH_NAME finds both.
file(MAKE_DIRECTORY "${LRT_HYDRA_PLUGIN_DIR}/lrtSchemas/resources")
configure_file(schemas/plugInfo.json "${LRT_HYDRA_PLUGIN_DIR}/lrtSchemas/resources/plugInfo.json" COPYONLY)
configure_file(schemas/generatedSchema.usda "${LRT_HYDRA_PLUGIN_DIR}/lrtSchemas/resources/generatedSchema.usda" COPYONLY)
