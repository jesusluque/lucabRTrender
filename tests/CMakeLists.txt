include(Catch)

# lrt_test(<name> SOURCES ... LIBS ...)
#
# A Catch2 executable, registered with ctest case by case. GPU tests skip when
# the machine has no device; they never fall back to anything else.
function(lrt_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain ${ARG_LIBS})
    lrt_set_target_defaults(${name})
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${SLANG_ROOT}/lib")
    add_dependencies(${name} lrt_shaders_copy)
    catch_discover_tests(${name}
        PROPERTIES ENVIRONMENT "LRT_SHADER_DIR=${LRT_SHADER_OUTPUT_DIR}"
        DISCOVERY_MODE PRE_TEST)
endfunction()

lrt_test(lrt_gpu_tests
    SOURCES gpu/test_algo.cpp gpu/test_textures.cpp
    LIBS lrt::gpu lrt::render)

lrt_test(lrt_scene_tests
    SOURCES scene/test_loading.cpp
    LIBS lrt::scene)
if(LRT_HAVE_SPZ)
    # The SPZ test writes its fixtures with zlib.
    target_compile_definitions(lrt_scene_tests PRIVATE LRT_TEST_SPZ=1)
    target_link_libraries(lrt_scene_tests PRIVATE $<IF:$<BOOL:${APPLE}>,z,ZLIB::ZLIB>)
endif()

lrt_test(lrt_render_tests
    SOURCES render/test_splat_render.cpp render/test_points.cpp render/test_ray_tracing.cpp
            $<$<BOOL:${LRT_HAVE_SPZ}>:render/test_spz.cpp>
            $<$<BOOL:${LRT_HAVE_WEBP}>:render/test_sog.cpp>
    LIBS lrt::render lrt::io)
target_compile_definitions(lrt_render_tests PRIVATE LRT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
if(LRT_HAVE_SPZ)
    # Niantic's packer writes the fixture the SPZ reader is checked against.
    target_link_libraries(lrt_render_tests PRIVATE lrt_spz)
endif()

lrt_test(lrt_geom_tests
    SOURCES geom/test_mesh.cpp
    LIBS lrt::geom)

lrt_test(lrt_technique_tests
    SOURCES technique/test_visibility.cpp
    LIBS lrt::technique)

lrt_test(lrt_lod_tests
    SOURCES lod/test_lod.cpp
    LIBS lrt::lod)

lrt_test(lrt_usd_tests
    SOURCES usd/test_usd.cpp
    LIBS lrt::usd)
add_dependencies(lrt_usd_tests hdLrt)
target_compile_definitions(lrt_usd_tests PRIVATE LRT_HYDRA_PLUGIN_DIR="${LRT_HYDRA_PLUGIN_DIR}")
target_compile_options(lrt_usd_tests PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
set_target_properties(lrt_usd_tests PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF
                      BUILD_RPATH "${SLANG_ROOT}/lib;${LRT_USD_ROOT}/lib")

lrt_test(lrt_gpu_host_tests
    SOURCES gpu_host/test_shared_device.cpp
    LIBS lrt::gpu_host)

add_subdirectory(aofx/bundles)
lrt_test(lrt_aofx_tests
    SOURCES aofx/test_aofx.cpp
    LIBS lrt::aofx_host)
add_dependencies(lrt_aofx_tests aofx_test_bundles)
target_compile_definitions(lrt_aofx_tests PRIVATE
    LRT_AOFX_TEST_BUNDLES="${LRT_AOFX_TEST_BUNDLES}"
    LRT_AOFX_TEST_RUN_BUNDLES="${LRT_AOFX_TEST_RUN_BUNDLES}"
    LRT_AOFX_BUNDLES="${AOFX_BUNDLE_DIR}")
# The SDK headers against their recorded hashes: aofx compatibility is mandatory.
add_test(NAME aofx_sdk_manifest
    COMMAND ${CMAKE_COMMAND} -DSDK_DIR=${CMAKE_SOURCE_DIR}/modules/aofx/sdk/include/aofx
            -DMANIFEST=${CMAKE_CURRENT_SOURCE_DIR}/aofx/sdk_manifest.txt
            -P ${CMAKE_CURRENT_SOURCE_DIR}/aofx/CheckSdkManifest.cmake)

lrt_test(lrt_sched_tests
    SOURCES sched/test_clock.cpp
    LIBS lrt::sched lrt::io)

# One TBB in the process: OpenUSD's oneTBB, which OIDN (built without its CPU
# device) and everything else share.
add_test(NAME single_tbb COMMAND lrt info)
set_tests_properties(single_tbb PROPERTIES
    ENVIRONMENT "LRT_SHADER_DIR=${LRT_SHADER_OUTPUT_DIR}"
    PASS_REGULAR_EXPRESSION "tbb libraries  1 ")
