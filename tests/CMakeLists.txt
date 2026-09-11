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
    SOURCES gpu/test_algo.cpp
    LIBS lrt::gpu)

lrt_test(lrt_scene_tests
    SOURCES scene/test_loading.cpp
    LIBS lrt::scene)

lrt_test(lrt_render_tests
    SOURCES render/test_splat_render.cpp render/test_points.cpp render/test_ray_tracing.cpp
    LIBS lrt::render lrt::io)

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
