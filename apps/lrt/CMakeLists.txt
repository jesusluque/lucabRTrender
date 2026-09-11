add_executable(lrt
    src/main.cpp
    src/CmdInfo.cpp
    src/CmdRender.cpp
    src/CmdStage.cpp
    src/CmdAofx.cpp)
target_link_libraries(lrt PRIVATE lrt::usd lrt::aofx_host CLI11::CLI11)
# Bundles built in this tree are found without a --path.
target_compile_definitions(lrt PRIVATE LRT_AOFX_BUNDLE_DIR="${AOFX_BUNDLE_DIR}")
add_dependencies(lrt lrt_aofx_invert)
target_include_directories(lrt PRIVATE src)
lrt_set_target_defaults(lrt)
set_target_properties(lrt PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
lrt_copy_shaders(lrt)

# slang-rhi loads libslang-compiler at run time from beside the binary on some
# platforms and through the rpath on others; both are pointed at SLANG_ROOT.
set_target_properties(lrt PROPERTIES
    BUILD_RPATH "${SLANG_ROOT}/lib;${LRT_USD_ROOT}/lib"
    INSTALL_RPATH "@loader_path/../lib;${SLANG_ROOT}/lib")
