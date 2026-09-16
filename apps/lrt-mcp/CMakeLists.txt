add_executable(lrt-mcp main.cpp)
target_link_libraries(lrt-mcp PRIVATE lrt::mcp)
lrt_set_target_defaults(lrt-mcp)
set_target_properties(lrt-mcp PROPERTIES BUILD_RPATH "${SLANG_ROOT}/lib;${LRT_USD_ROOT}/lib")
