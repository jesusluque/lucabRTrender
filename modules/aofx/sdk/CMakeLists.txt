# AOFX -- the effect SDK. Headers only: a plugin that linked the host would
# share its allocator, logging and dependencies, and every one of those is a
# way for a bundle to take the application down with it.
add_library(aofx INTERFACE)
add_library(aofx::aofx ALIAS aofx)
target_include_directories(aofx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(aofx INTERFACE cxx_std_20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AofxKernel)
include(AofxBundle)
