# OpenUSD: where entities live. Built once per machine by scripts/build-usd.sh.
# pxrConfig finds its own dependencies (OpenSubdiv, TBB) through the prefix path.
list(APPEND CMAKE_PREFIX_PATH "${LRT_USD_ROOT}")
# Imaging targets name OpenGL::GL in their link interface (garch) even though
# nothing here draws with GL.
find_package(OpenGL REQUIRED)
find_package(pxr CONFIG REQUIRED HINTS "${LRT_USD_ROOT}")
message(STATUS "OpenUSD: ${pxr_DIR}")
# MaterialX, which this OpenUSD is built with (scripts/build-usd.sh): the
# engine's materials are generated from MaterialX documents by its Slang
# generator.
if(NOT TARGET MaterialXCore)
    find_package(MaterialX 1.39.5 CONFIG REQUIRED HINTS "${LRT_USD_ROOT}")
endif()
if(NOT TARGET MaterialXGenSlang)
    message(FATAL_ERROR "OpenUSD at ${LRT_USD_ROOT} carries MaterialX without its Slang generator; "
                        "rebuild with scripts/build-usd.sh")
endif()
# Imported targets are directory-scoped; the engine's other modules and apps
# link MaterialX too.
foreach(_mx MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenHw MaterialXGenSlang hgi)
    if(TARGET ${_mx})
        get_target_property(_global ${_mx} IMPORTED_GLOBAL)
        if(NOT _global)
            set_target_properties(${_mx} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()
endforeach()

# OpenVDB and NanoVDB's headers come with the same prefix: what a volume is
# read from and laid out as. PNanoVDB.h travels with the shaders, since
# lrt/volume/nanovdb.slang includes it.
find_library(LRT_OPENVDB_LIB openvdb HINTS "${LRT_USD_ROOT}/lib" NO_DEFAULT_PATH)
if(LRT_OPENVDB_LIB AND EXISTS "${LRT_USD_ROOT}/include/nanovdb/PNanoVDB.h")
    set(LRT_HAVE_OPENVDB ON)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/shaders/nanovdb/PNanoVDB.h"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LRT_USD_ROOT}/include/nanovdb/PNanoVDB.h"
                "${CMAKE_BINARY_DIR}/shaders/nanovdb/PNanoVDB.h"
        DEPENDS "${LRT_USD_ROOT}/include/nanovdb/PNanoVDB.h" VERBATIM)
    add_custom_target(lrt_nanovdb_header ALL DEPENDS "${CMAKE_BINARY_DIR}/shaders/nanovdb/PNanoVDB.h")
    if(TARGET lrt_shaders_copy)
        add_dependencies(lrt_shaders_copy lrt_nanovdb_header)
    endif()
    message(STATUS "OpenVDB: ${LRT_OPENVDB_LIB} (volumes read and laid out as NanoVDB)")
else()
    set(LRT_HAVE_OPENVDB OFF)
    message(STATUS "OpenVDB: not found in ${LRT_USD_ROOT}; .vdb is refused with a message")
endif()
