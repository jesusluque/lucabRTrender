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
