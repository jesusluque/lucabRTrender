# AOFX plugins built with this engine's SDK. Each lands in build/<preset>/aofx.
#
# One folder per plugin with its own CMakeLists, as aopenfx's examples are
# laid out: adding a plugin is adding a directory, and nothing here has to
# learn its name.
file(GLOB plugin_dirs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS "*")
foreach(plugin IN LISTS plugin_dirs)
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${plugin}"
       AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt")
        add_subdirectory(${plugin})
    endif()
endforeach()
