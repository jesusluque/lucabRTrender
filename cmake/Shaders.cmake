# Slang sources compiled at run time by slang-rhi.
#
# The engine's own shaders are not embedded: they are compiled by the device
# the process opened, for the backend it opened, and a blob embedded at build
# time would pin one target. They are copied beside the binaries instead, and
# found through LRT_SHADER_DIR (set at build) or <exe>/../shaders.

set(LRT_SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(LRT_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

# One target that copies every shader, so tests and apps depend on the same one.
function(lrt_shader_copy_target)
    if(TARGET lrt_shaders_copy)
        return()
    endif()
    file(GLOB_RECURSE _shaders CONFIGURE_DEPENDS
        "${LRT_SHADER_SOURCE_DIR}/*.slang" "${LRT_SHADER_SOURCE_DIR}/*.slangh")
    set(_outputs)
    foreach(_src IN LISTS _shaders)
        file(RELATIVE_PATH _rel "${LRT_SHADER_SOURCE_DIR}" "${_src}")
        set(_dst "${LRT_SHADER_OUTPUT_DIR}/${_rel}")
        add_custom_command(OUTPUT "${_dst}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}" "${_dst}"
            DEPENDS "${_src}" VERBATIM)
        list(APPEND _outputs "${_dst}")
    endforeach()
    add_custom_target(lrt_shaders_copy ALL DEPENDS ${_outputs})
endfunction()

function(lrt_copy_shaders target)
    lrt_shader_copy_target()
    add_dependencies(${target} lrt_shaders_copy)
endfunction()

function(_lrt_copy_shaders_unused target)
    file(GLOB_RECURSE _shaders CONFIGURE_DEPENDS
        "${LRT_SHADER_SOURCE_DIR}/*.slang" "${LRT_SHADER_SOURCE_DIR}/*.slangh")
    set(_outputs)
    foreach(_src IN LISTS _shaders)
        file(RELATIVE_PATH _rel "${LRT_SHADER_SOURCE_DIR}" "${_src}")
        set(_dst "${LRT_SHADER_OUTPUT_DIR}/${_rel}")
        add_custom_command(OUTPUT "${_dst}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}" "${_dst}"
            DEPENDS "${_src}" VERBATIM)
        list(APPEND _outputs "${_dst}")
    endforeach()
    if(_outputs)
        add_custom_target(${target}_shaders DEPENDS ${_outputs})
        add_dependencies(${target} ${target}_shaders)
    endif()
endfunction()
