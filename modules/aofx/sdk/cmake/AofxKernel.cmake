# Compiling a plugin's kernels and embedding them in its bundle.
#
#   aofx_add_kernel(<target> <name> ENTRY <entryPoint> [ENTRY <entryPoint>...]
#                   [SOURCE <file.slang>])
#
# openFXplayer's rule, unchanged in use: <name>.slang from the current source
# directory, compiled for this build's gpe backend (metallib on Metal, PTX on
# CUDA), embedded as `k_<name>` / `k_<name>Bytes` in aofx_kernels_<name>.h.
#
# ONE ADDITION: the trailer. slangc is also asked for its reflection, and the
# embedded bytes end with gpe's kernel trailer (gpe/cmake/KernelTrailer.cmake):
# each entry's [numthreads], each buffer's element size, the uniform block's
# size. gpe strips it at registration and then launches the kernel's own group
# shape, bounds-checks in the kernel's own element size, and refuses a dispatch
# with the wrong number of buffers or the wrong uniform size. Nothing about
# KernelDesc or the ABI changes -- a trailer is more bytes in the same blob.

function(aofx_add_kernel target name)
    cmake_parse_arguments(ARG "" "SOURCE" "ENTRY" ${ARGN})
    if(NOT ARG_ENTRY)
        message(FATAL_ERROR "aofx_add_kernel(${name}) needs ENTRY")
    endif()
    if(NOT GPE_SLANGC OR GPE_BACKEND STREQUAL "none")
        message(FATAL_ERROR "aofx_add_kernel(${name}): no gpe backend to compile for")
    endif()
    set(entryFlags "")
    foreach(entry IN LISTS ARG_ENTRY)
        list(APPEND entryFlags -entry ${entry} -stage compute)
    endforeach()

    set(source "${CMAKE_CURRENT_SOURCE_DIR}/${name}.slang")
    if(ARG_SOURCE)
        get_filename_component(source "${ARG_SOURCE}" ABSOLUTE)
    endif()
    set(dir "${CMAKE_CURRENT_BINARY_DIR}/kernels")
    file(MAKE_DIRECTORY "${dir}")
    set(blob "${dir}/${name}.blob")
    set(json "${dir}/${name}.reflection.json")
    set(header "${dir}/aofx_kernels_${name}.h")

    if(GPE_BACKEND STREQUAL "Metal")
        set(msl "${dir}/${name}.metal")
        set(air "${dir}/${name}.air")
        add_custom_command(
            OUTPUT "${blob}" "${json}"
            COMMAND ${GPE_SLANGC} "${source}" -target metal ${entryFlags}
                    -o "${msl}" -reflection-json "${json}"
            COMMAND ${GPE_METAL} -c "${msl}" -o "${air}"
            COMMAND ${GPE_METALLIB} "${air}" -o "${blob}"
            DEPENDS "${source}"
            COMMENT "slang -> metallib: ${name}"
            VERBATIM)
    elseif(GPE_BACKEND STREQUAL "CUDA")
        set(cu "${dir}/${name}.cu")
        add_custom_command(
            OUTPUT "${blob}" "${json}"
            COMMAND ${GPE_SLANGC} "${source}" -target cuda ${entryFlags}
                    -o "${cu}" -reflection-json "${json}"
            COMMAND ${GPE_NVCC} -ptx "${cu}" -o "${blob}" -arch=${GPE_CUDA_ARCH}
            DEPENDS "${source}"
            COMMENT "slang -> ptx: ${name}"
            VERBATIM)
    endif()

    set(script "${dir}/write_${name}_header.cmake")
    file(WRITE "${script}" "\
include(\"${GPE_KERNEL_TRAILER_SCRIPT}\")
file(READ \"${blob}\" hex HEX)
gpe_kernel_trailer_hex(\"${json}\" \"${ARG_ENTRY}\" trailer)
string(APPEND hex \"\${trailer}\")
string(REGEX REPLACE \"(..)\" \"0x\\\\1,\" bytes \"\${hex}\")
string(REGEX REPLACE \"((0x..,){12})\" \"\\\\1\\n    \" bytes \"\${bytes}\")
file(WRITE \"${header}\" \"\
// Generated at build time from ${name}.slang. Do not edit.
#pragma once

#include <cstddef>

inline constexpr unsigned char k_${name}[] = {
    \${bytes}
};
inline constexpr size_t k_${name}Bytes = sizeof(k_${name});
\")
")
    add_custom_command(
        OUTPUT "${header}"
        COMMAND ${CMAKE_COMMAND} -P "${script}"
        DEPENDS "${blob}" "${json}" "${script}" "${GPE_KERNEL_TRAILER_SCRIPT}"
        COMMENT "embedding kernel: ${name}"
        VERBATIM)
    target_sources(${target} PRIVATE "${header}")
    target_include_directories(${target} PRIVATE "${dir}")
endfunction()
