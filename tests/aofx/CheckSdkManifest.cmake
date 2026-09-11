# The aofx SDK headers, hashed against the manifest recorded beside this file.
#
# aofx compatibility is not negotiable: bundles openFXplayer builds must load
# here unchanged, so its SDK is copied as it is and follows openFXplayer's ABI
# bumps, never this engine's needs. This test fails on any change to a header
# -- an edit, a new header, a removed one -- so a change is always a decision:
# re-record with -DRECORD=ON only when openFXplayer's SDK moved the same way.
#
#   cmake -DSDK_DIR=... -DMANIFEST=... [-DRECORD=ON] -P CheckSdkManifest.cmake
get_filename_component(SDK_DIR "${SDK_DIR}" ABSOLUTE)
file(GLOB headers RELATIVE "${SDK_DIR}" "${SDK_DIR}/*.h")
if(NOT headers)
    message(FATAL_ERROR "no headers in ${SDK_DIR}")
endif()
list(SORT headers)
set(lines "")
foreach(header IN LISTS headers)
    file(SHA256 "${SDK_DIR}/${header}" hash)
    string(APPEND lines "${hash}  ${header}\n")
endforeach()
if(RECORD)
    file(WRITE "${MANIFEST}" "${lines}")
    message(STATUS "recorded ${MANIFEST}")
    return()
endif()
file(READ "${MANIFEST}" recorded)
if(NOT recorded STREQUAL lines)
    message(FATAL_ERROR "the aofx SDK headers differ from ${MANIFEST}.\n"
                        "recorded:\n${recorded}\nnow:\n${lines}\n"
                        "aofx compatibility is mandatory: undo the change, or re-record only if "
                        "openFXplayer's SDK changed identically.")
endif()
message(STATUS "aofx SDK matches its manifest (${headers})")
