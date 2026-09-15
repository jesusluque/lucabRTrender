lrt_add_module(core
    SOURCES
        src/Error.cpp
        src/Log.cpp
        src/Platform.cpp)
find_package(Threads REQUIRED)
target_link_libraries(lrt_core PUBLIC Threads::Threads)
if(UNIX AND NOT APPLE)
    # A headless OpenGL context for libraries that draw with GL (Storm).
    find_library(LRT_EGL_LIB EGL)
    find_path(LRT_EGL_INCLUDE EGL/egl.h)
    if(LRT_EGL_LIB AND LRT_EGL_INCLUDE)
        target_link_libraries(lrt_core PUBLIC ${LRT_EGL_LIB})
        target_compile_definitions(lrt_core PRIVATE LRT_HAVE_EGL=1)
    endif()
endif()
if(APPLE)
    # Platform reaches a window's Metal layer through the Objective-C runtime.
    target_link_libraries(lrt_core PUBLIC objc "-framework CoreFoundation" "-framework CoreGraphics")
endif()
