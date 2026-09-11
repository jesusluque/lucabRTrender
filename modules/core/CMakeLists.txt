lrt_add_module(core
    SOURCES
        src/Error.cpp
        src/Log.cpp
        src/Platform.cpp)
find_package(Threads REQUIRED)
target_link_libraries(lrt_core PUBLIC Threads::Threads)
if(APPLE)
    # Platform reaches a window's Metal layer through the Objective-C runtime.
    target_link_libraries(lrt_core PUBLIC objc)
endif()
