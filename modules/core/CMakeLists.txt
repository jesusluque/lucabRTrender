lrt_add_module(core
    SOURCES
        src/Error.cpp
        src/Log.cpp
        src/Platform.cpp)
find_package(Threads REQUIRED)
target_link_libraries(lrt_core PUBLIC Threads::Threads)
