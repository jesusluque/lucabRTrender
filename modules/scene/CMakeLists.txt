lrt_add_module(scene
    SOURCES
        src/GpuClouds.cpp
        src/SplatSkinner.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::gpu
        lrt::io)
