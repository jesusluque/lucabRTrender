# The scene on the device: pools of mesh data and per-instance records, read
# by every technique.
lrt_add_module(world
    SOURCES
        src/GpuScene.cpp
        src/Instancing.cpp
        src/RayTracingScene.cpp
        src/BvhScene.cpp
        src/VolumeSet.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::gpu
        lrt::io
        lrt::geom
        lrt::render)
