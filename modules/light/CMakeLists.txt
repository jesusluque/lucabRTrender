# Lights: what a scene emits, and how a shading point samples it.
lrt_add_module(light
    SOURCES
        src/LightTable.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::gpu
        lrt::render
        lrt::io)
