lrt_add_module(aofx_host
    SOURCES
        src/EffectRegistry.cpp
        src/EffectRunner.cpp
        src/EffectRender.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::image
        lrt::gpu_host
        aofx::aofx
        gpe)
target_link_libraries(lrt_aofx_host PRIVATE ${CMAKE_DL_LIBS})
# The host's own kernel, compiled by the SDK's own rule: if that rule is not
# good enough for the host, it is not good enough for a plugin.
aofx_add_kernel(lrt_aofx_host channels ENTRY channelsMain SOURCE kernels/channels.slang)
