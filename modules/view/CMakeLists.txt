# lrt view: a window onto a stage. GLFW for the window, Dear ImGui for panels
# drawn by the engine's own device; the frames never leave the GPU.
lrt_add_module(view
    SOURCES
        src/Window.cpp
        src/ImGuiRenderer.cpp
        src/Viewer.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::gpu
        lrt::technique
        lrt::usd
    PRIVATE_LIBS
        lrt_imgui
        glfw)
# USD's headers, through StageRenderer's, predate the warnings this engine asks for.
target_compile_options(lrt_view PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-deprecated-declarations
                       -Wno-double-promotion -Wno-cast-align)
set_target_properties(lrt_view PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
