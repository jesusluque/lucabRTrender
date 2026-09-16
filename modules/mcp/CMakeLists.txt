# MCP over the renderer: the protocol and the tools. Links the engine's USD
# side, since a tool's subject is a stage.
lrt_add_module(mcp
    SOURCES
        src/Server.cpp
        src/Tools.cpp
    PUBLIC_LIBS lrt::usd)
target_link_libraries(lrt_mcp PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(lrt_mcp PRIVATE -Wno-old-style-cast -Wno-shadow -Wno-cast-align
                       -Wno-deprecated-declarations -Wno-double-promotion)
set_target_properties(lrt_mcp PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
