find_package(TBB CONFIG REQUIRED HINTS "${LRT_USD_ROOT}")
lrt_add_module(io
    SOURCES
        src/PlyHeader.cpp
        src/SplatReaders.cpp
        src/PointReaders.cpp
        src/Exr.cpp
        src/Sog.cpp
        src/Ies.cpp
    PUBLIC_LIBS
        lrt::core
    PRIVATE_LIBS
        TBB::tbb
        lrt_tinyexr
        nlohmann_json::nlohmann_json)
if(LRT_HAVE_SPZ)
    target_link_libraries(lrt_io PRIVATE lrt_spz)
    target_compile_definitions(lrt_io PRIVATE LRT_HAVE_SPZ=1)
endif()

# .vdb: OpenVDB reads it and NanoVDB lays it out (cmake/Usd.cmake finds both).
target_sources(lrt_io PRIVATE src/Vdb.cpp)
if(LRT_HAVE_OPENVDB)
    target_link_libraries(lrt_io PRIVATE ${LRT_OPENVDB_LIB})
    target_include_directories(lrt_io SYSTEM PRIVATE "${LRT_USD_ROOT}/include")
    target_compile_definitions(lrt_io PRIVATE LRT_HAVE_OPENVDB=1)
    # OpenVDB's headers predate the warnings this engine asks for.
    set_source_files_properties(src/Vdb.cpp PROPERTIES COMPILE_OPTIONS
        "-Wno-old-style-cast;-Wno-shadow;-Wno-deprecated-declarations;-Wno-double-promotion;-Wno-cast-align;-Wno-sign-conversion;-Wno-conversion;-Wno-unused-parameter;-Wno-implicit-int-conversion;-Wno-implicit-float-conversion;-Wno-float-conversion;-Wno-shorten-64-to-32")
endif()

# SOG: zlib for the bundle, libwebp for its images. Without libwebp .sog is
# refused with a message.
find_package(ZLIB REQUIRED)
if(APPLE)
    target_link_libraries(lrt_io PRIVATE z)   # see third_party/spz: not ZLIB::ZLIB on Apple
else()
    target_link_libraries(lrt_io PRIVATE ZLIB::ZLIB)
endif()
find_path(LRT_WEBP_INCLUDE webp/decode.h HINTS /opt/homebrew/opt/webp/include /usr/local/opt/webp/include)
find_library(LRT_WEBP_LIB webp HINTS /opt/homebrew/opt/webp/lib /usr/local/opt/webp/lib)
if(LRT_WEBP_INCLUDE AND LRT_WEBP_LIB)
    target_include_directories(lrt_io SYSTEM PRIVATE ${LRT_WEBP_INCLUDE})
    target_link_libraries(lrt_io PRIVATE ${LRT_WEBP_LIB})
    target_compile_definitions(lrt_io PRIVATE LRT_HAVE_WEBP=1)
    set(LRT_HAVE_WEBP TRUE CACHE INTERNAL "")
    message(STATUS "io: reading .sog (libwebp ${LRT_WEBP_LIB})")
else()
    set(LRT_HAVE_WEBP FALSE CACHE INTERNAL "")
    message(STATUS "io: .sog disabled -- libwebp not found")
endif()
