find_package(TBB CONFIG REQUIRED HINTS "${LRT_USD_ROOT}")
lrt_add_module(io
    SOURCES
        src/PlyHeader.cpp
        src/SplatReaders.cpp
        src/PointReaders.cpp
        src/Exr.cpp
    PUBLIC_LIBS
        lrt::core
    PRIVATE_LIBS
        TBB::tbb
        lrt_tinyexr)
if(LRT_HAVE_SPZ)
    target_link_libraries(lrt_io PRIVATE lrt_spz)
    target_compile_definitions(lrt_io PRIVATE LRT_HAVE_SPZ=1)
endif()
