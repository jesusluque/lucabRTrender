# Geometry on the device: meshes (triangulation, normals, primvars), and later
# subdivision, curves, skinning and instancing.
lrt_add_module(geom
    SOURCES
        src/Mesh.cpp
    PUBLIC_LIBS
        lrt::core
        lrt::gpu
        lrt::scene)
