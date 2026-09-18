# Electronic Arts' mesh2splat as an effect: a mesh in, gaussian splats out.
# Two entries out of one blob -- the counters are cleared by one thread and
# then a thread walks each triangle -- which is why KERNELS names both.
aofx_add_bundle(lrt_aofx_mesh2splat
    NAME Mesh2Splat
    SOURCES Mesh2Splat.cpp
    KERNELS mesh2splat ENTRY m2sClear ENTRY m2sEmit SOURCE mesh2splat.slang)
