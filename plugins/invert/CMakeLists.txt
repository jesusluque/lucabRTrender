# Invert is openFXplayer's, copied unchanged: the proof that a plugin written
# against openFXplayer's SDK builds and runs here, now with a kernel trailer.
aofx_add_bundle(lrt_aofx_invert
    NAME Invert
    SOURCES Invert.cpp
    KERNELS invert ENTRY invertMain SOURCE invert.slang)
