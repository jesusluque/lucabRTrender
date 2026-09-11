lrt_add_module(gpu
    SOURCES
        src/Device.cpp
        src/Buffer.cpp
        src/ShaderLibrary.cpp
        src/ComputeKernel.cpp
        src/CommandBatch.cpp
        src/algo/PrefixSum.cpp
        src/algo/RadixSort.cpp
    PUBLIC_LIBS
        lrt::core
        slang-rhi)
target_compile_definitions(lrt_gpu PRIVATE
    LRT_BUILD_SHADER_DIR="${LRT_SHADER_OUTPUT_DIR}")
