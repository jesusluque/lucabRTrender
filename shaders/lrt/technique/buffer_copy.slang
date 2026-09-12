// Copyright (c) 2026 lucabRTrender contributors.
//
// A copy between buffers done by a kernel, word by word. The denoiser's
// staging needs it: slang-rhi's Metal copyBuffer does nothing, silently, when
// either side is a buffer wrapped from a native handle (measured: 4095 of 4096
// words untouched in both directions, while a kernel reads and writes the same
// buffer exactly). Any type goes through as its words.
module buffer_copy;

struct CopyParams {
    uint words;
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>     src;
RWStructuredBuffer<uint>   dst;
ConstantBuffer<CopyParams> copy;

[shader("compute")]
[numthreads(256, 1, 1)]
void copyWords(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < copy.words) {
        dst[tid.x] = src[tid.x];
    }
}
