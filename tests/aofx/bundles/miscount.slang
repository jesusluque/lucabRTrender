// A kernel that declares two buffers, for a plugin that runs it with one.
struct Params { uint width; uint height; };
StructuredBuffer<float4>   src;
RWStructuredBuffer<float4> dst;
ConstantBuffer<Params>     params;

[shader("compute")]
[numthreads(16, 16, 1)]
void miscountMain(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    dst[tid.y * params.width + tid.x] = src[tid.y * params.width + tid.x];
}
