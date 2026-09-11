// Copyright (c) 2026 lucabRTrender contributors.
//
// A table of textures and samplers bound by slot in one parameter block (a
// Metal argument buffer), read back by slot: what materials sample through.
module texture_table;

struct TextureTable {
    Texture2D<float4> textures[1024];
    SamplerState      samplers[16];
};

struct ProbeParams {
    uint count;
    uint stride;
    uint pad0; uint pad1;
};

ParameterBlock<TextureTable>   table;
RWStructuredBuffer<float4>     result;
ConstantBuffer<ProbeParams>    params;

[shader("compute")]
[numthreads(1, 1, 1)]
void textureTableProbe(uint3 tid: SV_DispatchThreadID) {
    for (uint i = 0; i < params.count; ++i) {
        result[i] = table.textures[i * params.stride].SampleLevel(table.samplers[i % 2], float2(0.5, 0.5), 0.0);
    }
}
