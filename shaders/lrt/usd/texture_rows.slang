// Copyright (c) 2026 lucabRTrender contributors.
//
// A texture as rows of `float4`, for an effect that is handed pictures.
//
// An AOFX effect reads buffers, not textures -- it never sees a sampler and
// never chooses a mip. So a map a material names has to arrive as a picture,
// and this is what turns one into the other: a thread a texel, sampled through
// the same texture table every material in this engine samples
// (`material/texture_table.slang`), so the colour space, the wrap and the
// filtering are the ones the renderer would have used.
//
// It is a kernel and not a host loop on purpose. A texture's bytes are 8-bit
// sRGB as often as not, and turning those into linear floats is arithmetic on
// scene data -- which the host does not do here. `TextureStore` already put
// the file on the device; this reads it back out in the shape the effect wants.
//
// Rows are bottom row first (v up), as every picture this engine carries is.
module texture_rows;
import lrt.material.texture_table;

struct RowsParams {
    uint  id;          ///< the texture, in the table
    uint  sampler;     ///< the sampler slot
    uint  width;       ///< of the destination
    uint  height;

    uint  stride;      ///< pixels a row of the destination
    uint  channels;    ///< 15: all four. Otherwise a mask: 1 r, 2 g, 4 b, 8 a
    uint  fromChannel; ///< which channel of the sample the masked write takes
    uint  clear;       ///< 1: write the fallback where the mask does not reach

    float4 fallback;   ///< where the texture did not load, or is not there
};

RWStructuredBuffer<float4>   rows;
ParameterBlock<TextureTable> table;
ConstantBuffer<RowsParams>   params;

[shader("compute")]
[numthreads(8, 8, 1)]
void textureRows(uint3 tid: SV_DispatchThreadID) {
    const uint x = tid.x;
    const uint y = tid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    const uint at = y * params.stride + x;
    const float2 uv = (float2(float(x), float(y)) + 0.5) / float2(float(params.width), float(params.height));
    // One texel of the destination is one texel's worth of footprint: the mip
    // this picks is the one a renderer would pick at this density.
    const float2 dx = float2(1.0 / float(max(params.width, 1u)), 0.0);
    const float2 dy = float2(0.0, 1.0 / float(max(params.height, 1u)));
    bool found = false;
    const float4 sampled = sampleTexture(table, params.id, params.sampler, uv, dx, dy, found);
    const float4 value = found ? sampled : params.fallback;

    if (params.channels == 15u) {
        rows[at] = value;
        return;
    }
    // One channel of a picture that several textures share: a metallic file
    // and a roughness file become the glTF packing the conversion expects,
    // written a channel at a time into the same rows.
    float4 held = params.clear != 0 ? params.fallback : rows[at];
    const float one = value[min(params.fromChannel, 3u)];
    for (uint c = 0; c < 4; ++c) {
        if ((params.channels & (1u << c)) != 0) {
            held[c] = one;
        }
    }
    rows[at] = held;
}
