// Copyright (c) 2026 lucabRTrender contributors.
//
// TextureStore's checks: a byte pattern generated here and written to a file,
// read back through the store and compared texel by texel; samples through
// the texture table, as materials take them.
module material_textures;
import lrt.material.texture_table;

struct PatternParams {
    uint  width;
    uint  height;
    uint  constant;     // 1: every texel `colour`, for UDIM tiles
    uint  count;        // probes
    float colourR; float colourG; float colourB; float tolerance;
};

RWStructuredBuffer<uint>      words;     // pattern: RGBA8 file bytes, top row first
Texture2D<float4>             level;     // check: a slot's texture, a level of it
RWStructuredBuffer<uint>      counts;
RWStructuredBuffer<float4>    probes;    // table probe: uv (xy), id (z), expected-found (w) in; sample out
RWStructuredBuffer<float4>    sampled;
ParameterBlock<TextureTable>  table;
ConstantBuffer<PatternParams> params;

uint patternByte(uint x, uint y, uint c) {
    if (params.constant != 0) {
        const float3 colour = float3(params.colourR, params.colourG, params.colourB);
        return c == 3 ? 255u : uint(round(colour[c] * 255.0));
    }
    return (x * 31u + y * 17u + c * 53u + (x * y) % 7u) % 256u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void materialPattern(uint3 tid: SV_DispatchThreadID) {
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            for (uint c = 0; c < 4; ++c) {
                const uint index = (y * params.width + x) * 4 + c;
                words[index >> 2] |= patternByte(x, y, c) << ((index & 3u) * 8u);
            }
        }
    }
}

/// Level 0 of the slot's texture against the pattern, v up: texel row y is
/// file row height - 1 - y. Stored bytes, whatever the view decodes.
[shader("compute")]
[numthreads(1, 1, 1)]
void materialLevelCheck(uint3 tid: SV_DispatchThreadID) {
    uint wrong = 0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            const float4 got = level.Load(int3(int(x), int(y), 0));
            for (uint c = 0; c < 4; ++c) {
                const float want = float(patternByte(x, params.height - 1 - y, c)) / 255.0;
                wrong += abs(got[c] - want) > 0.5 / 255.0 ? 1u : 0u;
            }
        }
    }
    counts[0] = wrong;
    counts[1] = params.width * params.height * 4;
}

/// A level's mean, decoded from sRGB when asked (colour 0: raw, 1: sRGB).
[shader("compute")]
[numthreads(1, 1, 1)]
void materialLevelMean(uint3 tid: SV_DispatchThreadID) {
    float4 sum = float4(0.0);
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            float4 t = level.Load(int3(int(x), int(y), 0));
            if (params.colourR > 0.5) {
                t.rgb = select(t.rgb <= 0.04045, t.rgb / 12.92, pow((t.rgb + 0.055) / 1.055, 2.4));
            }
            sum += t;
        }
    }
    sampled[0] = sum / float(params.width * params.height);
}

/// Samples through the table at given uvs; found goes into w.
[shader("compute")]
[numthreads(1, 1, 1)]
void materialTableProbe(uint3 tid: SV_DispatchThreadID) {
    for (uint i = 0; i < params.count; ++i) {
        bool found;
        const float4 value = sampleTexture(table, uint(probes[i].z), 0, probes[i].xy, float2(0.0), float2(0.0), found);
        sampled[i] = float4(value.rgb, found ? 1.0 : 0.0);
    }
}
