// Copyright (c) 2026 lucabRTrender contributors.
//
// Every texture a frame's materials sample, by id: a table of texture slots
// and samplers in one parameter block, and a record per id saying which slot
// holds it -- or, for a UDIM set, which slot holds each of its tiles.
// See modules/material/src/TextureStore.cpp.
module texture_table;

public static const uint kTextureSlots = 1024;
public static const uint kSamplerSlots = 16;

public struct TextureRecord {
    public uint  slot;       // the texture's slot; a UDIM set's: unused
    public uint  udimBase;   // a UDIM set: its 100 cells in `udim` (tiles 1001..1100), each slot + 1 or 0
    public uint  flags;      // bit 0: loaded, bit 1: a UDIM set
    public uint  pad0;
    public float width;
    public float height;
    public float pad1;
    public float pad2;
};

public struct TextureTable {
    public Texture2D<float4>                textures[kTextureSlots];
    public SamplerState                     samplers[kSamplerSlots];
    public StructuredBuffer<TextureRecord>  records;
    public StructuredBuffer<uint>           udim;
};

/// Texture `id` at `uv` (v up, UDIM tiles past 1), filtered over the
/// footprint `dx`, `dy`; `found` is false where the texture did not load or
/// the UDIM set has no tile there -- the caller's default applies.
/// The mip level a footprint covers: its wider side in texels, as
/// SampleGrad would choose.
public float footprintLod(TextureRecord record, float2 dx, float2 dy) {
    const float2 size = float2(max(record.width, 1u), max(record.height, 1u));
    const float2 a = dx * size;
    const float2 b = dy * size;
    const float length2 = max(dot(a, a), dot(b, b));
    return length2 > 0.0 ? 0.5 * log2(length2) : 0.0;
}

public float4 sampleTexture(TextureTable table, uint id, uint sampler, float2 uv, float2 dx, float2 dy,
                            out bool found) {
    const TextureRecord record = table.records[id];
    found = false;
    if ((record.flags & 1u) == 0) {
        return float4(0.0);
    }
    uint slot = record.slot;
    if ((record.flags & 2u) != 0) {
        const float2 tile = floor(uv);
        if (tile.x < 0.0 || tile.y < 0.0 || tile.x > 9.0 || tile.y > 9.0) {
            return float4(0.0);
        }
        const uint cell = table.udim[record.udimBase + uint(tile.y) * 10 + uint(tile.x)];
        if (cell == 0) {
            return float4(0.0);
        }
        slot = cell - 1;
        uv -= tile;
    }
    found = true;
    // A footprint, filtered as the target can: with gradients where a compute
    // entry point may ask for them, and otherwise from the level the wider
    // side of the footprint lands on. CUDA has no SampleGrad in compute.
    __target_switch {
    case cuda:
        return table.textures[slot].SampleLevel(table.samplers[sampler], uv,
                                                footprintLod(record, dx, dy));
    default:
        return table.textures[slot].SampleGrad(table.samplers[sampler], uv, dx, dy);
    }
}
