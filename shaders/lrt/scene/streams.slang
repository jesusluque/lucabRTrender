// Copyright (c) 2026 lucabRTrender contributors.
//
// Clouds as USD holds them -- one array per attribute, float32 or float16 --
// interleaved into raw records on the device, which then take the same
// validate and decode every file format takes. The CPU uploads each array's
// bytes as they are: no loop over elements, no half to float on the host.
//
//   splat record  0..2 position   3 opacity (linear)   4..6 scale (linear)
//                 7 w  8 x  9 y  10 z   11..13 SH DC   14.. rest, rgb per basis
//   point record  0..2 position   3..5 colour (linear)
module streams;
import lrt.common.packing;

/// Value `index` of a float array: float32 words, or float16 halves packed two
/// to a little-endian word, first in the low bits.
float streamValue(StructuredBuffer<uint> words, bool half, uint index) {
    if (half) {
        const uint word = words[index / 2];
        return (index & 1u) == 0 ? lowHalf(word) : highHalf(word);
    }
    return bit_cast<float>(words[index]);
}

static const uint kPositions = 1;
static const uint kRotations = 2;
static const uint kScales = 4;
static const uint kOpacities = 8;
static const uint kSh = 16;
static const uint kColours = 32;
static const uint kMetallic = 64;
static const uint kRoughness = 128;

struct StreamParams {
    uint count;          // records in this slice
    uint first;          // the slice's first element
    uint stride;         // floats per record
    uint keep;           // rest basis functions written per colour
    uint coefficients;   // SH coefficients per splat in the stream (DC first)
    uint present;        // which streams are bound (k* bits)
    uint halves;         // which of them are float16
    uint colourMode;     // points: 0 none (white), 1 one colour, 2 one per point
    uint pbrBase;        // where metallic and roughness go in a record, or kNoPbr
    uint pad0; uint pad1; uint pad2;
};

static const uint kNoPbr = 0xFFFFFFFFu;

StructuredBuffer<uint>     positions;   // xyz
StructuredBuffer<uint>     rotations;   // xyzw, GfQuat's layout
StructuredBuffer<uint>     scales;      // xyz
StructuredBuffer<uint>     opacities;
StructuredBuffer<uint>     sh;          // rgb per coefficient
StructuredBuffer<uint>     colours;     // rgb
StructuredBuffer<uint>     metallic;    // one a splat, where the stage carries it
StructuredBuffer<uint>     roughness;
RWStructuredBuffer<float>  records;
ConstantBuffer<StreamParams> params;

bool present(uint stream) { return (params.present & stream) != 0; }
bool isHalf(uint stream) { return (params.halves & stream) != 0; }

[shader("compute")]
[numthreads(256, 1, 1)]
void splatStreams(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint e = params.first + i;
    const uint r = i * params.stride;
    for (uint k = 0; k < 3; ++k) {
        records[r + k] = streamValue(positions, isHalf(kPositions), e * 3 + k);
        records[r + 4 + k] = present(kScales) ? streamValue(scales, isHalf(kScales), e * 3 + k) : 1.0;
    }
    records[r + 3] = present(kOpacities) ? streamValue(opacities, isHalf(kOpacities), e) : 1.0;
    if (present(kRotations)) {
        records[r + 7] = streamValue(rotations, isHalf(kRotations), e * 4 + 3);
        for (uint k = 0; k < 3; ++k) {
            records[r + 8 + k] = streamValue(rotations, isHalf(kRotations), e * 4 + k);
        }
    } else {
        records[r + 7] = 1.0;
        records[r + 8] = 0.0;
        records[r + 9] = 0.0;
        records[r + 10] = 0.0;
    }
    const bool haveSh = present(kSh) && params.coefficients > 0;
    const uint base = e * params.coefficients;
    for (uint k = 0; k < 3; ++k) {
        // No harmonics: DC 0, which is base colour 0.5, the schema's default.
        records[r + 11 + k] = haveSh ? streamValue(sh, isHalf(kSh), base * 3 + k) : 0.0;
    }
    for (uint basis = 0; basis < params.keep; ++basis) {
        for (uint c = 0; c < 3; ++c) {
            records[r + 14 + basis * 3 + c] = streamValue(sh, isHalf(kSh), (base + 1 + basis) * 3 + c);
        }
    }
    // What the gaussian reflects with, after the harmonics: a cloud converted
    // from a mesh brings them, a capture does not.
    if (params.pbrBase != kNoPbr) {
        records[r + params.pbrBase] = present(kMetallic) ? streamValue(metallic, isHalf(kMetallic), e) : 0.0;
        records[r + params.pbrBase + 1] = present(kRoughness) ? streamValue(roughness, isHalf(kRoughness), e) : 1.0;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pointStreams(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint e = params.first + i;
    const uint r = i * 6;
    const uint from = params.colourMode == 2 ? e : 0;
    for (uint k = 0; k < 3; ++k) {
        records[r + k] = streamValue(positions, isHalf(kPositions), e * 3 + k);
        records[r + 3 + k] = params.colourMode != 0 ? streamValue(colours, isHalf(kColours), from * 3 + k) : 1.0;
    }
}
