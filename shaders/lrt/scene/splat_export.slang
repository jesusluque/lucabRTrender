// Copyright (c) 2026 lucabRTrender contributors.
//
// Raw records -> the values USD's ParticleField stores: linear opacity, linear
// scale, a normalised quaternion, SH coefficients with the DC term first. Done
// here for the reason every decode is: the CPU writes the file, the GPU does
// the arithmetic. Full float32 -- this feeds a file, not a frame.
module splat_export;
import lrt.common.packing;
import splat_encoding;
import lrt.common.color;

StructuredBuffer<float>      raw;
RWStructuredBuffer<float4>   posOpacity;   // xyz, opacity
RWStructuredBuffer<float4>   rotation;     // x y z w
RWStructuredBuffer<float4>   scaleValid;   // xyz, 1 when the record is a splat
RWStructuredBuffer<float4>   coefficients; // (1 + keepPerColour) per record: rgb, 0
ConstantBuffer<DecodeParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatExport(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint s = params.stride;
    const float3 p = decodePosition(raw, i, params);
    const float a = decodeOpacity(fieldOf(raw, i, s, params.opacity), params.opacityMode);
    const float3 ls = float3(decodeLogScale(fieldOf(raw, i, s, params.scale0), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale1), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale2), params.scaleMode));
    const bool valid = all(isfinite(p)) && all(isfinite(ls)) && all(abs(ls) < 60.0) && a >= 1.0 / 255.0;
    float4 q = decodeQuaternion(raw, i, params);
    const float qLength = length(q);
    q = qLength > 1e-8 ? q / qLength : float4(0, 0, 0, 1);

    // A RECORD THAT IS NOT A SPLAT IS STILL A SLOT.
    //
    // It used to be dropped -- the writer skipped it and the file came out
    // shorter than the conversion. That is fine for one still frame and
    // impossible for a sequence: a gaussian is followed from one frame to the
    // next by being the same element, and a triangle that goes degenerate in
    // one pose would take its gaussians out of that frame's array alone and
    // put every later one out of step. So the slot is kept and written with
    // nothing in it: no opacity, no size, and a position that is at least a
    // number. Nothing draws, and the count is the conversion's.
    const float3 safe = all(isfinite(p)) ? p : float3(0.0);
    posOpacity[i] = valid ? float4(p, saturate(a)) : float4(safe, 0.0);
    rotation[i] = q;
    scaleValid[i] = valid ? float4(exp(ls), 1.0) : float4(0.0, 0.0, 0.0, 0.0);

    const uint row = i * (1 + params.keepPerColour);
    float3 dc = float3(fieldOf(raw, i, s, params.dc0), fieldOf(raw, i, s, params.dc1),
                       fieldOf(raw, i, s, params.dc2));
    if (params.colourMode == 4) {
        // Light, which the cloud carries in the space it is blended in.
        dc = (linearToSrgb(dc) - 0.5) / kSH0;
    } else if (params.colourMode == 1) {
        dc = (dc - 0.5) / kSH0;
    } else if (params.colourMode == 2) {
        dc = (dc / 255.0 - 0.5) / kSH0;
    } else if (params.colourMode == 3) {
        dc = (dc / 255.0 - 0.5) / 0.15;   // SPZ's DC scale
    }
    coefficients[row] = valid ? float4(dc, 0.0) : float4(0.0, 0.0, 0.0, 0.0);
    for (uint basis = 0; basis < params.keepPerColour; ++basis) {
        float3 c = float3(0.0);
        for (uint colour = 0; colour < 3; ++colour) {
            const uint field = params.restColourOuter != 0
                                   ? params.restBase + colour * params.filePerColour + basis
                                   : params.restBase + basis * 3 + colour;
            c[colour] = decodeRest(fieldOf(raw, i, s, field), params.restMode) *
                        (params.flipYZ != 0 ? restFlipSign(basis) : 1.0);
        }
        coefficients[row + 1 + basis] = valid ? float4(c, 0.0) : float4(0.0, 0.0, 0.0, 0.0);
    }
}
