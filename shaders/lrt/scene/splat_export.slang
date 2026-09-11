// Copyright (c) 2026 lucabRTrender contributors.
//
// Raw records -> the values USD's ParticleField stores: linear opacity, linear
// scale, a normalised quaternion, SH coefficients with the DC term first. Done
// here for the reason every decode is: the CPU writes the file, the GPU does
// the arithmetic. Full float32 -- this feeds a file, not a frame.
module splat_export;
import lrt.common.packing;
import splat_encoding;

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
    const float3 p = float3(fieldOf(raw, i, s, params.x), fieldOf(raw, i, s, params.y),
                            fieldOf(raw, i, s, params.z));
    const float a = decodeOpacity(fieldOf(raw, i, s, params.opacity), params.opacityMode);
    const float3 ls = float3(decodeLogScale(fieldOf(raw, i, s, params.scale0), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale1), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale2), params.scaleMode));
    const bool valid = all(isfinite(p)) && all(isfinite(ls)) && all(abs(ls) < 60.0) && a >= 1.0 / 255.0;
    float4 q = float4(decodeRotation(fieldOf(raw, i, s, params.rotX), params.rotationMode),
                      decodeRotation(fieldOf(raw, i, s, params.rotY), params.rotationMode),
                      decodeRotation(fieldOf(raw, i, s, params.rotZ), params.rotationMode),
                      decodeRotation(fieldOf(raw, i, s, params.rotW), params.rotationMode));
    const float qLength = length(q);
    q = qLength > 1e-8 ? q / qLength : float4(0, 0, 0, 1);

    posOpacity[i] = float4(p, saturate(a));
    rotation[i] = q;
    scaleValid[i] = float4(exp(ls), valid ? 1.0 : 0.0);

    const uint row = i * (1 + params.keepPerColour);
    float3 dc = float3(fieldOf(raw, i, s, params.dc0), fieldOf(raw, i, s, params.dc1),
                       fieldOf(raw, i, s, params.dc2));
    if (params.colourMode == 1) {
        dc = (dc - 0.5) / kSH0;
    } else if (params.colourMode == 2) {
        dc = (dc / 255.0 - 0.5) / kSH0;
    }
    coefficients[row] = float4(dc, 0.0);
    for (uint basis = 0; basis < params.keepPerColour; ++basis) {
        float3 c = float3(0.0);
        for (uint colour = 0; colour < 3; ++colour) {
            const uint field = params.restColourOuter != 0
                                   ? params.restBase + colour * params.filePerColour + basis
                                   : params.restBase + basis * 3 + colour;
            c[colour] = fieldOf(raw, i, s, field);
        }
        coefficients[row + 1 + basis] = float4(c, 0.0);
    }
}
