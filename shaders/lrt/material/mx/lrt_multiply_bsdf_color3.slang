// Copyright (c) 2026 lucabRTrender contributors.
//
// <multiply> by a colour
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_multiply_bsdf_color3(ClosureData closureData, BSDF in1, float3 in2, out BSDF result)
{
    result = in1;
    const float3 tint = clamp(in2, 0.0, 1.0);
    for (uint k = 0; k < kBuildLobes; ++k)
        result.weights[k] = in1.weights[k] * tint;
}
