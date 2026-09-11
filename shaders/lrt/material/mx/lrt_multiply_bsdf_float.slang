// Copyright (c) 2026 lucabRTrender contributors.
//
// <multiply> by a scalar
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_multiply_bsdf_float(ClosureData closureData, BSDF in1, float in2, out BSDF result)
{
    result = in1;
    const float weight = clamp(in2, 0.0, 1.0);
    for (uint k = 0; k < kBuildLobes; ++k)
        result.weights[k] = in1.weights[k] * weight;
}
