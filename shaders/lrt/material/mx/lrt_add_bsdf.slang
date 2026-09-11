// Copyright (c) 2026 lucabRTrender contributors.
//
// <add>
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_add_bsdf(ClosureData closureData, BSDF in1, BSDF in2, out BSDF result)
{
    result = lrt_bsdf_zero();
    for (uint k = 0; k < kBuildLobes; ++k)
        result.weights[k] = in1.weights[k] + in2.weights[k];
    result.throughput = max(in1.throughput + in2.throughput - 1.0, 0.0);
}
