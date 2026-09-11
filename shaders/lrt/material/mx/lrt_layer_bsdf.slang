// Copyright (c) 2026 lucabRTrender contributors.
//
// <layer>: the base beneath what the top lets through.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_layer_bsdf(ClosureData closureData, BSDF top, BSDF base, out BSDF result)
{
    result = lrt_bsdf_zero();
    for (uint k = 0; k < kBuildLobes; ++k)
        result.weights[k] = top.weights[k] + base.weights[k] * top.throughput;
    result.throughput = top.throughput * base.throughput;
}
