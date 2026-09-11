// Copyright (c) 2026 lucabRTrender contributors.
//
// <layer> of a BSDF over a VDF: what the medium lets through scales the top's
// lobes, as genglsl scales its response. Absorption along real paths waits
// for the path tracer's media (M9).
#include "libraries/pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_layer_vdf(ClosureData closureData, BSDF top, VDF base, out BSDF result)
{
    result = top;
    for (uint k = 0; k < kBuildLobes; ++k)
        result.weights[k] = top.weights[k] * base.throughput;
    result.throughput = top.throughput * base.throughput;
}
