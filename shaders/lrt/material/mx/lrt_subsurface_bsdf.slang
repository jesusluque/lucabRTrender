// Copyright (c) 2026 lucabRTrender contributors.
//
// <subsurface_bsdf>: diffuse, until the path tracer walks inside (M6).
#include "libraries/pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_subsurface_bsdf(ClosureData closureData, float weight, float3 color, float3 radius, float anisotropy,
                         float3 N, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);
    if (weight < M_FLOAT_EPS)
        return;
    Lobe lobe = lrtLobe(kLobeOrenNayar);
    lobe.colour0 = color;
    bsdf = lrtBsdfOf(lrtPushLobe(lobe, N, float3(1.0, 0.0, 0.0)), float3(weight), float3(0.0));
}
