// Copyright (c) 2026 lucabRTrender contributors.
//
// <burley_diffuse_bsdf>
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_burley_diffuse_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N,
                             inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);
    if (weight < M_FLOAT_EPS)
        return;
    Lobe lobe = lrtLobe(kLobeBurley);
    lobe.colour0 = color;
    lobe.roughness = roughness;
    bsdf = lrtBsdfOf(lrtPushLobe(lobe, N, float3(1.0, 0.0, 0.0)), float3(weight), float3(0.0));
}
