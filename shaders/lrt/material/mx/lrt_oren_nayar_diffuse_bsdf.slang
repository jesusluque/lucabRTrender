// Copyright (c) 2026 lucabRTrender contributors.
//
// <oren_nayar_diffuse_bsdf>: a diffuse lobe; no throughput, as genglsl.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_oren_nayar_diffuse_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N,
                                 bool energy_compensation, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);
    if (weight < M_FLOAT_EPS)
        return;
    Lobe lobe = lrtLobe(kLobeOrenNayar);
    lobe.colour0 = color;
    lobe.roughness = roughness;
    lobe.flags = energy_compensation ? kFlagEnergyCompensation : 0u;
    bsdf = lrtBsdfOf(lrtPushLobe(lobe, N, float3(1.0, 0.0, 0.0)), float3(weight), float3(0.0));
}
