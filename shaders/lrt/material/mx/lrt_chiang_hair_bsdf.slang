// Copyright (c) 2026 lucabRTrender contributors.
//
// <chiang_hair_bsdf>: the hair lobe; no throughput, as genglsl.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_chiang_hair_bsdf(ClosureData closureData, float3 tint_R, float3 tint_TT, float3 tint_TRT, float ior,
                          float2 roughness_R, float2 roughness_TT, float2 roughness_TRT, float cuticle_angle,
                          float3 absorption_coefficient, float3 N, float3 X, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);
    Lobe lobe = lrtLobe(kLobeHair);
    lobe.colour0 = tint_R;
    lobe.colour1 = tint_TT;
    lobe.colour2 = tint_TRT;
    lobe.ior = ior;
    lobe.alpha = roughness_R;
    lobe.roughness = cuticle_angle;
    setHair(lobe, absorption_coefficient, roughness_TT, roughness_TRT);
    bsdf = lrtBsdfOf(lrtPushLobe(lobe, N, X), float3(1.0), float3(0.0));
}
