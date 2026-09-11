// Copyright (c) 2026 lucabRTrender contributors.
//
// <conductor_bsdf>
#include "libraries/pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_conductor_bsdf(ClosureData closureData, float weight, float3 ior_n, float3 ior_k, float2 roughness,
                        bool retroreflective, float thinfilm_thickness, float thinfilm_ior, float3 N, float3 X,
                        int distribution, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);
    if (weight < M_FLOAT_EPS)
        return;
    Lobe lobe = lrtLobe(kLobeConductor);
    lobe.colour0 = ior_n;
    lobe.colour1 = ior_k;
    lobe.alpha = roughness;
    bsdf = lrtBsdfOf(lrtPushLobe(lobe, N, X), float3(weight), float3(0.0));
}
