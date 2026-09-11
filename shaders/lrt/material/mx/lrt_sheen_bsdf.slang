// Copyright (c) 2026 lucabRTrender contributors.
//
// <sheen_bsdf>
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_sheen_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N, int mode,
                    inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
        return;
    const float3 V = closureData.V;
    const float3 n = dot(N, V) < 0.0 ? -N : N;
    const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
    const float dirAlbedo = mode == 0 ? imageworksSheenDirectionalAlbedo(NdotV, roughness)
                                      : zeltnerSheenDirectionalAlbedo(NdotV, clamp(roughness, 0.01, 1.0));
    Lobe lobe = lrtLobe(kLobeSheen);
    lobe.colour0 = color;
    lobe.roughness = roughness;
    lobe.flags = mode == 0 ? 0u : kFlagZeltner;
    bsdf = lrtBsdfOf(lrtPushLobe(lobe, N, float3(1.0, 0.0, 0.0)), float3(weight), float3(1.0 - dirAlbedo * weight));
}
