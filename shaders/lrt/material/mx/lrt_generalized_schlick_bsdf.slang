// Copyright (c) 2026 lucabRTrender contributors.
//
// <generalized_schlick_bsdf>
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_generalized_schlick_bsdf(ClosureData closureData, float weight, float3 color0, float3 color82, float3 color90,
                                  float exponent, float2 roughness, bool retroreflective, float thinfilm_thickness,
                                  float thinfilm_ior, float3 N, float3 X, int distribution, int scatter_mode,
                                  inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
        return;
    const float3 V = closureData.V;
    const float3 n = dot(N, V) < 0.0 ? -N : N;
    const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
    const float avgAlpha = averageAlpha(clamp(roughness, M_FLOAT_EPS, 1.0));
    const float3 F = fresnelHoffmanSchlick(NdotV, max(color0, 0.0), max(color82, 0.0), max(color90, 0.0), exponent);
    const float3 comp = ggxEnergyCompensation(NdotV, avgAlpha, F);
    const float3 dirAlbedo = ggxDirectionalAlbedo(NdotV, avgAlpha, max(color0, 0.0), max(color90, 0.0)) * comp;
    const float average = dot(dirAlbedo, float3(1.0 / 3.0));
    Lobe lobe = lrtLobe(kLobeSchlick);
    lobe.colour0 = color0;
    lobe.colour1 = color90;
    lobe.colour2 = color82;
    lobe.exponent = exponent;
    lobe.alpha = roughness;
    lobe.scatter = uint(clamp(scatter_mode, 0, 2));
    // Transmission only leaves the throughput alone in genglsl's reflection pass.
    bsdf = lrtBsdfOf(lrtPushLobe(lobe, N, X), float3(weight),
                     scatter_mode == 1 ? float3(1.0) : float3(1.0 - average * weight));
}
