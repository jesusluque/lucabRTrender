// Copyright (c) 2026 lucabRTrender contributors.
//
// The path tracer's first-hit outputs, checked as counters: an albedo that is
// the authored colour where the lobe returns everything (a Lambert of
// roughness 0 has directional albedo 1), a normal of unit length, and a normal
// that faces the eye -- which for a plane at z = -5 seen from the origin is
// +z in world space.
module aux_check;

struct AuxParams {
    float  albedoR; float albedoG; float albedoB;
    float  tolerance;
    uint   pixels;
    uint   pad0; uint pad1; uint pad2;
};

StructuredBuffer<float4>   planes;   // the albedo's plane, then the normal's
RWStructuredBuffer<uint>   counts;   // [covered, albedo off, not unit, not facing]
RWStructuredBuffer<float>  worst;    // [albedo error, length error]
ConstantBuffer<AuxParams>  aux;

[shader("compute")]
[numthreads(1, 1, 1)]
void auxCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint covered = 0;
    uint albedoOff = 0;
    uint notUnit = 0;
    uint notFacing = 0;
    float worstAlbedo = 0.0;
    float worstLength = 0.0;
    const float3 expected = float3(aux.albedoR, aux.albedoG, aux.albedoB);
    for (uint i = 0; i < aux.pixels; ++i) {
        const float4 a = planes[i];
        const float4 n = planes[aux.pixels + i];
        if (a.w <= 0.0) {
            continue;
        }
        ++covered;
        const float3 d = abs(a.rgb - expected);
        const float e = max(d.r, max(d.g, d.b));
        worstAlbedo = max(worstAlbedo, e);
        albedoOff += e > aux.tolerance ? 1u : 0u;
        const float len = length(n.xyz);
        worstLength = max(worstLength, abs(len - 1.0));
        notUnit += abs(len - 1.0) > 1e-4 ? 1u : 0u;
        notFacing += n.z > 0.99 ? 0u : 1u;
    }
    counts[0] = covered;
    counts[1] = albedoOff;
    counts[2] = notUnit;
    counts[3] = notFacing;
    worst[0] = worstAlbedo;
    worst[1] = worstLength;
}
