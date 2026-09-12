// Copyright (c) 2026 lucabRTrender contributors.
//
// Each lobe against itself on the device:
//   - chi-square: directions drawn by lobeSample, binned over the sphere in
//     the lobe's frame, against lobePdf integrated over each bin;
//   - furnace: the lobe's albedo from its own samples' weights, and from
//     eval over uniformly drawn directions -- the two must agree, and a lobe
//     that loses no light returns 1.
module lobe_check;
import lrt.material.lobes;

struct LobeParams {
    uint   kind;
    uint   flags;
    uint   scatter;
    uint   samples;
    float3 colour0;  float ior;
    float3 colour1;  float exponent;
    float3 colour2;  float roughness;
    float  alphaX;   float alphaY;   float cosThetaO;   uint seed;
    uint   thetaBins; uint phiBins; uint pad0; uint pad1;
    float4 hairA;    float4 hairB;
};

RWStructuredBuffer<uint>    bins;        // per sample: its bin, or thetaBins * phiBins where none
RWStructuredBuffer<float4>  weights;     // per sample: the lobe's sample weight (rgb), valid (a)
RWStructuredBuffer<float4>  uniforms;    // per sample: eval over a uniform direction, times 4 pi
RWStructuredBuffer<float>   observed;    // per bin
RWStructuredBuffer<float>   expected;    // per bin
RWStructuredBuffer<float4>  result;
ConstantBuffer<LobeParams>  params;

Lobe lobeOf() {
    Lobe lobe;
    lobe.kind = params.kind;
    lobe.flags = params.flags;
    lobe.scatter = params.scatter;
    lobe.weight = float3(1.0);
    lobe.roughness = params.roughness;
    lobe.colour0 = params.colour0;
    lobe.colour1 = params.colour1;
    lobe.colour2 = params.colour2;
    lobe.ior = params.ior;
    lobe.exponent = params.exponent;
    lobe.alpha = float2(params.alphaX, params.alphaY);
    lobe.normal = float3(0.0, 0.0, 1.0);
    lobe.tangent = float3(1.0, 0.0, 0.0);
    if (lobe.kind == kLobeHair) {
        setHair(lobe, params.hairA.xyz, float2(params.hairA.w, params.hairB.x), params.hairB.yz);
    }
    return lobe;
}

float3 viewer() {
    const float s = sqrt(max(0.0, 1.0 - params.cosThetaO * params.cosThetaO));
    return float3(s * 0.8, s * 0.6, params.cosThetaO);
}

uint binOf(float3 w) {
    const float theta = acos(clamp(w.z, -1.0, 1.0));
    float phi = atan2(w.y, w.x);
    if (phi < 0.0) {
        phi += 2.0 * kPi;
    }
    const uint t = min(uint(theta / kPi * float(params.thetaBins)), params.thetaBins - 1);
    const uint p = min(uint(phi / (2.0 * kPi) * float(params.phiBins)), params.phiBins - 1);
    return t * params.phiBins + p;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lobeDraw(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.samples) {
        return;
    }
    Rng rng = Rng(i, params.seed);
    const Lobe lobe = lobeOf();
    const float3 wo = viewer();
    const float3 u = float3(rng.next(), rng.next(), rng.next());
    const LobeSample s = lobeSample(lobe, wo, u);
    bins[i] = s.valid && !s.delta ? binOf(s.wi) : params.thetaBins * params.phiBins;
    weights[i] = s.valid ? float4(s.weight, 1.0) : float4(0.0);
    const float3 w = sampleUniformSphere(float2(rng.next(), rng.next()));
    uniforms[i] = float4(lobeEval(lobe, wo, w) * 4.0 * kPi, 1.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lobeCount(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.thetaBins * params.phiBins;
    for (uint b = 0; b <= n; ++b) {
        observed[b] = 0.0;
    }
    float3 sampled = float3(0.0);
    float3 uniform = float3(0.0);
    for (uint i = 0; i < params.samples; ++i) {
        observed[bins[i]] += 1.0;
        sampled += weights[i].rgb;
        uniform += uniforms[i].rgb;
    }
    result[0] = float4(sampled / float(params.samples), 0.0);
    result[1] = float4(uniform / float(params.samples), 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lobeExpect(uint3 tid: SV_DispatchThreadID) {
    const Lobe lobe = lobeOf();
    const float3 wo = viewer();
    const uint sub = 8;
    const float dTheta = kPi / float(params.thetaBins * sub);
    const float dPhi = 2.0 * kPi / float(params.phiBins * sub);
    float integral = 0.0;
    for (uint t = 0; t < params.thetaBins; ++t) {
        for (uint p = 0; p < params.phiBins; ++p) {
            float sum = 0.0;
            for (uint j = 0; j < sub; ++j) {
                const float theta = (float(t * sub + j) + 0.5) * dTheta;
                const float sinTheta = sin(theta);
                for (uint k = 0; k < sub; ++k) {
                    const float phi = (float(p * sub + k) + 0.5) * dPhi;
                    const float3 w = float3(sinTheta * cos(phi), sinTheta * sin(phi), cos(theta));
                    sum += lobePdf(lobe, wo, w) * sinTheta * dTheta * dPhi;
                }
            }
            expected[t * params.phiBins + p] = sum * float(params.samples);
            integral += sum;
        }
    }
    result[2] = float4(integral, 0.0, 0.0, 0.0);
}

/// Pearson's statistic over bins expecting at least 5, the rest pooled.
[shader("compute")]
[numthreads(1, 1, 1)]
void lobeStatistic(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.thetaBins * params.phiBins;
    float statistic = 0.0;
    float dof = 0.0;
    float pooledObserved = 0.0;
    float pooledExpected = 0.0;
    float drawn = 0.0;
    for (uint b = 0; b < n; ++b) {
        drawn += observed[b];
        if (expected[b] < 5.0) {
            pooledObserved += observed[b];
            pooledExpected += expected[b];
            continue;
        }
        statistic += (observed[b] - expected[b]) * (observed[b] - expected[b]) / expected[b];
        dof += 1.0;
    }
    if (pooledExpected >= 5.0) {
        statistic += (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected) / pooledExpected;
        dof += 1.0;
    }
    result[3] = float4(statistic, max(dof - 1.0, 1.0), drawn / float(params.samples), 0.0);
}
