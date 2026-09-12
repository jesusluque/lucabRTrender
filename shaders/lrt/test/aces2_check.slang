// Copyright (c) 2026 lucabRTrender contributors.
//
// ACES 2.0 against what it must hold to: a neutral in is a neutral out
// whose luminance is the tonescale's -- written here a second time, from
// the published parameters, in its own form -- and monotonic in the input;
// the inverse transform undoes the forward one over the display cube; and
// the gamut compression leaves nothing outside the cube. Counts and worst
// cases come back, never images.
import lrt.technique.aces2;

struct CheckParams {
    uint  count;         // samples
    uint  pad0;
    float peakLuminance;
    float tolerance;
};

ConstantBuffer<CheckParams>   check;
StructuredBuffer<Aces2Params> params;
StructuredBuffer<float>       tables;
RWStructuredBuffer<uint>      counts;   // neutral: [not neutral, off the tonescale, not monotonic]; round trip: [off]; gamut: [outside]
RWStructuredBuffer<uint>      worst;    // float bits: [largest relative error, the 18% grey's luminance]

void maxBits(uint slot, float value) {
    InterlockedMax(worst[slot], asuint(max(value, 0.0)));
}

/// The tonescale, from the published constants, in the form of the ACES 2
/// tonescale paper: the Michaelis-Menten curve f = m_2 (x / (x + s_2))^g
/// with the flare term h = f^2 / (f + t_1), in exp/log rather than pow.
float tonescaleAgain(float x, float n) {
    const float n_r = 100.0, g = 1.15, c = 0.18, c_d = 10.013, w_g = 0.14, t_1 = 0.04;
    const float r_hit = 128.0 + (896.0 - 128.0) * (log(n / n_r) / log(100.0));
    const float m_0 = n / n_r;
    const float m_1 = 0.5 * (m_0 + sqrt(m_0 * m_0 + 4.0 * t_1 * m_0));
    const float u = exp(g * log((r_hit / m_1) / (r_hit / m_1 + 1.0)));
    const float m = m_1 / u;
    const float w_i = log2(n / 100.0);
    const float c_t = c_d / n_r * (1.0 + w_i * w_g);
    const float g_ip = 0.5 * (c_t + sqrt(c_t * c_t + 4.0 * t_1 * c_t));
    const float q = exp(log(g_ip / m) / g);
    const float g_ipp2 = -(m_1 * q) / (q - 1.0);
    const float w_2 = c / g_ipp2;
    const float s_2 = w_2 * m_1;
    const float u_2 = exp(g * log((r_hit / m_1) / (r_hit / m_1 + w_2)));
    const float m_2 = m_1 / u_2;
    const float f = x <= 0.0 ? 0.0 : m_2 * exp(g * log(x / (x + s_2)));
    const float h = f * f / (f + t_1);
    return h * n_r / 100.0;   // in the transform's units: 1.0 is 100 nits
}

float3 neutralOut(uint i) {
    // Scene greys from 2^-10 to 2^6, through the transform as the display kernel runs it.
    const float x = exp2(-10.0 + 16.0 * float(i) / float(check.count));
    const Aces2Params p = params[0];
    return outputTransformFwd(mul(float3(x), p.sceneToAP0), p, tables);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aces2Neutral(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.count) return;
    const float x = exp2(-10.0 + 16.0 * float(i) / float(check.count));
    const float3 out = neutralOut(i);
    const float mean = (out.x + out.y + out.z) / 3.0;
    const float spread = max(max(abs(out.x - mean), abs(out.y - mean)), abs(out.z - mean)) / max(mean, 1e-3);
    maxBits(0, spread);
    if (spread > check.tolerance) InterlockedAdd(counts[0], 1);
    const float expected = tonescaleAgain(x, check.peakLuminance);
    const float off = abs(mean - expected) / max(expected, 1e-3);
    maxBits(2, off);
    if (off > check.tolerance) InterlockedAdd(counts[1], 1);
    if (i > 0) {
        const float3 before = neutralOut(i - 1);
        if ((before.x + before.y + before.z) >= (out.x + out.y + out.z)) InterlockedAdd(counts[2], 1);
    }
    // The 18% grey, for the print: 10.013 nits less its flare, 0.09999 of 100.
    if (i == 0) {
        const float3 grey = outputTransformFwd(mul(float3(0.18), params[0].sceneToAP0), params[0], tables);
        worst[1] = asuint((grey.x + grey.y + grey.z) / 3.0);
    }
}

uint hashOf(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float unitOf(uint x) { return float(hashOf(x) & 0xFFFFFFu) / 16777216.0; }

[shader("compute")]
[numthreads(64, 1, 1)]
void aces2RoundTrip(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.count) return;
    // Display RGB inside the cube, the peak's units, away from its faces.
    const float scale = check.peakLuminance / 100.0;
    const float3 rgb = scale * (0.03 + 0.94 * float3(unitOf(i * 3u + 1u), unitOf(i * 3u + 2u), unitOf(i * 3u + 3u)));
    const Aces2Params p = params[0];
    const float3 aces = outputTransformInv(rgb, p, tables);
    const float3 back = outputTransformFwd(aces, p, tables);
    const float off = max(max(abs(back.x - rgb.x), abs(back.y - rgb.y)), abs(back.z - rgb.z)) / scale;
    maxBits(0, off);
    if (off > check.tolerance) InterlockedAdd(counts[0], 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aces2Gamut(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.count) return;
    // Scene colours of any saturation, over sixteen stops.
    const float exposure = exp2(-8.0 + 16.0 * unitOf(i * 5u + 1u));
    const float3 c = exposure * float3(unitOf(i * 5u + 2u), unitOf(i * 5u + 3u), unitOf(i * 5u + 4u));
    const Aces2Params p = params[0];
    const float3 out = outputTransformFwd(mul(c, p.sceneToAP0), p, tables);
    const float scale = check.peakLuminance / 100.0;
    const float beyond = max(max(max(-out.x, -out.y), max(-out.z, out.x - scale)), max(out.y - scale, out.z - scale));
    maxBits(0, beyond);
    if (beyond > check.tolerance) InterlockedAdd(counts[0], 1);
}
