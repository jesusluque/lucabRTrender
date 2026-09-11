// Copyright (c) 2026 lucabRTrender contributors.
//
// How a Gaussian is stored on the GPU, in one place.
//
//   positions  float4   xyz, opacity (linear, 0..1)            16 bytes
//   shape      uint * 4                                         16 bytes
//     [0] rotation, smallest-three: 10+10+10 bits and the dropped index
//     [1] f16 log(scale.x) | f16 log(scale.y) << 16
//     [2] f16 log(scale.z) | f16 base colour r << 16
//     [3] f16 base colour g | f16 base colour b << 16
//   sh         uint * words   f16 rest coefficients, rgb per basis
//
// 32 bytes before harmonics, against openFXplayer's 64 float32. Why each
// field is what it is:
//   - position stays float32: fp16 would step two pixels at 4K close up
//     (openFXplayer measured it), and it is what a camera flies into.
//   - scale is stored as a logarithm in fp16: 0.1% relative precision at every
//     size, where fp16 of the scale itself loses thin splats below 1e-4.
//   - rotation in 10-bit smallest-three is within a tenth of a degree.
//   - colours are fp16: base colour can leave 0..1, and harmonics are signed.
module packing;

public static const float kSqrtHalf = 0.70710678;

/// Smallest-three quaternion (x, y, z, w) -> uint.
public uint encodeQuaternion(float4 q) {
    q = normalize(q);
    uint largest = 0;
    float best = abs(q.x);
    for (uint k = 1; k < 4; ++k) {
        if (abs(q[k]) > best) {
            best = abs(q[k]);
            largest = k;
        }
    }
    if (q[largest] < 0.0) {
        q = -q;
    }
    uint word = largest;
    uint shift = 22;
    for (uint k = 0; k < 4; ++k) {
        if (k == largest) {
            continue;
        }
        const float unit = clamp(q[k] / kSqrtHalf * 0.5 + 0.5, 0.0, 1.0);
        word |= uint(round(unit * 1023.0)) << shift;
        shift -= 10;
    }
    return word;
}

public float4 decodeQuaternion(uint word) {
    const uint largest = word & 3u;
    float3 small = float3(float((word >> 22) & 1023u), float((word >> 12) & 1023u),
                          float((word >> 2) & 1023u));
    small = (small / 1023.0 * 2.0 - 1.0) * kSqrtHalf;
    const float rest = sqrt(max(0.0, 1.0 - dot(small, small)));
    float4 q = float4(0.0);
    uint next = 0;
    for (uint k = 0; k < 4; ++k) {
        if (k == largest) {
            q[k] = rest;
        } else {
            q[k] = small[next];
            next += 1;
        }
    }
    return q;
}

public uint packHalves(float a, float b) {
    return (f32tof16(a) & 0xFFFFu) | ((f32tof16(b) & 0xFFFFu) << 16);
}

public float lowHalf(uint word) { return f16tof32(word & 0xFFFFu); }
public float highHalf(uint word) { return f16tof32(word >> 16); }

/// f16 can hold +-65504; anything past that becomes infinity, which a
/// covariance turns into NaN. Clamped, not trusted.
public float halfSafe(float v) { return clamp(v, -65000.0, 65000.0); }

public static const float kSH0 = 0.28209479177387814;

/// Harmonic words per splat for `restPerColour` basis functions (3, 8, 15).
public uint shWordsFor(uint restPerColour) { return (restPerColour * 3 + 1) / 2; }
