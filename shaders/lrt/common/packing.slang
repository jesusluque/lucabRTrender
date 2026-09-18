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

/// A rotation matrix whose columns are the axes, as (x, y, z, w).
///
/// It lives here beside the quaternion's own packing because two things now
/// need it: the LOD merge, which takes the eigenframe of a merged covariance,
/// and the splat skinner, which takes the frame a joint left a gaussian in.
public float4 quaternionOfAxes(float3x3 m) {
    const float trace = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (trace > 0.0) {
        const float s = 0.5 / sqrt(trace + 1.0);
        q = float4((m[2][1] - m[1][2]) * s, (m[0][2] - m[2][0]) * s, (m[1][0] - m[0][1]) * s, 0.25 / s);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const float s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        q = float4(0.25 * s, (m[1][0] + m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s);
    } else if (m[1][1] > m[2][2]) {
        const float s = 2.0 * sqrt(1.0 - m[0][0] + m[1][1] - m[2][2]);
        q = float4((m[1][0] + m[0][1]) / s, 0.25 * s, (m[2][1] + m[1][2]) / s, (m[0][2] - m[2][0]) / s);
    } else {
        const float s = 2.0 * sqrt(1.0 - m[0][0] - m[1][1] + m[2][2]);
        q = float4((m[0][2] + m[2][0]) / s, (m[2][1] + m[1][2]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s);
    }
    return normalize(q);
}

/// The other way: the three axes a quaternion turns the basis into, as the
/// columns of a matrix.
public float3x3 axesOfQuaternion(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
                    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
                    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
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

/// A texel written as the texture's own bytes, for a device whose store into
/// a texture does not convert (CUDA: surf2Dwrite writes the float4's bytes).
/// `kind`: 1 four 8-bit unorms in one word, 2 four halves in two, 3 four
/// floats in four. The words are the texture's rows packed, row 0 first,
/// which is what copyBufferToTexture takes.
public void lrtPackTexel(RWStructuredBuffer<uint> packed, uint kind, uint index, float4 texel) {
    if (kind == 1) {
        const uint4 bytes = uint4(round(saturate(texel) * 255.0));
        packed[index] = bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
    } else if (kind == 2) {
        packed[index * 2] = (f32tof16(texel.x) & 0xFFFFu) | ((f32tof16(texel.y) & 0xFFFFu) << 16);
        packed[index * 2 + 1] = (f32tof16(texel.z) & 0xFFFFu) | ((f32tof16(texel.w) & 0xFFFFu) << 16);
    } else {
        packed[index * 4] = asuint(texel.x);
        packed[index * 4 + 1] = asuint(texel.y);
        packed[index * 4 + 2] = asuint(texel.z);
        packed[index * 4 + 3] = asuint(texel.w);
    }
}
