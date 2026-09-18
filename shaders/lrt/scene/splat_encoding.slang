// Copyright (c) 2026 lucabRTrender contributors.
//
// Where a raw record's fields are and how they are encoded. Mirrors
// lrt::io::SplatEncoding and lrt::scene's DecodeUniforms field for field.
module splat_encoding;
import lrt.common.color;
import lrt.common.packing;

public struct DecodeParams {
    public uint count;            // records in this slice
    public uint stride;           // floats per record
    public uint base;             // first output index for this slice
    public uint keepPerColour;    // rest basis functions kept (0, 3, 8, 15)
    public uint x; public uint y; public uint z; public uint opacity;
    public uint scale0; public uint scale1; public uint scale2; public uint rotW;
    public uint rotX; public uint rotY; public uint rotZ; public uint dc0;
    public uint dc1; public uint dc2; public uint restBase; public uint filePerColour;
    public uint restColourOuter; public uint opacityMode; public uint scaleMode; public uint colourMode;
    public uint rotationMode; public uint shWords; public uint restMode; public uint flipYZ;
    // What a gaussian reflects with, where the file carries it: 0xFFFFFFFF
    // means it does not, which is every file a trainer writes.
    public float positionScale; public uint metallic; public uint roughness; public uint transmission;
};

public float fieldOf(StructuredBuffer<float> raw, uint record, uint stride, uint field) {
    return raw[record * stride + field];
}

public float decodeOpacity(float v, uint mode) {
    if (mode == 0) return 1.0 / (1.0 + exp(-v));   // logit
    if (mode == 2) return v / 255.0;               // byte
    return v;
}

/// Position `field` 0..2 (x, y, z) of a record.
public float3 decodePosition(StructuredBuffer<float> raw, uint record, DecodeParams p) {
    float3 v = float3(fieldOf(raw, record, p.stride, p.x), fieldOf(raw, record, p.stride, p.y),
                      fieldOf(raw, record, p.stride, p.z)) * p.positionScale;
    // Right-up-back to right-down-front: a half turn about x.
    return p.flipYZ != 0 ? float3(v.x, -v.y, -v.z) : v;
}

public float decodeLogScale(float v, uint mode) {
    if (mode == 2) return v / 16.0 - 10.0;         // SPZ byte
    return mode == 0 ? v : log(max(v, 1e-30));
}

public float decodeColour(float v, uint mode) {
    if (mode == 0) return 0.5 + kSH0 * v;          // SH DC coefficient
    if (mode == 2) return v / 255.0;
    if (mode == 3) return 0.5 + kSH0 * ((v / 255.0 - 0.5) / 0.15);   // SPZ byte: its DC scale
    if (mode == 4) return linearToSrgb(v);         // light, into the space a cloud is blended in
    return v;
}

/// A rest coefficient.
public float decodeRest(float v, uint mode) {
    return mode == 1 ? (v - 128.0) / 128.0 : v;
}

/// The sign a rest coefficient takes when y and z change sign, per basis in
/// 3DGS order (degree 1: y z x; 2: xy yz 3z^2-1 xz x^2-y^2; 3: ...): odd in
/// y xor z flips. Niantic's spz computes the same table (splat-types.h).
public float restFlipSign(uint basis) {
    static const float kSigns[15] = { -1.0, -1.0, 1.0,
                                      -1.0, 1.0, 1.0, -1.0, 1.0,
                                      -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0 };
    return kSigns[min(basis, 14u)];
}

public float decodeRotation(float v, uint mode) {
    return mode == 1 ? (v - 128.0) / 128.0 : v;
}

/// A record's rotation as (x, y, z, w), not yet normalised.
public float4 decodeQuaternion(StructuredBuffer<float> raw, uint record, DecodeParams p) {
    const uint s = p.stride;
    float4 q;
    if (p.rotationMode == 2) {
        // SPZ v2: x y z bytes, w from unit length.
        const float3 xyz = float3(fieldOf(raw, record, s, p.rotX), fieldOf(raw, record, s, p.rotY),
                                  fieldOf(raw, record, s, p.rotZ)) / 127.5 - 1.0;
        q = float4(xyz, sqrt(max(0.0, 1.0 - dot(xyz, xyz))));
    } else if (p.rotationMode == 3) {
        // SPZ v3+: smallest three, the uint32 in two 16-bit halves.
        uint comp = uint(fieldOf(raw, record, s, p.rotX)) | (uint(fieldOf(raw, record, s, p.rotY)) << 16);
        const uint largest = comp >> 30;
        q = float4(0.0);
        float sum = 0.0;
        for (int i = 3; i >= 0; --i) {
            if (uint(i) == largest) {
                continue;
            }
            const float magnitude = kSqrtHalf * float(comp & 511u) / 511.0;
            q[i] = ((comp >> 9) & 1u) != 0 ? -magnitude : magnitude;
            sum += q[i] * q[i];
            comp = comp >> 10;
        }
        q[largest] = sqrt(max(0.0, 1.0 - sum));
    } else {
        // Stored as (w, x, y, z).
        q = float4(decodeRotation(fieldOf(raw, record, s, p.rotX), p.rotationMode),
                   decodeRotation(fieldOf(raw, record, s, p.rotY), p.rotationMode),
                   decodeRotation(fieldOf(raw, record, s, p.rotZ), p.rotationMode),
                   decodeRotation(fieldOf(raw, record, s, p.rotW), p.rotationMode));
    }
    // A half turn about x: conjugating by it negates the y and z parts.
    return p.flipYZ != 0 ? float4(q.x, -q.y, -q.z, q.w) : q;
}

public static const uint kNoField = 0xFFFFFFFFu;

/// What a splat reflects with: metallic, roughness and transmission, a byte
/// each and one spare. Nothing is decoded here -- all three are already 0 to 1.
public uint packPbr(float metallic, float roughness, float transmission) {
    const uint m = uint(saturate(metallic) * 255.0 + 0.5);
    const uint r = uint(saturate(roughness) * 255.0 + 0.5);
    const uint t = uint(saturate(transmission) * 255.0 + 0.5);
    return m | (r << 8) | (t << 16);
}

public float pbrMetallic(uint word) { return float(word & 0xFFu) / 255.0; }
public float pbrRoughness(uint word) { return float((word >> 8) & 0xFFu) / 255.0; }
public float pbrTransmission(uint word) { return float((word >> 16) & 0xFFu) / 255.0; }
