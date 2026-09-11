// Copyright (c) 2026 lucabRTrender contributors.
//
// Where a raw record's fields are and how they are encoded. Mirrors
// lrt::io::SplatEncoding and lrt::scene's DecodeUniforms field for field.
module splat_encoding;
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
    public uint rotationMode; public uint shWords; public uint pad0; public uint pad1;
};

public float fieldOf(StructuredBuffer<float> raw, uint record, uint stride, uint field) {
    return raw[record * stride + field];
}

public float decodeOpacity(float v, uint mode) {
    if (mode == 0) return 1.0 / (1.0 + exp(-v));   // logit
    if (mode == 2) return v / 255.0;               // byte
    return v;
}

public float decodeLogScale(float v, uint mode) {
    return mode == 0 ? v : log(max(v, 1e-30));
}

public float decodeColour(float v, uint mode) {
    if (mode == 0) return 0.5 + kSH0 * v;          // SH DC coefficient
    if (mode == 2) return v / 255.0;
    return v;
}

public float decodeRotation(float v, uint mode) {
    return mode == 1 ? (v - 128.0) / 128.0 : v;
}
