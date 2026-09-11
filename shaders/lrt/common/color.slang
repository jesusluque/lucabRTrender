// Copyright (c) 2026 lucabRTrender contributors.
module color;

/// The sRGB curve, extended past 1: a trained cloud's colours are not bounded
/// to the display, and clamping here clips every highlight the scene has
/// (a sky at 1.6 came out 0.93). Negative input is taken as zero.
public float srgbToLinear(float c) {
    c = max(c, 0.0);
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

public float3 srgbToLinear(float3 c) {
    return float3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z));
}

public float hash01(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0;
}
