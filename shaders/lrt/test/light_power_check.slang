// Copyright (c) 2026 lucabRTrender contributors.
//
// Each light's share of the frame's power, as the prefix accumulated it,
// against the shares written again here from the records: radiance over area
// for an area light, over the scene's cross-section for a dome (pi R^2) and a
// sun (R^2). A count of lights whose share differs, and the worst ratio.
import lrt.light.lights;

struct PowerCheck {
    uint  count;
    float sceneRadius;
    float tolerance;   // relative
    uint  pad0;
};

StructuredBuffer<LightRecord> lights;
ConstantBuffer<PowerCheck>    check;
RWStructuredBuffer<uint>      counts;   // [shares off, lights]
RWStructuredBuffer<float>     worst;    // [largest relative difference]

float expectedPower(LightRecord l) {
    const float luminance = max(0.2126 * l.colour.x + 0.7152 * l.colour.y + 0.0722 * l.colour.z, 0.0) * exp2(l.exposure);
    if ((l.flags & kLightNormalize) != 0) {
        return max(luminance, 1.0e-6);
    }
    const float pi = 3.14159265358979;
    float extent = 1.0;
    if (l.kind == kLightSphere) extent = 4.0 * pi * l.sizeX * l.sizeX;
    if (l.kind == kLightDisk) extent = pi * l.sizeX * l.sizeX;
    if (l.kind == kLightRect) extent = l.sizeX * l.sizeY;
    if (l.kind == kLightCylinder) extent = 2.0 * pi * l.sizeX * l.sizeY;
    if (l.kind == kLightDome) extent = pi * check.sceneRadius * check.sceneRadius;
    if (l.kind == kLightDistant) extent = check.sceneRadius * check.sceneRadius;
    return max(luminance * extent, 1.0e-6);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lightPowerCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    float total = 0.0;
    for (uint k = 0; k < check.count; ++k) {
        total += expectedPower(lights[k]);
    }
    const float accumulated = lights[check.count - 1].cumulative;
    uint off = 0;
    float largest = 0.0;
    for (uint k = 0; k < check.count; ++k) {
        const float share = (lights[k].cumulative - (k > 0 ? lights[k - 1].cumulative : 0.0)) / accumulated;
        const float expected = expectedPower(lights[k]) / total;
        const float difference = abs(share - expected) / max(expected, 1e-12);
        largest = max(largest, difference);
        if (difference > check.tolerance) off += 1;
    }
    counts[0] = off;
    counts[1] = check.count;
    worst[0] = largest;
}
