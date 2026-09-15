// Copyright (c) 2026 lucabRTrender contributors.
//
// Each light's share of the frame's power, accumulated in order, so a light
// can be chosen in proportion to it. One thread walks the records: a frame
// has hundreds of lights at most until a light BVH arrives, and a walk in
// order is exact. The total is the last record's cumulative, which is where
// chooseLight reads it -- nothing comes back to the host.
//
// A rough estimate is all a choice needs: it only has to be positive and
// roughly proportional, since the density it implies is divided back out.
// Roughly proportional across kinds, though: an area light's is its radiance
// over its area (the pi of a Lambertian emitter is common to all and left
// out), and a dome's or a sun's has to be in the same units, or a room lit by
// a dome and a window chose the dome once in four hundred thousand samples
// -- unbiased, and dark and blotchy at any count a frame takes. A dome of
// radiance L lights a scene of radius R with pi L pi R^2, a sun of irradiance
// E with E pi R^2; over the same pi, L pi R^2 and E R^2.
module light_prefix;
import lrt.light.lights;

RWStructuredBuffer<LightRecord> records;
uniform uint                    count;
uniform float                   sceneRadius;   // the scene's reach, in its units

static const float3 kPrefixLuminance = float3(0.2126, 0.7152, 0.0722);

float powerOf(LightRecord l) {
    float power = max(dot(l.colour, kPrefixLuminance), 0.0) * exp2(l.exposure);
    if ((l.flags & kLightNormalize) == 0) {
        switch (l.kind) {
        case kLightSphere:   power *= 4.0 * 3.14159265358979 * l.sizeX * l.sizeX; break;
        case kLightDisk:     power *= 3.14159265358979 * l.sizeX * l.sizeX; break;
        case kLightRect:     power *= l.sizeX * l.sizeY; break;
        case kLightCylinder: power *= 2.0 * 3.14159265358979 * l.sizeX * l.sizeY; break;
        case kLightDome:     power *= 3.14159265358979 * sceneRadius * sceneRadius; break;
        case kLightDistant:  power *= sceneRadius * sceneRadius; break;
        default: break;
        }
    }
    return max(power, 1.0e-6);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lightPrefix(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float total = 0.0;
    for (uint k = 0; k < count; ++k) {
        total += powerOf(records[k]);
        records[k].cumulative = total;
    }
}
