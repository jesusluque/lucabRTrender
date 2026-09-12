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
module light_prefix;
import lrt.light.lights;

RWStructuredBuffer<LightRecord> records;
uniform uint                    count;

static const float3 kPrefixLuminance = float3(0.2126, 0.7152, 0.0722);

float powerOf(LightRecord l) {
    float power = max(dot(l.colour, kPrefixLuminance), 0.0) * exp2(l.exposure);
    if ((l.flags & kLightNormalize) == 0) {
        switch (l.kind) {
        case kLightSphere:   power *= 4.0 * 3.14159265358979 * l.sizeX * l.sizeX; break;
        case kLightDisk:     power *= 3.14159265358979 * l.sizeX * l.sizeX; break;
        case kLightRect:     power *= l.sizeX * l.sizeY; break;
        case kLightCylinder: power *= 2.0 * 3.14159265358979 * l.sizeX * l.sizeY; break;
        default: break;   // distant and dome: over the whole sky either way
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
