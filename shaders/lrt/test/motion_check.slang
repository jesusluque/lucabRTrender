// Copyright (c) 2026 lucabRTrender contributors.
//
// Motion blur checked against its own closed form. A uniformly lit plane
// ending in a straight vertical edge slides along x over the shutter, so a
// pixel's coverage is the fraction of the shutter during which the edge is
// past it: a ramp from the edge's start column to its end. The tracer cuts
// the shutter into buckets at whose centres it places the plane, so what it
// must read is the ramp quantised to those centres -- a staircase of
// `buckets` steps, which is what this evaluates -- and with more buckets the
// staircase is the ramp.
module motion_check;

struct MotionParams {
    uint  width; uint height;
    float edge0;      // the edge's column at the shutter's open
    float edge1;      // and at its close
    uint  buckets;
    float plateau;    // a column always inside the lit side
    float tolerance;  // absolute, relative to the plateau
    float pad0;
};

StructuredBuffer<float4>       image;
RWStructuredBuffer<uint>       counts;   // [checked, beyond]
RWStructuredBuffer<float>      worst;
ConstantBuffer<MotionParams>   motion;

[shader("compute")]
[numthreads(1, 1, 1)]
void motionEdge(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint checked = 0;
    uint beyond = 0;
    float largest = 0.0;
    const float lo = min(motion.edge0, motion.edge1) - 2.0;
    const float hi = max(motion.edge0, motion.edge1) + 2.0;
    for (uint y = 0; y < motion.height; ++y) {
        const float lit = image[y * motion.width + uint(motion.plateau)].r;
        for (uint x = 0; x < motion.width; ++x) {
            const float centre = float(x) + 0.5;
            if (centre < lo || centre > hi) {
                continue;
            }
            uint inside = 0;
            for (uint b = 0; b < motion.buckets; ++b) {
                const float t = (float(b) + 0.5) / float(motion.buckets);
                const float edge = lerp(motion.edge0, motion.edge1, t);
                inside += edge > centre ? 1u : 0u;
            }
            const float expected = lit * float(inside) / float(motion.buckets);
            const float got = image[y * motion.width + x].r;
            const float gap = abs(got - expected) / max(lit, 1.0e-6);
            checked += 1;
            beyond += gap > motion.tolerance ? 1u : 0u;
            largest = max(largest, gap);
        }
    }
    counts[0] = checked;
    counts[1] = beyond;
    worst[0] = largest;
}
