// Copyright (c) 2026 lucabRTrender contributors.
//
// IES profiles, checked as counters: every authored node returns its own
// candela, the interpolation between nodes follows a closed form to the
// bound piecewise-linear interpolation allows, and through a whole render a
// cutoff profile leaves the plane lit inside its cone exactly as it would be
// without one, and black outside.
module ies_check;
import lrt.light.lights;
import lrt.technique.camera;

struct IesCheckParams {
    uint  nodes;        // vertical nodes of the profile on light 0
    uint  samples;      // directions for the closed-form comparison
    float exponent;     // the closed form: 1000 * cos^exponent(theta)
    float bound;        // absolute, on that comparison
    float4 lightCentre; // xyz, world; w: the cutoff angle, degrees
    float4 row0;        // view to world
    float4 row1;
    float4 row2;
};

StructuredBuffer<LightRecord> lights;
uniform uint                  lightCount;
RWStructuredBuffer<uint>   counts;   // [nodes off, closed form off, inside differing, outside lit, inside, outside]
RWStructuredBuffer<float>  worst;    // [node error, closed-form error]
ConstantBuffer<IesCheckParams> check;
StructuredBuffer<float4>   with;     // rendered with the cutoff profile
StructuredBuffer<float4>   without;  // rendered without
StructuredBuffer<float>    depth;
ConstantBuffer<CameraParams> camera;

static const float kIesPi = 3.14159265358979;

/// A direction from a point to the light such that the light's emission
/// towards the point makes `theta` with its -Z axis, in its xz plane.
float3 towardsLightAt(LightRecord l, float theta) {
    const float3 e = lightAxisX(l) * sin(theta) - lightAxisZ(l) * cos(theta);   // emission, theta from -Z
    return -e;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void iesNodes(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const LightRecord l = lights[0];
    const IesRecord r = iesRecords[l.ies];
    uint off = 0;
    float largest = 0.0;
    for (uint i = 0; i < min(check.nodes, r.verticalCount); ++i) {
        const float thetaDeg = iesValues[r.verticalOffset + i];
        const float3 wi = towardsLightAt(l, radians(thetaDeg));
        const float got = iesIntensity(l, wi);
        const float expected = iesValues[r.candelaOffset + i] * r.multiplier;
        const float error = abs(got - expected) / max(expected, 1.0e-3);
        largest = max(largest, error);
        off += error > 1.0e-4 ? 1u : 0u;
    }
    counts[0] = off;
    worst[0] = largest;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void iesCos(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const LightRecord l = lights[0];
    uint off = 0;
    float largest = 0.0;
    for (uint k = 0; k < check.samples; ++k) {
        const float theta = (float(k) + 0.5) / float(check.samples) * kIesPi;
        const float3 wi = towardsLightAt(l, theta);
        const float got = iesIntensity(l, wi);
        const float expected = 1000.0 * pow(max(cos(theta), 0.0), check.exponent);
        const float error = abs(got - expected);
        largest = max(largest, error);
        off += error > check.bound ? 1u : 0u;
    }
    counts[1] = off;
    worst[1] = largest;
}

/// Through a render: inside the cutoff cone the profile is one, so the two
/// frames must match; outside it is zero, so the lit one must be black.
[shader("compute")]
[numthreads(1, 1, 1)]
void iesCutoff(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint insideDiffer = 0;
    uint outsideLit = 0;
    uint inside = 0;
    uint outside = 0;
    float largestOutside = 0.0;
    float farthestLit = 0.0;
    const float cutoff = radians(check.lightCentre.w);
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            const uint i = y * camera.width + x;
            if (!(depth[i] > 0.0)) {
                continue;
            }
            float3 origin;
            float3 direction;
            viewRay(camera, float2(x, y) + 0.5, origin, direction);
            const float3 v = origin + direction * depth[i];
            const float3 p = float3(dot(check.row0.xyz, v) + check.row0.w, dot(check.row1.xyz, v) + check.row1.w,
                                    dot(check.row2.xyz, v) + check.row2.w);
            // The light's emission towards p, against its -Z (world -z here).
            const float3 e = normalize(p - check.lightCentre.xyz);
            const float theta = acos(clamp(-e.z, -1.0, 1.0));
            // The profile authors one at 20 degrees and zero at 25, so the
            // shader's linear interpolation is a ramp between them; and a
            // sphere of radius r at distance d spreads any direction by
            // asin(r/d) -- 1.43 degrees here. So the frame is the unshaped
            // one below 20 - 1.43 and black above 25 + 1.43; the band between
            // is the ramp's, and skipped. (With a band of 3 degrees about 20
            // the check read 1626 lit pixels outside, all within 26.3 degrees
            // and all the ramp's.) `w` of lightCentre is the cutoff node.
            const float inner = cutoff - radians(2.0);
            const float outer = cutoff + radians(7.0);
            if (theta >= inner && theta <= outer) {
                continue;
            }
            const float4 a = with[i];
            const float4 b = without[i];
            if (theta < inner) {
                ++inside;
                insideDiffer += any(abs(a.rgb - b.rgb) > 1.0e-6) ? 1u : 0u;
            } else {
                ++outside;
                const bool lit = any(a.rgb > 1.0e-6);
                outsideLit += lit ? 1u : 0u;
                if (lit) {
                    largestOutside = max(largestOutside, max(a.r, max(a.g, a.b)));
                    farthestLit = max(farthestLit, theta);
                }
            }
        }
    }
    counts[2] = insideDiffer;
    counts[3] = outsideLit;
    counts[4] = inside;
    counts[5] = outside;
    worst[0] = largestOutside;
    worst[1] = degrees(farthestLit);
}
