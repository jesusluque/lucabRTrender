// Copyright (c) 2026 lucabRTrender contributors.
//
// A curve tube checked against the curve: every tube vertex must sit at
// half the width from the point of the curve it rings, that point being the
// Bezier, B-spline, Catmull-Rom or linear form of the span's control points
// written a second time here (the Bernstein and the polynomial forms
// straight, not through a basis matrix).
module curve_check;

struct CurveCheckParams {
    uint  vertices;
    uint  sides;
    uint  segments;
    uint  basis;        // 0 linear, 1 bezier, 2 bspline, 3 catmullRom
    float width;        // constant
    float tolerance;
    uint  pad0; uint pad1;
};

struct SpanRecord {
    uint  first;
    uint  curve;
    uint  count;
    uint  varyingFirst;
    uint  wrap[4];
};

StructuredBuffer<float4>     points;
StructuredBuffer<SpanRecord> spans;
StructuredBuffer<float4>     tube;
StructuredBuffer<float4>     tangents;   // the tube's tangent primvar
RWStructuredBuffer<uint>     counts;     // [checked, beyond, tangents off]
RWStructuredBuffer<float>    worst;
ConstantBuffer<CurveCheckParams> check;

float3 curveAt(uint s, float t) {
    const SpanRecord span = spans[s];
    const float3 p0 = points[span.wrap[0]].xyz;
    const float3 p1 = points[span.wrap[1]].xyz;
    const float3 p2 = points[span.wrap[2]].xyz;
    const float3 p3 = points[span.wrap[3]].xyz;
    const float u = 1.0 - t;
    if (check.basis == 1) {
        return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3;
    }
    if (check.basis == 2) {
        // De Boor's uniform cubic B-spline segment, as a sum of the four polynomials.
        return ((u * u * u) * p0 + (3.0 * t * t * t - 6.0 * t * t + 4.0) * p1 +
                (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) * p2 + (t * t * t) * p3) / 6.0;
    }
    if (check.basis == 3) {
        // Catmull-Rom: the Hermite form with tangents (p2 - p0) / 2 and (p3 - p1) / 2.
        const float3 m0 = 0.5 * (p2 - p0);
        const float3 m1 = 0.5 * (p3 - p1);
        const float h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
        const float h10 = t * t * t - 2.0 * t * t + t;
        const float h01 = -2.0 * t * t * t + 3.0 * t * t;
        const float h11 = t * t * t - t * t;
        return h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1;
    }
    return lerp(p0, p1, t);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void curveCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint beyond = 0;
    uint tangentsOff = 0;
    float largest = 0.0;
    const uint perSpan = (check.segments + 1) * check.sides;
    for (uint v = 0; v < check.vertices; ++v) {
        const uint s = v / perSpan;
        const uint ring = (v - s * perSpan) / check.sides;
        const float t = float(ring) / float(check.segments);
        const float3 centre = curveAt(s, t);
        const float gap = abs(length(tube[v].xyz - centre) - 0.5 * check.width);
        largest = max(largest, gap);
        beyond += gap > check.tolerance ? 1u : 0u;
        // The tangent: unit, perpendicular to the ring's radius, and along
        // the curve's own difference quotient about t (a step either way).
        const float3 tangent = tangents[v].xyz;
        const float step = 1.0e-3;
        const float3 along = normalize(curveAt(s, min(t + step, 1.0)) - curveAt(s, max(t - step, 0.0)));
        const float3 radius = tube[v].xyz - centre;
        const bool unit = abs(length(tangent) - 1.0) < 1.0e-4;
        const bool perpendicular = abs(dot(tangent, radius)) < 1.0e-4 * max(length(radius), 1.0e-6);
        const bool aligned = dot(tangent, along) > 0.9999;
        tangentsOff += (unit && perpendicular && aligned) ? 0u : 1u;
    }
    counts[0] = check.vertices;
    counts[1] = beyond;
    counts[2] = tangentsOff;
    worst[0] = largest;
}
