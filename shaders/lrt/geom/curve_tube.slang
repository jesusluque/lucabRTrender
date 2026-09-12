// Copyright (c) 2026 lucabRTrender contributors.
//
// Basis curves as tubes: every span of a curve is evaluated at a few
// parameters (the Bezier, B-spline, Catmull-Rom or linear form of its
// control points), and at each a ring of `sides` vertices is laid at half
// the width around the centre line, in a frame taken from the tangent. The
// host lays out the rings' topology (a bookkeeping pattern); the positions
// are this kernel's, one thread a vertex.
module curve_tube;

struct CurveParams {
    uint  vertices;       // tube vertices in all
    uint  sides;
    uint  segments;       // sub-segments a span: rings a span = segments + 1
    uint  basis;          // 0 linear, 1 bezier, 2 bspline, 3 catmullRom
    uint  widthMode;      // 0 constant, 1 uniform (a curve), 2 vertex (a control point), 3 varying (a span end)
    float width;          // widthMode 0
    uint  pad0; uint pad1;
};

struct SpanRecord {
    uint  first;          // its first control point, in `points`
    uint  curve;          // the curve it belongs to
    uint  count;          // control points: 2 linear, 4 cubic
    uint  varyingFirst;   // its first varying width (mode 3)
    uint  wrap[4];        // control point indices, wrapped for a periodic curve (count of them)
};

StructuredBuffer<float4>     points;      // control points
StructuredBuffer<SpanRecord> spans;
StructuredBuffer<float>      widths;      // by widthMode
RWStructuredBuffer<float4>   positions;   // tube vertices

static const float kPi = 3.14159265358979;

void weights(uint basis, float t, out float4 b, out float4 db) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    if (basis == 1) {          // Bezier
        const float s = 1.0 - t;
        b = float4(s * s * s, 3.0 * t * s * s, 3.0 * t2 * s, t3);
        db = float4(-3.0 * s * s, 3.0 * s * s - 6.0 * t * s, 6.0 * t * s - 3.0 * t2, 3.0 * t2);
    } else if (basis == 2) {   // B-spline
        b = float4((1.0 - t) * (1.0 - t) * (1.0 - t), 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
        db = float4(-3.0 * (1.0 - t) * (1.0 - t), 9.0 * t2 - 12.0 * t, -9.0 * t2 + 6.0 * t + 3.0, 3.0 * t2) / 6.0;
    } else if (basis == 3) {   // Catmull-Rom
        b = float4(-t3 + 2.0 * t2 - t, 3.0 * t3 - 5.0 * t2 + 2.0, -3.0 * t3 + 4.0 * t2 + t, t3 - t2) * 0.5;
        db = float4(-3.0 * t2 + 4.0 * t - 1.0, 9.0 * t2 - 10.0 * t, -9.0 * t2 + 8.0 * t + 1.0, 3.0 * t2 - 2.0 * t) * 0.5;
    } else {                   // linear
        b = float4(1.0 - t, t, 0.0, 0.0);
        db = float4(-1.0, 1.0, 0.0, 0.0);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void curveTube(uint3 tid: SV_DispatchThreadID) {
    const uint v = tid.x;
    if (v >= params.vertices) {
        return;
    }
    const uint ringsPerSpan = params.segments + 1;
    const uint perSpan = ringsPerSpan * params.sides;
    const uint s = v / perSpan;
    const uint within = v - s * perSpan;
    const uint ring = within / params.sides;
    const uint side = within - ring * params.sides;
    const SpanRecord span = spans[s];
    const float t = float(ring) / float(params.segments);
    float4 b, db;
    weights(params.basis, t, b, db);
    float3 p = float3(0.0);
    float3 tangent = float3(0.0);
    float width = params.width;
    if (params.widthMode == 1) {
        width = widths[span.curve];
    } else if (params.widthMode == 2) {
        width = 0.0;
    } else if (params.widthMode == 3) {
        width = lerp(widths[span.varyingFirst], widths[span.varyingFirst + 1], t);
    }
    for (uint k = 0; k < span.count; ++k) {
        const uint index = span.wrap[k];
        p += points[index].xyz * b[k];
        tangent += points[index].xyz * db[k];
        if (params.widthMode == 2) {
            width += widths[index] * b[k];
        }
    }
    // A frame about the tangent: the axis least along it, made perpendicular.
    const float3 a = normalize(tangent);
    const float3 axis = abs(a.x) < 0.5 ? float3(1.0, 0.0, 0.0) : (abs(a.y) < 0.5 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0));
    const float3 n1 = normalize(cross(a, axis));
    const float3 n2 = cross(a, n1);
    const float angle = 2.0 * kPi * float(side) / float(params.sides);
    positions[v] = float4(p + 0.5 * width * (cos(angle) * n1 + sin(angle) * n2), 1.0);
}

ConstantBuffer<CurveParams> params;
