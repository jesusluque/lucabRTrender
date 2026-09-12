// Copyright (c) 2026 lucabRTrender contributors.
//
// The camera's lens, checked against closed forms that share nothing with
// the tracer. A straight edge of a uniformly lit plane, out of focus, is
// blurred by a uniform disk -- the lens disk projected -- so a pixel at
// signed distance d from the edge reads the fraction of that disk on the lit
// side: a circular segment. And a radial distortion moves the edge to where
// the distorted pixel ray meets the plane, row by row.
module dof_check;

struct DofParams {
    uint  width; uint height;
    float edge;      // the edge's column in the pinhole image, in pixels
    float radius;    // the circle of confusion, in pixels
    float plateau;   // a column well inside the lit side
    float tolerance; // absolute, relative to the plateau
    float band;      // columns within band * radius of the edge are checked
    float pad0;
};
struct DistortionParams {
    uint  width; uint height;
    float focalX; float focalY; float centreX; float centreY;
    float k1; float k2;
    float edgeX;     // x / z of the edge: lit where the ray's distorted x / z is below it
    float pad0; float pad1; float pad2;
};

StructuredBuffer<float4>          image;
RWStructuredBuffer<uint>          counts;
RWStructuredBuffer<float>         worst;
ConstantBuffer<DofParams>         dof;
ConstantBuffer<DistortionParams>  lens;

static const float kPi = 3.14159265358979;

/// The fraction of a disk of radius r whose x is below d from its centre.
float segment(float d, float r) {
    if (d >= r) return 1.0;
    if (d <= -r) return 0.0;
    return (r * r * acos(-d / r) + d * sqrt(r * r - d * d)) / (kPi * r * r);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void dofEdge(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint checked = 0;
    uint beyond = 0;
    float largest = 0.0;
    const int plateauColumn = int(dof.plateau);
    for (uint y = 0; y < dof.height; ++y) {
        const float lit = image[y * dof.width + uint(plateauColumn)].r;
        for (uint x = 0; x < dof.width; ++x) {
            const float d = dof.edge - (float(x) + 0.5);
            if (abs(d) > dof.band * dof.radius) {
                continue;
            }
            const float expected = lit * segment(d, dof.radius);
            const float got = image[y * dof.width + x].r;
            const float gap = abs(got - expected) / max(lit, 1.0e-6);
            checked += 1;
            beyond += gap > dof.tolerance ? 1u : 0u;
            largest = max(largest, gap);
        }
    }
    counts[0] = checked;
    counts[1] = beyond;
    worst[0] = largest;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void distortionEdge(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint rows = 0;
    uint offByMore = 0;
    uint offAtAll = 0;
    uint moved = 0;
    for (uint y = 0; y < lens.height; ++y) {
        const float plateau = image[y * lens.width].r;
        uint expected = 0;
        uint undistorted = 0;
        uint observed = 0;
        for (uint x = 0; x < lens.width; ++x) {
            const float2 p = float2((float(x) + 0.5 - lens.centreX) / lens.focalX,
                                    (float(y) + 0.5 - lens.centreY) / lens.focalY);
            const float2 ndc = float2(p.x * lens.focalX / (0.5 * float(lens.width)),
                                      p.y * lens.focalY / (0.5 * float(lens.height)));
            const float r2 = dot(ndc, ndc);
            const float scale = 1.0 + lens.k1 * r2 + lens.k2 * r2 * r2;
            expected += p.x * scale < lens.edgeX ? 1u : 0u;
            undistorted += p.x < lens.edgeX ? 1u : 0u;
            observed += image[y * lens.width + x].r > 0.5 * plateau ? 1u : 0u;
        }
        rows += 1;
        const uint gap = expected > observed ? expected - observed : observed - expected;
        offAtAll += gap != 0 ? 1u : 0u;
        offByMore += gap > 1 ? 1u : 0u;
        moved += expected != undistorted ? 1u : 0u;
    }
    counts[0] = rows;
    counts[1] = offByMore;
    counts[2] = offAtAll;
    counts[3] = moved;
}
