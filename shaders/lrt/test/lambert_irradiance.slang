// Copyright (c) 2026 lucabRTrender contributors.
//
// A Lambert plane's rendered colour against the irradiance the light
// analytically puts on it. Nothing here is shared with the renderer: the
// light is given as geometry and an emitted radiance, and the irradiance is
// closed form -- a sphere's from its subtended cone, a polygon's from
// Lambert's formula over its edges, a disk's as a polygon of `vertices`
// sides, which converges to the disk far below the tolerance.
module lambert_irradiance;
import lrt.technique.camera;

static const uint kSphere  = 0;
static const uint kDisk    = 1;
static const uint kRect    = 2;
static const uint kDistant = 3;
static const uint kPoint   = 4;
static const uint kDome    = 5;
static const uint kCylinder = 6;   // axisX: its axis, w the radius; axisY.w the length
static const uint kCylinderIntegral = 7;   // the same, by dense quadrature of the form-factor integral

struct IrradianceParams {
    uint   kind;
    uint   vertices;       // the disk's polygon
    uint   pad0; uint pad1;
    float4 row0;           // view to world, rows of a 3x4
    float4 row1;
    float4 row2;
    float4 centre;         // xyz: the light's centre
    float4 axisX;          // xyz: its x axis; w: sizeX (radius, or the rect's width)
    float4 axisY;          // xyz: its y axis; w: sizeY (the rect's height)
    float4 normal;         // xyz: the plane's world normal; w: the surface's albedo
    float4 radiance;       // xyz: what the light emits; w: the relative tolerance
    float4 occluder;       // xyz: the centre of a square occluder; w: its half size, 0 for none
    float4 occluderAxes;   // x: ignore pixels nearer than this view depth (the occluder's own)
};

StructuredBuffer<float4>         colour;
StructuredBuffer<float>          depth;
RWStructuredBuffer<uint>         counts;   // [covered, beyond the tolerance]
RWStructuredBuffer<float>        worst;    // [worst relative difference, then a probe pixel: expected, x, y, z]
ConstantBuffer<CameraParams>     camera;
ConstantBuffer<IrradianceParams> plane;

float3 toWorld(float3 v, float w) {
    return float3(dot(plane.row0.xyz, v) + plane.row0.w * w, dot(plane.row1.xyz, v) + plane.row1.w * w,
                  dot(plane.row2.xyz, v) + plane.row2.w * w);
}

/// The k-th vertex of the light's polygon, of `count` in all.
float3 vertexAt(uint k, uint count) {
    const float3 x = plane.axisX.xyz;
    const float3 y = plane.axisY.xyz;
    if (plane.kind == kRect) {
        const float w = plane.axisX.w * 0.5;
        const float h = plane.axisY.w * 0.5;
        const float sx = (k == 0 || k == 3) ? -1.0 : 1.0;
        const float sy = (k < 2) ? -1.0 : 1.0;
        return plane.centre.xyz + x * (sx * w) + y * (sy * h);
    }
    const float angle = 2.0 * 3.14159265358979 * float(k) / float(count);
    return plane.centre.xyz + (x * cos(angle) + y * sin(angle)) * plane.axisX.w;
}

/// Lambert's formula over the four edges of one planar strip of a cylinder:
/// the k-th of `count` about its axis, between the two ends. A strip whose
/// outward normal looks away from `p` is not seen at all (the cylinder is
/// convex), and the winding is taken by its magnitude.
float stripFactor(float3 p, float3 n, uint k, uint count) {
    const float3 axis = normalize(plane.axisX.xyz);
    const float3 ay = normalize(plane.axisY.xyz);
    const float3 az = cross(axis, ay);
    const float r = plane.axisX.w;
    const float halfLength = 0.5 * plane.axisY.w;
    const float phi = 2.0 * 3.14159265358979 * (float(k) + 0.5) / float(count);
    const float delta = 3.14159265358979 / float(count);
    const float3 normal = ay * cos(phi) + az * sin(phi);
    const float3 centre = plane.centre.xyz + normal * r;
    if (dot(normal, p - centre) <= 0.0) {
        return 0.0;
    }
    const float3 r0 = ay * cos(phi - delta) + az * sin(phi - delta);
    const float3 r1 = ay * cos(phi + delta) + az * sin(phi + delta);
    float3 v[4];
    v[0] = plane.centre.xyz - axis * halfLength + r0 * r;
    v[1] = plane.centre.xyz + axis * halfLength + r0 * r;
    v[2] = plane.centre.xyz + axis * halfLength + r1 * r;
    v[3] = plane.centre.xyz - axis * halfLength + r1 * r;
    float sum = 0.0;
    float3 previous = normalize(v[0] - p);
    for (uint e = 1; e <= 4; ++e) {
        const float3 current = normalize(v[e % 4] - p);
        const float3 edgeAxis = cross(previous, current);
        const float length2 = dot(edgeAxis, edgeAxis);
        if (length2 > 1.0e-20) {
            const float theta = acos(clamp(dot(previous, current), -1.0, 1.0));
            sum += theta * dot(n, edgeAxis / sqrt(length2));
        }
        previous = current;
    }
    return abs(sum) * 0.5;
}

/// The defining integral, done numerically: over the lateral surface, the
/// receiver's cosine times the emitter's cosine over the distance squared,
/// where the emitter faces the point. Shares nothing with Lambert's edge
/// formula, so the two are each other's check. `vertices` cells about the
/// axis, and as many along it.
float cylinderIntegral(float3 p, float3 n, uint cells) {
    const float3 axis = normalize(plane.axisX.xyz);
    const float3 ay = normalize(plane.axisY.xyz);
    const float3 az = cross(axis, ay);
    const float r = plane.axisX.w;
    const float length = plane.axisY.w;
    const float dPhi = 2.0 * 3.14159265358979 / float(cells);
    const float dX = length / float(cells);
    const float dA = r * dPhi * dX;
    float sum = 0.0;
    for (uint i = 0; i < cells; ++i) {
        const float phi = (float(i) + 0.5) * dPhi;
        const float3 normal = ay * cos(phi) + az * sin(phi);
        for (uint j = 0; j < cells; ++j) {
            const float x = -0.5 * length + (float(j) + 0.5) * dX;
            const float3 q = plane.centre.xyz + axis * x + normal * r;
            const float3 d = q - p;
            const float d2 = dot(d, d);
            const float3 w = d / sqrt(d2);
            const float cosP = dot(n, w);
            const float cosQ = dot(normal, -w);
            if (cosP <= 0.0 || cosQ <= 0.0) {
                continue;
            }
            sum += cosP * cosQ / d2 * dA;
        }
    }
    return sum;
}

/// Lambert's formula: the irradiance a polygon of unit radiance puts on a
/// point, summed over its edges.
float polygonFactor(float3 p, float3 n, uint count) {
    float sum = 0.0;
    float3 previous = normalize(vertexAt(0, count) - p);
    for (uint k = 1; k <= count; ++k) {
        const float3 current = normalize(vertexAt(k % count, count) - p);
        const float3 axis = cross(previous, current);
        const float length2 = dot(axis, axis);
        if (length2 > 1.0e-20) {
            const float theta = acos(clamp(dot(previous, current), -1.0, 1.0));
            sum += theta * dot(n, axis / sqrt(length2));
        }
        previous = current;
    }
    return 0.5 * abs(sum);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lambertIrradiance(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint covered = 0;
    uint beyond = 0;
    float worstRelative = 0.0;
    const float3 n = normalize(plane.normal.xyz);
    const float albedo = plane.normal.w;
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            const uint i = y * camera.width + x;
            if (!(depth[i] > 0.0) || depth[i] < plane.occluderAxes.x) {
                continue;   // nothing drawn, or something in front of the plane
            }
            ++covered;
            float3 origin;
            float3 direction;
            viewRay(camera, float2(x, y) + 0.5, origin, direction);
            const float3 p = toWorld(origin + direction * depth[i], 1.0);
            // A square occluder on the light's axis, and a light small enough
            // to be a point: the umbra is exactly what it projects.
            if (plane.occluder.w > 0.0) {
                const float3 toLight = plane.centre.xyz - p;
                const float t = (plane.occluder.z - p.z) / max(abs(toLight.z), 1.0e-9) * sign(toLight.z);
                const float3 at = p + toLight * t;
                if (abs(at.x - plane.occluder.x) <= plane.occluder.w &&
                    abs(at.y - plane.occluder.y) <= plane.occluder.w && t > 0.0 && t < 1.0) {
                    const float3 got = colour[i].rgb;
                    const float dark = max(max(got.r, got.g), got.b);
                    worstRelative = max(worstRelative, dark);
                    beyond += dark > plane.radiance.w ? 1u : 0u;
                    continue;
                }
            }
            float factor = 0.0;
            if (plane.kind == kDome) {
                // A hemisphere of it, unoccluded: the cosine integrates to pi
                // whatever way the surface faces.
                factor = 3.14159265358979;
            } else if (plane.kind == kDistant) {
                // A disc of angular diameter sizeX around the light's axis,
                // of radiance intensity / solid angle so that a surface facing
                // it takes the intensity as irradiance. A spherical cap of
                // half-angle a has vector irradiance pi sin^2(a) along its
                // axis, so the plane takes that over the cap's solid angle,
                // times the cosine to the axis.
                const float3 towards = normalize(plane.axisY.xyz);
                const float half = 0.5 * plane.axisX.w;
                const float solidAngle = 2.0 * 3.14159265358979 * (1.0 - cos(half));
                const float cap = 3.14159265358979 * sin(half) * sin(half);
                factor = max(dot(n, towards), 0.0) * (plane.axisX.w > 0.0 ? cap / solidAngle : 1.0);
            } else if (plane.kind == kPoint) {
                const float3 offset = plane.centre.xyz - p;
                const float distance2 = dot(offset, offset);
                factor = max(dot(n, offset / sqrt(distance2)), 0.0) / distance2;
            } else if (plane.kind == kSphere) {
                // A sphere fully above the horizon: its cone, times the
                // cosine to its centre.
                const float3 offset = plane.centre.xyz - p;
                const float distance2 = dot(offset, offset);
                const float sin2 = min(plane.axisX.w * plane.axisX.w / distance2, 1.0);
                const float cosine = max(dot(n, offset / sqrt(distance2)), 0.0);
                factor = 3.14159265358979 * sin2 * cosine;
            } else if (plane.kind == kCylinderIntegral) {
                // 128 cells a side agrees with the 512-strip edge formula to
                // 0.2%; 512 took 18 s a frame for nothing.
                factor = cylinderIntegral(p, n, clamp(plane.vertices, 8u, 128u));
            } else if (plane.kind == kCylinder) {
                // The lateral surface as strips, each a rectangle seen from
                // its outside only; enough of them and the polygon is the
                // cylinder to far below the tolerance.
                const uint strips = max(plane.vertices, 8u);
                for (uint k = 0; k < strips; ++k) {
                    factor += stripFactor(p, n, k, strips);
                }
            } else {
                factor = polygonFactor(p, n, plane.kind == kRect ? 4u : max(plane.vertices, 3u));
            }
            // Lambert: what leaves is the albedo over pi times what arrives.
            const float3 expected = plane.radiance.xyz * factor * albedo / 3.14159265358979;
            const float3 got = colour[i].rgb;
            const float scale = max(max(expected.r, expected.g), max(expected.b, 1.0e-6));
            const float relative = max(max(abs(got.r - expected.r), abs(got.g - expected.g)),
                                       abs(got.b - expected.b)) / scale;
            worstRelative = max(worstRelative, relative);
            beyond += relative > plane.radiance.w ? 1u : 0u;
            // One pixel reported in full, to see what a difference is made of.
            if (x == camera.width / 2 + 20 && y == camera.height / 2) {
                worst[1] = expected.r;
                worst[2] = p.x;
                worst[3] = p.y;
                worst[4] = p.z;
            }
        }
    }
    counts[0] = covered;
    counts[1] = beyond;
    worst[0] = worstRelative;
}
