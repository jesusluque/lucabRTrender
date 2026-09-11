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
            if (plane.kind == kDistant) {
                // A disc of angular diameter sizeX around the light's axis:
                // its solid angle times the cosine to it.
                const float3 towards = normalize(plane.axisY.xyz);
                const float solidAngle = 2.0 * 3.14159265358979 * (1.0 - cos(0.5 * plane.axisX.w));
                factor = max(dot(n, towards), 0.0) * (plane.axisX.w > 0.0 ? solidAngle : 1.0);
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
            if (x == camera.width / 2 + 40 && y == camera.height / 2) {
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
