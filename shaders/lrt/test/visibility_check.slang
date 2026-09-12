// Copyright (c) 2026 lucabRTrender contributors.
//
// A square of half size `half`, facing the camera at view depth `z`, checked
// pixel by pixel against what it analytically is: covered where the pixel's
// ray meets the square, at depth z, lit by the cosine to the eye.
module visibility_check;
import lrt.technique.camera;

struct PlaneParams {
    float z;
    float half;
    float colourR; float colourG; float colourB;
    float pad0; float pad1; float pad2;
};

StructuredBuffer<float4>     colour;
StructuredBuffer<float>      depth;
RWStructuredBuffer<uint>     counts;   // [coverage mismatches, colour mismatches, covered]
RWStructuredBuffer<float>    worst;    // [depth error]
ConstantBuffer<CameraParams> camera;
ConstantBuffer<PlaneParams>  plane;

[shader("compute")]
[numthreads(1, 1, 1)]
void planeCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint coverage = 0;
    uint colours = 0;
    uint covered = 0;
    float depthError = 0.0;
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            float3 origin;
            float3 direction;
            viewRay(camera, float2(x, y) + 0.5, origin, direction);
            const float3 at = origin + direction * plane.z;
            // Pixel centres exactly on an edge are the rasteriser's to decide.
            const float margin = max(abs(at.x), abs(at.y)) - plane.half;
            const uint i = y * camera.width + x;
            const bool drawn = depth[i] > 0.0;
            if (abs(margin) > 1e-4) {
                coverage += (drawn != (margin < 0.0)) ? 1u : 0u;
            }
            if (drawn) {
                ++covered;
                depthError = max(depthError, abs(depth[i] - plane.z));
                const float cosine = 1.0 / length(direction);
                const float3 expected = float3(plane.colourR, plane.colourG, plane.colourB) * cosine;
                colours += any(abs(colour[i].rgb - expected) > 1e-5) || abs(colour[i].w - 1.0) > 0.0 ? 1u : 0u;
            }
        }
    }
    counts[0] = coverage;
    counts[1] = colours;
    counts[2] = covered;
    worst[0] = depthError;
}

// A sphere of radius r at (0, 0, -z) in view space, as the analytic ray meets
// it, against what a tessellation of it drew. A tessellation of n segments
// sits inside the sphere by up to r (1 - cos(pi / n)) -- a chord's sag -- so
// coverage is judged outside a band of that width about the silhouette, and
// the depth within that sag of the analytic front.
struct SphereParams {
    float z;        // distance to the centre, in front
    float radius;
    float sag;      // r (1 - cos(pi / n)) for the tessellation's n
    float pad0;
};
ConstantBuffer<SphereParams> sphere;

[shader("compute")]
[numthreads(1, 1, 1)]
void sphereCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint coverage = 0;
    uint covered = 0;
    uint judged = 0;
    float depthError = 0.0;
    const float3 centre = float3(0.0, 0.0, sphere.z);
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            float3 origin;
            float3 direction;
            viewRay(camera, float2(x, y) + 0.5, origin, direction);
            const float3 d = normalize(direction);
            const float3 oc = origin - centre;
            const float b = dot(oc, d);
            const float c = dot(oc, oc) - sphere.radius * sphere.radius;
            const float disc = b * b - c;
            const uint i = y * camera.width + x;
            const bool drawn = depth[i] > 0.0;
            // How far the ray passes from the centre, against the radius.
            const float perp = sqrt(max(dot(oc, oc) - b * b, 0.0));
            const float miss = perp - sphere.radius;
            if (abs(miss) > sphere.sag) {
                ++judged;
                coverage += (drawn != (miss < 0.0)) ? 1u : 0u;
            }
            // Depth where the ray meets the sphere squarely enough: within
            // 0.8 r of the axis a facet's error along the ray is at most the
            // sag over cos(asin 0.8); at the rim it is unbounded.
            if (drawn && disc > 0.0 && perp < 0.8 * sphere.radius) {
                ++covered;
                const float t = -b - sqrt(disc);
                const float3 hit = origin + d * t;
                depthError = max(depthError, abs(depth[i] - hit.z));
            }
        }
    }
    counts[0] = coverage;
    counts[1] = judged;
    counts[2] = covered;
    worst[0] = depthError;
}

// A cylinder along view x, its axis at (y, z), of radius r, as a tube of n
// sides draws it: coverage judged outside a band of the facets' sag about
// the silhouette (and clear of the ends), depth on the rows meeting the
// cylinder squarely (within half the radius of the axis) against its front.
struct CylinderParams {
    float y;         // the axis, in view space
    float z;
    float radius;
    float sag;       // r (1 - cos(pi / n))
    float xMin;      // view x of the ends: judged only between them
    float xMax;
    float pad0; float pad1;
};
ConstantBuffer<CylinderParams> cylinder;

[shader("compute")]
[numthreads(1, 1, 1)]
void cylinderCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint coverage = 0;
    uint covered = 0;
    float depthError = 0.0;
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            float3 origin;
            float3 direction;
            viewRay(camera, float2(x, y) + 0.5, origin, direction);
            // Where the ray is at the axis's depth: its offset from the axis.
            const float3 at = origin + direction * cylinder.z;
            if (at.x < cylinder.xMin + 3.0 * cylinder.radius || at.x > cylinder.xMax - 3.0 * cylinder.radius) {
                continue;
            }
            const float dy = at.y - cylinder.y;
            const uint i = y * camera.width + x;
            const bool drawn = depth[i] > 0.0;
            const float miss = abs(dy) - cylinder.radius;
            if (abs(miss) > cylinder.sag) {
                coverage += (drawn != (miss < 0.0)) ? 1u : 0u;
            }
            if (drawn && abs(dy) < 0.5 * cylinder.radius) {
                ++covered;
                // The ray's front hit: solve |(o + d t) - axis|^2 = r^2 in y, z.
                const float2 o = float2(origin.y - cylinder.y, origin.z - cylinder.z);
                const float2 d = float2(direction.y, direction.z);
                const float a = dot(d, d);
                const float b = dot(o, d);
                const float c = dot(o, o) - cylinder.radius * cylinder.radius;
                const float disc = b * b - a * c;
                if (disc > 0.0) {
                    const float t = (-b - sqrt(disc)) / a;
                    depthError = max(depthError, abs(depth[i] - (origin.z + direction.z * t)));
                }
            }
        }
    }
    counts[0] = coverage;
    counts[1] = 0;
    counts[2] = covered;
    worst[0] = depthError;
}
