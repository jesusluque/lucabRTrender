// Copyright (c) 2026 lucabRTrender contributors.
//
// Ray tracing Gaussians, the way 3DGRT does (Moenne-Loccoz et al. 2024):
// each Gaussian is enclosed by a stretched icosahedron, the proxies go into a
// hardware BVH, and a ray gathers the k nearest proxies it enters, integrates
// those particles at their maximum response along the ray, and continues from
// the last one. No projection, no tiles, no sort of the scene: the order is
// the ray's own.
module rt_common;
import lrt.common.packing;
__exported import lrt.common.edit;

/// A ray, as both routes carry it. Not `RayDesc`: that type exists on a
/// target only where ray tracing does, and the compute BVH route is meant to
/// run where there is none (CUDA without OptiX -- measured, nvrtc: "identifier
/// RayDesc is undefined"). The hardware route fills a RayDesc from this.
public struct LrtRay {
    public float3 Origin;
    public float  TMin;
    public float3 Direction;
    public float  TMax;
};

public static const uint kIcoVertices = 12;
public static const uint kIcoTriangles = 20;
/// Inradius of the unit-circumradius icosahedron: vertices are scaled by
/// cutoff / this, so the icosahedron's faces clear the cutoff ellipsoid.
public static const float kIcoInradius = 0.7946544723;

public struct RtParams {
    public uint  width;
    public uint  height;
    public float focalX;
    public float focalY;
    public float centreX;
    public float centreY;
    public float nearZ;
    public float farZ;
    public uint  orthographic;
    public uint  count;          // splats in this cloud (proxy, frames, shade)
    public uint  base;           // proxy: splats per chunk; frames, shade: first slot written
    public uint  shWords;        // shade: this cloud's words per splat
    // view -> world, rows of a 3x4 (camera to world with the kernel's +z)
    public float v00; public float v01; public float v02; public float v03;
    public float v10; public float v11; public float v12; public float v13;
    public float v20; public float v21; public float v22; public float v23;
    public uint  shLimit;
    public uint  linearise;
    public float bgR; public float bgG; public float bgB; public float bgA;
    public uint  restPerColour;  // shade: this cloud's harmonics
    public uint  instances;      // bvh render: instances to traverse
    public uint  maxSegments;
    public uint  depthMode;      // 0 alpha-weighted mean, 1 first crossing
    public float depthThreshold;
    /// Where a query is cut in two and composed again (rtRenderSplit): what
    /// proves the segment query composes. 0: one query, as a frame draws.
    public float splitAt;
    public uint  pad1; public uint pad2;
    // shade: what this cloud is relit with. The rows take a splat to the
    // world, where the lights are; the eye is there too.
    public uint  relight;        // 1: light these splats rather than show them as baked
    public uint  lightCount;
    public uint  categoriesLo; public uint categoriesHi;
    public uint  hasPbr; public uint pad3; public uint pad4; public uint pad5;
    public float w00; public float w01; public float w02; public float w03;
    public float w10; public float w11; public float w12; public float w13;
    public float w20; public float w21; public float w22; public float w23;
    public float eyeWorldX; public float eyeWorldY; public float eyeWorldZ; public float pad6;
    public EditParams edit;      // shade: this instance's SplatEdit (common/edit.slang)
};

/// The squared Mahalanobis radius a particle is cut at: the raster's rule, so
/// the two routes draw the same extent of every particle.
public float rtCutoff(float alpha) {
    return min(2.0 * log(max(255.0 * alpha, 1.0001)), 9.0);
}

public float3x3 rotationOf(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}
