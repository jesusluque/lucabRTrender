// Copyright (c) 2026 lucabRTrender contributors.
//
// What both visibility-by-rays routes share: the camera's ray, the ids they
// write, and where a pixel's ray starts. The traversal itself is the routes'
// own -- inline in visibility_trace.slang, a pipeline in
// visibility_trace_rays.slang -- because a module compiles every entry point
// it holds, and neither device can compile the other's.
module visibility_trace_common;
__exported import camera;

public struct TraceParams {
    // view to world, rows of a 3x4
    public float v00; public float v01; public float v02; public float v03;
    public float v10; public float v11; public float v12; public float v13;
    public float v20; public float v21; public float v22; public float v23;
};

public uniform RaytracingAccelerationStructure scene;
public RWTexture2D<uint4>           ids;
public ConstantBuffer<CameraParams> camera;
public ConstantBuffer<TraceParams>  trace;

public struct TraceHit {
    public uint4 seen;   // (instance + 1, triangle, 0, 0); 0 for nothing
    public float t;      // view depth: the ray's direction has view z = 1
};

/// The pixel's primary ray in world space.
public void traceRay(uint2 pixel, out float3 origin, out float3 direction) {
    float3 viewOrigin;
    float3 viewDirection;
    viewRay(camera, float2(pixel) + 0.5, viewOrigin, viewDirection);
    const float3x4 toWorld = float3x4(trace.v00, trace.v01, trace.v02, trace.v03, trace.v10, trace.v11, trace.v12,
                                      trace.v13, trace.v20, trace.v21, trace.v22, trace.v23);
    origin = mul(toWorld, float4(viewOrigin, 1.0));
    direction = mul(toWorld, float4(viewDirection, 0.0));
}

/// Where a pixel's ray starts: the rasteriser's near plane.
public float traceTMin() {
    return camera.orthographic != 0 ? 0.0 : camera.nearZ;
}

