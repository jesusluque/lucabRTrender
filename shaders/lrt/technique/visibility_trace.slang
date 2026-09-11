// Copyright (c) 2026 lucabRTrender contributors.
//
// Visibility by rays: the pixel centre's primary ray against the scene's
// hardware acceleration structure, the nearest triangle written as the
// rasteriser writes it -- (instance + 1, triangle), row 0 on top -- so
// shading cannot tell the two passes apart.
module visibility_trace;
import camera;

struct TraceParams {
    // view to world, rows of a 3x4
    float v00; float v01; float v02; float v03;
    float v10; float v11; float v12; float v13;
    float v20; float v21; float v22; float v23;
};

uniform RaytracingAccelerationStructure scene;
RWTexture2D<uint4>           ids;
ConstantBuffer<CameraParams> camera;
ConstantBuffer<TraceParams>  trace;

[shader("compute")]
[numthreads(16, 16, 1)]
void visibilityTrace(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    float3 origin;
    float3 direction;
    viewRay(camera, float2(tid.xy) + 0.5, origin, direction);
    const float3x4 toWorld = float3x4(trace.v00, trace.v01, trace.v02, trace.v03, trace.v10, trace.v11, trace.v12,
                                      trace.v13, trace.v20, trace.v21, trace.v22, trace.v23);
    RayDesc ray;
    ray.Origin = mul(toWorld, float4(origin, 1.0));
    ray.Direction = mul(toWorld, float4(direction, 0.0));
    // t is view depth (direction has view z = 1): the rasteriser's near
    // plane, and no far plane, as its reversed infinite depth has none.
    ray.TMin = camera.orthographic != 0 ? 0.0 : camera.nearZ;
    ray.TMax = 3.0e38;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    query.Proceed();
    const int2 texel = int2(int(tid.x), int(camera.height - 1 - tid.y));
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        ids[texel] = uint4(query.CommittedInstanceID() + 1, query.CommittedPrimitiveIndex(), 0, 0);
    } else {
        ids[texel] = uint4(0);
    }
}
