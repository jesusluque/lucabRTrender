// Copyright (c) 2026 lucabRTrender contributors.
//
// Which triangle of which instance each pixel sees, by rasterisation: the
// fragment writes (instance + 1, triangle), 0 meaning nothing. What the pixel
// shows is then computed from those two numbers, the same way a ray traced
// visibility pass's are. The geometry is visibility_geometry's, which the
// cutout pass draws too.
module visibility_raster;
import lrt.technique.visibility_geometry;

ConstantBuffer<CameraParams> camera;

[shader("vertex")]
VertexOut visibilityVertex(uint vertex : SV_VertexID, uint instanceIndex : SV_InstanceID,
                           uint startVertex : SV_StartVertexLocation, uint startInstance : SV_StartInstanceLocation) {
    return visibilityVertexOf(camera, vertex, instanceIndex, startVertex, startInstance);
}

[shader("fragment")]
uint4 visibilityFragment(VertexOut input, bool front : SV_IsFrontFace) : SV_Target0 {
    if (visibilityCulled(input, front)) {
        discard;
    }
    return uint4(input.instance + 1, input.triangle, 0, 0);
}
