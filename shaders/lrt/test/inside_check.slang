// Copyright (c) 2026 lucabRTrender contributors.
//
// What a material is told about the side a pixel sees, over a closed mesh:
// `inside` (a dielectric's relative IOR turns on it) and the shading normal
// it is handed, which must face the viewer whichever side that is. Seen from
// outside a closed mesh no pixel is inside; seen from within, every one is.
module inside_check;
import lrt.technique.material_surface;

Texture2D<uint4>             visibility;   // (instance + 1, triangle); row 0 on top
ConstantBuffer<CameraParams> camera;
ConstantBuffer<ViewToWorld>  toWorld;
RWStructuredBuffer<uint>     counts;       // [covered, inside, normal facing away from the eye]

[shader("compute")]
[numthreads(1, 1, 1)]
void insideCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    uint covered = 0;
    uint inside = 0;
    uint away = 0;
    for (uint y = 0; y < camera.height; ++y) {
        for (uint x = 0; x < camera.width; ++x) {
            const uint4 seen = visibility.Load(int3(int(x), int(camera.height - 1 - y), 0));
            if (seen.x == 0) continue;
            const Surface s = surfaceAt(camera, x, y, seen);
            const MaterialInputs m = materialInputsAt(camera, toWorld, x, y, s, 0.0);
            covered += 1;
            inside += m.inside ? 1u : 0u;
            away += dot(m.normalWorld, m.positionWorld - m.viewPosition) > 0.0 ? 1u : 0u;
        }
    }
    counts[0] = covered;
    counts[1] = inside;
    counts[2] = away;
}
