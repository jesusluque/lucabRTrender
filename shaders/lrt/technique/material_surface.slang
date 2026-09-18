// Copyright (c) 2026 lucabRTrender contributors.
//
// A visibility sample as a material sees it: surface.slang's reconstruction
// carried into world and object space, with a tangent frame from the texture
// coordinates (dP/du, as surface gradients have it) and the texture
// footprint across the pixel from the neighbouring pixels' rays.
module material_surface;
__exported import surface;
__exported import lrt.material.material_inputs;

public struct ViewToWorld {
    public float4 row0;
    public float4 row1;
    public float4 row2;
};

public float3 applyRows(ViewToWorld m, float3 x, float w) {
    return float3(dot(m.row0.xyz, x) + m.row0.w * w, dot(m.row1.xyz, x) + m.row1.w * w,
                  dot(m.row2.xyz, x) + m.row2.w * w);
}

/// Barycentrics (w1, w2) of the ray's hit on the triangle's plane, unclamped.
float2 planeBarycentrics(float3 origin, float3 direction, float3 a, float3 b, float3 c) {
    const float3 e1 = b - a;
    const float3 e2 = c - a;
    const float3 p = cross(direction, e2);
    const float det = dot(e1, p);
    const float inverse = abs(det) > 1e-20 ? 1.0 / det : 0.0;
    const float3 s = origin - a;
    const float3 q = cross(s, e1);
    return float2(dot(s, p) * inverse, dot(direction, q) * inverse);
}

/// `cameraFootprint` is how wide a texture lookup here is: the pixel's own,
/// which is what a frame wants. A bake has no camera -- its neighbouring
/// threads are gaussians somewhere else on the model -- and taking the
/// footprint from one reads every texture at whatever mip that nonsense asks
/// for. Measured: the chess set's marble came back as its own average, its
/// veining gone and the gold ring on the pawn's collar grey, from a
/// conversion that had sampled the right texture all along. A bake asks for
/// no footprint and reads the top level, which is what the conversion's own
/// sampling does.
public MaterialInputs materialInputsAt(CameraParams camera, ViewToWorld toWorld, uint px, uint py, Surface s,
                                      float time) {
    return materialInputsAt(camera, toWorld, px, py, s, time, true);
}

public MaterialInputs materialInputsAt(CameraParams camera, ViewToWorld toWorld, uint px, uint py, Surface s,
                                      float time, bool cameraFootprint) {
    MaterialInputs inputs;
    const InstanceRecord r = s.instance;
    const uint base = s.mesh.firstPoint + s.instance.pointsOffset;
    const float3 objectA = positions[base + s.points.x].xyz;
    const float3 objectB = positions[base + s.points.y].xyz;
    const float3 objectC = positions[base + s.points.z].xyz;
    inputs.positionObject = objectA * s.weights.x + objectB * s.weights.y + objectC * s.weights.z;
    inputs.positionWorld = applyRows(toWorld, s.viewPosition, 1.0);
    inputs.viewPosition = applyRows(toWorld, float3(0.0), 1.0);
    // The shading normal facing the viewer's side as authored: flipped only
    // where the geometry itself faces away (a backface).
    const bool backface = dot(s.geometricNormal, s.viewDirection) > 0.0;
    inputs.inside = backface;
    const float3 facing = backface ? -s.normal : s.normal;
    inputs.normalWorld = normalize(applyRows(toWorld, facing, 0.0));
    inputs.normalObject = normalize(cross(objectB - objectA, objectC - objectA));

    // Texture coordinates at the corners and across the pixel.
    const uint stRecord = primvarRecordOf(s.mesh, kSlotSt);
    const float2 stA = primvarAt(stRecord, s.mesh, s.triangle, s.points, float3(1.0, 0.0, 0.0), float4(0.0)).xy;
    const float2 stB = primvarAt(stRecord, s.mesh, s.triangle, s.points, float3(0.0, 1.0, 0.0), float4(0.0)).xy;
    const float2 stC = primvarAt(stRecord, s.mesh, s.triangle, s.points, float3(0.0, 0.0, 1.0), float4(0.0)).xy;
    const float3 va = rowsApply(r.view0, r.view1, r.view2, objectA, 1.0);
    const float3 vb = rowsApply(r.view0, r.view1, r.view2, objectB, 1.0);
    const float3 vc = rowsApply(r.view0, r.view1, r.view2, objectC, 1.0);
    float3 origin;
    float3 direction;
    viewRay(camera, float2(px, py) + float2(1.5, 0.5), origin, direction);
    const float2 bx = planeBarycentrics(origin, direction, va, vb, vc);
    viewRay(camera, float2(px, py) + float2(0.5, 1.5), origin, direction);
    const float2 by = planeBarycentrics(origin, direction, va, vb, vc);
    const float2 here = s.weights.yz;
    const float2 dstdb1 = stB - stA;
    const float2 dstdb2 = stC - stA;
    inputs.uvDx = cameraFootprint ? dstdb1 * (bx.x - here.x) + dstdb2 * (bx.y - here.y) : float2(0.0);
    inputs.uvDy = cameraFootprint ? dstdb1 * (by.x - here.x) + dstdb2 * (by.y - here.y) : float2(0.0);

    // dP/du and dP/dv in world space, orthonormalised against the normal.
    const float3 wa = applyRows(toWorld, va, 1.0);
    const float3 e1 = applyRows(toWorld, vb, 1.0) - wa;
    const float3 e2 = applyRows(toWorld, vc, 1.0) - wa;
    const float det = dstdb1.x * dstdb2.y - dstdb1.y * dstdb2.x;
    float3 tangent = float3(0.0);
    if (stRecord != kNoPrimvar && abs(det) > 1e-12) {
        tangent = (e1 * dstdb2.y - e2 * dstdb1.y) / det;
    }
    // A tube carries its fibre's direction as a primvar: a hair material's
    // frame is the fibre's, not the texture's.
    const uint tangentRecord = primvarRecordOf(s.mesh, kSlotTangent);
    if (tangentRecord != kNoPrimvar) {
        const float3 objectT = primvarAt(tangentRecord, s.mesh, s.triangle, s.points, s.weights, float4(0.0)).xyz;
        const float3 viewT = rowsApply(r.view0, r.view1, r.view2, objectT, 0.0);
        tangent = applyRows(toWorld, viewT, 0.0);
    }
    tangent -= dot(tangent, inputs.normalWorld) * inputs.normalWorld;
    if (dot(tangent, tangent) < 1e-20) {
        const float3 n = inputs.normalWorld;
        const float sign = n.z >= 0.0 ? 1.0 : -1.0;
        const float a = -1.0 / (sign + n.z);
        tangent = float3(1.0 + sign * n.x * n.x * a, sign * n.x * n.y * a, -sign * n.x);
    }
    inputs.tangentWorld = normalize(tangent);
    inputs.bitangentWorld = cross(inputs.normalWorld, inputs.tangentWorld);
    inputs.tangentObject = inputs.tangentWorld;
    inputs.bitangentObject = inputs.bitangentWorld;
    inputs.frame = 0.0;
    inputs.time = time;
    inputs.mesh = s.mesh;
    inputs.triangle = s.triangle;
    inputs.points = s.points;
    inputs.weights = s.weights;
    inputs.displayColor = r.colour;
    // Object to world: the instance's object-to-view rows, then view to world.
    const float3 c0 = applyRows(toWorld, float3(r.view0.x, r.view1.x, r.view2.x), 0.0);
    const float3 c1 = applyRows(toWorld, float3(r.view0.y, r.view1.y, r.view2.y), 0.0);
    const float3 c2 = applyRows(toWorld, float3(r.view0.z, r.view1.z, r.view2.z), 0.0);
    const float3 c3 = applyRows(toWorld, float3(r.view0.w, r.view1.w, r.view2.w), 1.0);
    inputs.worldFromObject0 = float4(c0.x, c1.x, c2.x, c3.x);
    inputs.worldFromObject1 = float4(c0.y, c1.y, c2.y, c3.y);
    inputs.worldFromObject2 = float4(c0.z, c1.z, c2.z, c3.z);
    return inputs;
}
