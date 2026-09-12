// Copyright (c) 2026 lucabRTrender contributors.
//
// A primvar's value where a ray met a triangle, whatever its interpolation:
// constant (one value), uniform (per authored face), vertex and varying (per
// point, blended by barycentrics), faceVarying (per face-vertex corner,
// blended). Primvars are found by slot -- displayColor, displayOpacity,
// normals, st, then whatever a frame asks for -- through the scene's slot
// table (lrt::world::GpuScene).
module primvars;
import scene_types;

public static const uint kConstant = 0;
public static const uint kUniform = 1;
public static const uint kVarying = 2;
public static const uint kVertex = 3;
public static const uint kFaceVarying = 4;
public static const uint kNoPrimvar = 0xFFFFFFFF;

public static const uint kSlotDisplayColor = 0;
public static const uint kSlotDisplayOpacity = 1;
public static const uint kSlotNormals = 2;
public static const uint kSlotSt = 3;
public static const uint kSlotTangent = 4;   // a curve tube's fibre direction, a vertex

public struct PrimvarRecord {
    public uint interpolation;
    public uint components;
    public uint first;      // in primvarValues
    public uint count;
};

public StructuredBuffer<PrimvarRecord> primvarRecords;
public StructuredBuffer<float4>        primvarValues;
public StructuredBuffer<uint>          primvarSlots;      // per mesh, per slot: record, or kNoPrimvar
public StructuredBuffer<uint>          triangleCorners;   // pool, 3 per triangle
public StructuredBuffer<uint>          triangleFaces;     // pool, 1 per triangle

public uint primvarRecordOf(MeshRecord mesh, uint slot) {
    return primvarSlots[mesh.slotBase + slot];
}

/// `points` are the triangle's three point indices (mesh-local); `weights`
/// its barycentrics (w0, w1, w2).
public float4 primvarAt(uint record, MeshRecord mesh, uint triangle, uint3 points, float3 weights,
                        float4 fallback) {
    if (record == kNoPrimvar) {
        return fallback;
    }
    const PrimvarRecord r = primvarRecords[record];
    const uint t = mesh.firstTriangle + triangle;
    if (r.interpolation == kConstant) {
        return primvarValues[r.first];
    }
    if (r.interpolation == kUniform) {
        const uint face = triangleFaces[t];
        return face < r.count ? primvarValues[r.first + face] : fallback;
    }
    uint3 at;
    if (r.interpolation == kFaceVarying) {
        at = uint3(triangleCorners[t * 3], triangleCorners[t * 3 + 1], triangleCorners[t * 3 + 2]);
    } else {
        at = points;
    }
    if (any(at >= r.count)) {
        return fallback;
    }
    return primvarValues[r.first + at.x] * weights.x + primvarValues[r.first + at.y] * weights.y +
           primvarValues[r.first + at.z] * weights.z;
}
