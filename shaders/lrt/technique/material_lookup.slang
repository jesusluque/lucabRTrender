// Copyright (c) 2026 lucabRTrender contributors.
//
// Which material a visibility sample wears, and what the shading kernels and
// the cutout visibility passes read to evaluate it: a row per material (its
// compiled function and where its values sit in the blob), the GeomSubset
// tables that override an instance's row, and the frame's view to world.
module material_lookup;
__exported import lrt.technique.material_surface;

public struct MaterialRecord {
    public uint function;   // 0: the fallback (displayColor); k: the k-th compiled module
    public uint blob;       // its first word in the blob
    public uint flags;
    public uint pad0;
};

/// MaterialRecord.flags
public static const uint kMaterialCutout = 1u;   // its opacity says whether a sample is there: cut, never blended

public struct MaterialLookupParams {
    public float time;
    /// 1: a fractional opacity is cut by a pixel's lot (the raster route,
    /// which draws once). 0: only an opacity of zero cuts, and the path
    /// tracer casts the lot itself, one a sample.
    public uint  alphaDither;
    public uint  pad1; public uint pad2;
};

public StructuredBuffer<MaterialRecord> materials;
public StructuredBuffer<uint>           triangleSubsets;   // per triangle: 0, or its GeomSubset + 1; top bit: invisible face
public StructuredBuffer<uint>           subsetRows;        // per mesh subset: its material row, 0 for the instance's
public ConstantBuffer<ViewToWorld>      toWorld;
public ConstantBuffer<MaterialLookupParams> lookup;

/// The row a surface wears: its instance's, unless its triangle is in a
/// GeomSubset that binds one of its own.
public uint materialRowOf(Surface s) {
    uint row = s.instance.flags >> 8;
    const uint subset = triangleSubsets[s.mesh.firstTriangle + s.triangle] & 0x7FFFFFFFu;
    if (subset != 0) {
        const uint subsetRow = subsetRows[s.mesh.subsetBase + subset - 1];
        row = subsetRow != 0 ? subsetRow : row;
    }
    return row;
}
