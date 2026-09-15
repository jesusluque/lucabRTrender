// Copyright (c) 2026 lucabRTrender contributors.
//
// What every function generated from a MaterialX graph reads, whichever way
// its closures are implemented: MaterialX's shader types, the shading point's
// inputs, the parameter blob, and the texture wrappers over the texture table.
// material_runtime.slang adds the lobe build stack; the genglsl reference
// (tests) uses this alone.
module material_inputs;
__exported import lrt.material.texture_table;
__exported import lrt.world.scene_types;
__exported import lrt.world.primvars;

public struct VDF {
    public float3 response;
    public float3 throughput;
};

public struct surfaceshader {
    public float3 color;          // emission, as the surface node here writes it
    public float3 transparency;
};

public struct volumeshader {
    public float3 color;
    public float3 transparency;
};

public struct displacementshader {
    public float3 offset;
    public float  scale;
};

public struct lightshader {
    public float3 intensity;
    public float3 direction;
};

// ---- the shading point --------------------------------------------------------

public struct MaterialInputs {
    public float3     positionWorld;
    public float3     normalWorld;            // shading normal, unit, facing the viewer's side as authored
    public float3     tangentWorld;
    public float3     bitangentWorld;
    public float3     positionObject;
    public float3     normalObject;
    public float3     tangentObject;
    public float3     bitangentObject;
    public float3     viewPosition;           // the camera, world space
    public float2     uvDx;                   // texcoord 0's footprint across the pixel
    public float2     uvDy;
    public float      frame;
    public float      time;
    public bool       inside;                 // the ray is inside the surface (backface of a closed mesh)
    public MeshRecord mesh;
    public uint       triangle;
    public uint3      points;
    public float3     weights;
    public float4     displayColor;           // the instance's fallback displayColor, displayOpacity
    // Object to world, as the rows of a 3x4 (column vectors): what MaterialX's
    // world matrices and object-space transforms are made of.
    public float4     worldFromObject0;
    public float4     worldFromObject1;
    public float4     worldFromObject2;
};

// Set by the material function for everything it calls.
public static MaterialInputs gLrtInputs;
public static uint           gLrtBlob;

public StructuredBuffer<float>        gMaterialBlob;
public ParameterBlock<TextureTable>   gTextures;

public float lrtBlob(uint offset) {
    return gMaterialBlob[gLrtBlob + offset];
}

/// A primvar of the shading point by scene slot, as float4.
public float4 lrtPrimvar(uint slot, float4 fallback) {
    if (slot == kNoPrimvar) {
        return fallback;
    }
    return primvarAt(primvarRecordOf(gLrtInputs.mesh, slot), gLrtInputs.mesh, gLrtInputs.triangle, gLrtInputs.points,
                     gLrtInputs.weights, fallback);
}

// ---- spaces -----------------------------------------------------------------------

/// An affine transform as the rows of a 3x4, acting on column vectors.
public struct Affine {
    public float4 r0;
    public float4 r1;
    public float4 r2;
};

public Affine lrtAffine(float4 r0, float4 r1, float4 r2) {
    Affine m;
    m.r0 = r0;
    m.r1 = r1;
    m.r2 = r2;
    return m;
}

public Affine lrtIdentityAffine() {
    return lrtAffine(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0));
}

/// m applied to v, with w 1 for a point and 0 for a vector.
public float3 lrtApply(Affine m, float3 v, float w) {
    return float3(dot(m.r0.xyz, v) + m.r0.w * w, dot(m.r1.xyz, v) + m.r1.w * w, dot(m.r2.xyz, v) + m.r2.w * w);
}

/// The 3x3 part's transpose applied to v.
public float3 lrtApplyTranspose(Affine m, float3 v) {
    return float3(m.r0.x * v.x + m.r1.x * v.y + m.r2.x * v.z, m.r0.y * v.x + m.r1.y * v.y + m.r2.y * v.z,
                  m.r0.z * v.x + m.r1.z * v.y + m.r2.z * v.z);
}

/// The inverse of an affine transform: the 3x3 part's by its cofactors, the
/// translation carried back through it. A singular one inverts to identity.
public Affine lrtInverse(Affine m) {
    const float3 a = m.r0.xyz;
    const float3 b = m.r1.xyz;
    const float3 c = m.r2.xyz;
    const float3 bc = cross(b, c);
    const float det = dot(a, bc);
    if (abs(det) < 1e-30) {
        return lrtIdentityAffine();
    }
    // The inverse's columns are the cross products of the rows over det.
    const float3 x = bc / det;
    const float3 y = cross(c, a) / det;
    const float3 z = cross(a, b) / det;
    Affine inv;
    inv.r0 = float4(x.x, y.x, z.x, 0.0);
    inv.r1 = float4(x.y, y.y, z.y, 0.0);
    inv.r2 = float4(x.z, y.z, z.z, 0.0);
    const float3 t = float3(m.r0.w, m.r1.w, m.r2.w);
    const float3 back = -lrtApply(inv, t, 0.0);
    inv.r0.w = back.x;
    inv.r1.w = back.y;
    inv.r2.w = back.z;
    return inv;
}

/// A transform as the float4x4 MaterialX's mx_matrix_mul(m, v) -- mul(v, m)
/// in genslang -- applies as m to a column vector: its transpose.
public float4x4 lrtMatrixApplying(Affine m) {
    return float4x4(m.r0.x, m.r1.x, m.r2.x, 0.0, m.r0.y, m.r1.y, m.r2.y, 0.0, m.r0.z, m.r1.z, m.r2.z, 0.0,
                    m.r0.w, m.r1.w, m.r2.w, 1.0);
}

/// The float4x4 that applies m's transpose.
public float4x4 lrtMatrixApplyingTranspose(Affine m) {
    return float4x4(m.r0, m.r1, m.r2, float4(0.0, 0.0, 0.0, 1.0));
}

public Affine lrtWorldFromObject() {
    return lrtAffine(gLrtInputs.worldFromObject0, gLrtInputs.worldFromObject1, gLrtInputs.worldFromObject2);
}

/// What a space name stands for: 0 world, 1 object (MaterialX's model and
/// object), 2 a coordinate system bound to the prim, whose transform to world
/// is three constant primvars (its rows) found at these scene slots. A system
/// the prim does not have is world.
public Affine lrtWorldFromSpace(uint kind, uint row0, uint row1, uint row2) {
    if (kind == 1) {
        return lrtWorldFromObject();
    }
    if (kind == 2) {
        return lrtAffine(lrtPrimvar(row0, float4(1.0, 0.0, 0.0, 0.0)), lrtPrimvar(row1, float4(0.0, 1.0, 0.0, 0.0)),
                         lrtPrimvar(row2, float4(0.0, 0.0, 1.0, 0.0)));
    }
    return lrtIdentityAffine();
}

/// MaterialX's transformpoint (w 1), transformvector (w 0) and
/// transformnormal (normal 1: by the inverse transposes, normalised) between
/// two spaces, through world.
public float3 lrtTransformBetween(float3 v, float w, bool normal, uint fromKind, uint f0, uint f1, uint f2,
                                  uint toKind, uint t0, uint t1, uint t2) {
    const Affine from = lrtWorldFromSpace(fromKind, f0, f1, f2);
    const Affine to = lrtWorldFromSpace(toKind, t0, t1, t2);
    if (normal) {
        // World from `from` takes a normal by inverse(from) transposed; `to`
        // from world by inverse(inverse(to)) transposed, which is to's transpose.
        return normalize(lrtApplyTranspose(to, lrtApplyTranspose(lrtInverse(from), v)));
    }
    return lrtApply(lrtInverse(to), lrtApply(from, v, w), w);
}

// ---- textures (mx_texture.slang's interface over the table) -------------------

public struct SamplerTexture2D {
    public uint id;
    public uint sampler;
};

public SamplerTexture2D lrtTextureOf(float id, float sampler) {
    SamplerTexture2D t;
    t.id = uint(id);
    t.sampler = uint(sampler);
    return t;
}

public bool lrtTextureFound(SamplerTexture2D tex) {
    return tex.id < 0xFFFFFFFFu && (gTextures.records[tex.id].flags & 1u) != 0;
}

public float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 dx, float2 dy) {
    bool found;
    return sampleTexture(gTextures, tex.id, tex.sampler, uv, dx, dy, found);
}

/// Without derivatives in compute, the footprint is texcoord 0's.
public float4 texture(SamplerTexture2D tex, float2 uv) {
    return textureGrad(tex, uv, gLrtInputs.uvDx, gLrtInputs.uvDy);
}

public float4 textureLod(SamplerTexture2D tex, float2 uv, float lod) {
    const float scale = pow(2.0, lod) / max(gTextures.records[tex.id].width, 1.0);
    return textureGrad(tex, uv, float2(scale, 0.0), float2(0.0, scale));
}

public int2 textureSize(SamplerTexture2D tex, int mipLevel) {
    const TextureRecord r = gTextures.records[tex.id];
    return int2(max(int(r.width) >> mipLevel, 1), max(int(r.height) >> mipLevel, 1));
}

// ---- screen derivatives -------------------------------------------------------
//
// A material is evaluated in a compute kernel, where there is no dFdx: the
// nodes that need one (heighttonormal, and so bump) take it from the quad the
// thread sits in. A quad is four threads whose lane indices differ in the two
// low bits, and those lanes are handed out along the group's rows -- measured
// on the device, not assumed -- so a kernel that wants derivatives has to
// walk its pixels in quad order itself (lrtQuadPixel): then bit 0 of the lane
// is x and bit 1 is y, and the difference is turned round on the far lane to
// give the forward derivative a coarse dFdx gives. Every thread of the quad
// must reach the call: where one of the four shades something else -- a
// silhouette -- or leaves early, the derivative is that of whatever it did
// evaluate.

/// The pixel a thread of a [numthreads(16, 16, 1)] group shades, in quad
/// order: each quad of four lanes covers a 2 x 2 block of pixels.
public uint2 lrtQuadPixel(uint2 group, uint index) {
    const uint quad = index >> 2;
    const uint lane = index & 3u;
    return uint2(group.x * 16u + (quad & 7u) * 2u + (lane & 1u),
                 group.y * 16u + (quad >> 3u) * 2u + (lane >> 1u));
}

public float lrtDdx(float v) {
    const float other = QuadReadAcrossX(v);
    return (WaveGetLaneIndex() & 1u) != 0u ? v - other : other - v;
}

public float lrtDdy(float v) {
    const float other = QuadReadAcrossY(v);
    return (WaveGetLaneIndex() & 2u) != 0u ? v - other : other - v;
}

public float2 lrtDdx(float2 v) { return float2(lrtDdx(v.x), lrtDdx(v.y)); }
public float2 lrtDdy(float2 v) { return float2(lrtDdy(v.x), lrtDdy(v.y)); }

public void lrtBeginInputs(MaterialInputs inputs, uint blob) {
    gLrtInputs = inputs;
    gLrtBlob = blob;
}

// The genglsl reference (tests only): the light direction a reference
// material is evaluated for, and the BSDF response it returns.
public static float3 gLrtReferenceL;
public static float3 gLrtReferenceResponse;
public static float3 gLrtReferenceEmission;
