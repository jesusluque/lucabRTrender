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

public void lrtBeginInputs(MaterialInputs inputs, uint blob) {
    gLrtInputs = inputs;
    gLrtBlob = blob;
}

// The genglsl reference (tests only): the light direction a reference
// material is evaluated for, and the BSDF response it returns.
public static float3 gLrtReferenceL;
public static float3 gLrtReferenceResponse;
public static float3 gLrtReferenceEmission;
