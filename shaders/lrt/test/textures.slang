// Copyright (c) 2026 lucabRTrender contributors.
//
// Texture tests' generators and checks: the pattern a texel should hold, what
// a texture holds against it, a level's mean, a raster pass's coverage.
module textures;

struct TextureParams {
    uint  width;
    uint  height;
    uint  mode;       // check: 0 Load, 1 SampleLevel at the texel centre
    uint  pad0;
    float colourR; float colourG; float colourB; float colourA;
};

Texture2D<float4>             texture;
RWTexture2D<float4>           written;
SamplerState                  linearClamp;
RWStructuredBuffer<uint>      counts;     // check: [mismatches, texels]
RWStructuredBuffer<float>     mean;       // mean: [value]
ConstantBuffer<TextureParams> params;

/// The pattern: distinct per texel and channel, exactly representable.
public float4 patternAt(uint x, uint y) {
    return float4(float((x * 7 + y * 13) % 251), float((x * 3 + y * 5 + 1) % 241),
                  float((x + y * 11 + 2) % 239), float((x * 5 + y + 3) % 233)) / 256.0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void textureFill(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    written[tid.xy] = patternAt(tid.x, tid.y);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void textureCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint mismatches = 0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            float4 value;
            if (params.mode == 0) {
                value = texture.Load(int3(int(x), int(y), 0));
            } else {
                const float2 uv = (float2(x, y) + 0.5) / float2(params.width, params.height);
                value = texture.SampleLevel(linearClamp, uv, 0.0);
            }
            mismatches += any(value != patternAt(x, y)) ? 1u : 0u;
        }
    }
    counts[0] = mismatches;
    counts[1] = params.width * params.height;
}

/// The mean of a level (Kahan-summed, all four channels).
[shader("compute")]
[numthreads(1, 1, 1)]
void textureMean(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float sum = 0.0;
    float carry = 0.0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            const float4 v = texture.Load(int3(int(x), int(y), 0));
            const float term = v.r + v.g + v.b + v.a;
            const float t = sum + term;
            carry += abs(sum) >= abs(term) ? (sum - t) + term : (term - t) + sum;
            sum = t;
        }
    }
    mean[0] = (sum + carry) / float(params.width * params.height * 4);
}

/// Texels holding exactly the colour in params.
[shader("compute")]
[numthreads(1, 1, 1)]
void textureCountColour(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const float4 colour = float4(params.colourR, params.colourG, params.colourB, params.colourA);
    uint n = 0;
    for (uint y = 0; y < params.height; ++y) {
        for (uint x = 0; x < params.width; ++x) {
            n += all(texture.Load(int3(int(x), int(y), 0)) == colour) ? 1u : 0u;
        }
    }
    counts[0] = n;
    counts[1] = params.width * params.height;
}

// A raster pass: two triangles over the left half of clip space.
struct RasterOut {
    float4 position : SV_Position;
};

[shader("vertex")]
RasterOut rasterHalfVertex(uint vertex : SV_VertexID) {
    static const float2 kCorners[6] = { float2(-1, -1), float2(0, -1), float2(0, 1),
                                        float2(-1, -1), float2(0, 1), float2(-1, 1) };
    RasterOut out;
    out.position = float4(kCorners[vertex], 0.5, 1.0);
    return out;
}

[shader("fragment")]
float4 rasterHalfFragment(RasterOut input) : SV_Target0 {
    return float4(params.colourR, params.colourG, params.colourB, params.colourA);
}


// What a draw's start locations become in the vertex stage, on this backend:
// a quad over the target, each pixel the ids its vertex saw.
struct StartsOut {
    float4 position : SV_Position;
    nointerpolation uint4 ids : IDS;
};

[shader("vertex")]
StartsOut rasterStartsVertex(uint vertex : SV_VertexID, uint instance : SV_InstanceID,
                             uint startVertex : SV_StartVertexLocation, uint startInstance : SV_StartInstanceLocation) {
    static const float2 kCorners[6] = { float2(-1, -1), float2(1, -1), float2(1, 1),
                                        float2(-1, -1), float2(1, 1), float2(-1, 1) };
    StartsOut out;
    out.position = float4(kCorners[vertex % 6], 0.5, 1.0);   // the draw starts at a multiple of 6
    out.ids = uint4(vertex - vertex % 6, instance, startVertex, startInstance);
    return out;
}

[shader("fragment")]
uint4 rasterStartsFragment(StartsOut input) : SV_Target0 {
    return input.ids;
}

Texture2D<uint4> idsTexture;

[shader("compute")]
[numthreads(1, 1, 1)]
void textureTexelIds(uint3 tid: SV_DispatchThreadID) {
    const uint4 v = idsTexture.Load(int3(int(params.width / 2), int(params.height / 2), 0));
    counts[0] = v.x;
    counts[1] = v.y;
    counts[2] = v.z;
    counts[3] = v.w;
}

/// A colour exactly representable in eight bits, written through an
/// RWTexture2D<float4> and read back as the texture actually holds it. Metal
/// converts a float4 store into the texture's format; a backend whose store
/// writes the float's own bytes instead shows up here, one kernel away from
/// the texture store's decode rather than four. The bits are reported raw so
/// that whatever comes back stays visible.
[shader("compute")]
[numthreads(1, 1, 1)]
void formatWrite(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    written[uint2(0, 0)] = float4(1.0, 0.0, 64.0 / 255.0, 1.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void formatRead(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const float4 v = texture.Load(int3(0, 0, 0));
    counts[0] = asuint(v.r);
    counts[1] = asuint(v.g);
    counts[2] = asuint(v.b);
    counts[3] = asuint(v.a);
}

/// The same colour, packed into the texture's own bytes and stored through a
/// uint view of it. Four bytes, written as four bytes, asking neither backend
/// to convert anything -- which is what an eight-bit texture written from a
/// compute kernel needs if the two targets disagree about what a float4 store
/// means. Read back through the ordinary view, it should be the colour again.
RWTexture2D<uint> packed;

[shader("compute")]
[numthreads(1, 1, 1)]
void packedWrite(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const float4 colour = float4(1.0, 0.0, 64.0 / 255.0, 1.0);
    const uint4 byte = uint4(round(saturate(colour) * 255.0));
    packed[uint2(0, 0)] = byte.x | (byte.y << 8) | (byte.z << 16) | (byte.w << 24);
}
