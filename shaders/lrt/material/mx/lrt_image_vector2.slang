// Copyright (c) 2026 lucabRTrender contributors.
//
// <image> (vector2): the texture table, or the default where the file or its tile is missing.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_image_vector2(SamplerTexture2D tex_sampler, int layer, float2 defaultval, float2 texcoord, int uaddressmode,
                   int vaddressmode, int filtertype, int framerange, int frameoffset, int frameendaction,
                   float2 uv_scale, float2 uv_offset, out float2 result)
{
    const float2 uv = texcoord * uv_scale + uv_offset;
    bool found = false;
    const float4 value = sampleTexture(gTextures, tex_sampler.id, tex_sampler.sampler, uv,
                                       gLrtInputs.uvDx * uv_scale, gLrtInputs.uvDy * uv_scale, found);
    result = found ? value.rg : defaultval;
}
