// Copyright (c) 2026 lucabRTrender contributors.
//
// <heighttonormal>, and through it <bump>: MaterialX's own formula, with the
// screen derivatives taken from the thread's 2x2 quad (material_inputs) since
// a material is evaluated in a compute kernel.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void lrt_heighttonormal_vector3(float height, float scale, float2 texcoord, out float3 result)
{
    // Scale factor for parity with traditional Sobel filtering.
    const float SOBEL_SCALE_FACTOR = 1.0 / 16.0;

    const float2 dHdS = float2(lrtDdx(height), lrtDdy(height)) * scale * SOBEL_SCALE_FACTOR;
    const float2 dUdS = float2(lrtDdx(texcoord.x), lrtDdy(texcoord.x));
    const float2 dVdS = float2(lrtDdx(texcoord.y), lrtDdy(texcoord.y));

    const float3 tangent = float3(dUdS.x, dVdS.x, dHdS.x);
    const float3 bitangent = float3(dUdS.y, dVdS.y, dHdS.y);
    float3 n = cross(tangent, bitangent);

    // Invalid and mirrored texture coordinates.
    if (dot(n, n) < M_FLOAT_EPS * M_FLOAT_EPS)
    {
        n = float3(0.0, 0.0, 1.0);
    }
    else if (n.z < 0.0)
    {
        n *= -1.0;
    }
    result = normalize(n) * 0.5 + 0.5;
}
