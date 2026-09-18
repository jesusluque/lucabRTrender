// Copyright (c) 2026 lucabRTrender contributors.
//
// What a bake's coefficients say the surface sends in one direction, against
// what it must send. It decodes them the way a cloud does -- the constant
// where 3DGS keeps it, the rest against the basis in the direction the eye
// looks -- so a fit that is right in the coefficients and wrong in the sum
// cannot pass.
module bake_check;
import lrt.common.sh;
import lrt.common.packing;

struct BakeCheckParams {
    uint   count;        // points
    uint   planes;       // coefficients each
    uint   pad0;
    uint   pad1;
    float4 direction;    // xyz: from the eye to the point; w: the relative tolerance
    float4 expected;     // xyz: the light it sends, in the space a cloud is blended in
};

StructuredBuffer<float4>        coefficients;
RWStructuredBuffer<uint>        counts;   // [checked, beyond the tolerance]
RWStructuredBuffer<float>       worst;    // [worst relative difference, then the first point's rgb]
ConstantBuffer<BakeCheckParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void bakeCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint  beyond = 0;
    float worstRelative = 0.0;
    for (uint p = 0; p < params.count; ++p) {
        const uint base = p * params.planes;
        float3 got = float3(0.5) + kSH0 * coefficients[base].xyz;
        for (uint k = 1; k < params.planes; ++k) {
            got += coefficients[base + k].xyz * shBasisValue(k, params.direction.xyz);
        }
        const float3 want = params.expected.xyz;
        const float scale = max(max(want.r, want.g), max(want.b, 1.0e-6));
        const float relative = max(max(abs(got.r - want.r), abs(got.g - want.g)), abs(got.b - want.b)) / scale;
        worstRelative = max(worstRelative, relative);
        beyond += relative > params.direction.w ? 1u : 0u;
        if (p == 0) {
            worst[1] = got.r;
            worst[2] = got.g;
            worst[3] = got.b;
        }
    }
    counts[0] = params.count;
    counts[1] = beyond;
    worst[0] = worstRelative;
}
