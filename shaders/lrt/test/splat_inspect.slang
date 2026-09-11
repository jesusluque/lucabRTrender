// Copyright (c) 2026 lucabRTrender contributors.
//
// Unpacks one stored splat, on the GPU, with the same functions the renderer
// uses, so a test can compare 20 numbers without a CPU decoder of its own.
//   out: pos.xyz, opacity, scale.xyz, quat.xyzw, base.rgb, sh[0..5]
module splat_inspect;
import lrt.common.packing;

struct InspectParams {
    uint index;
    uint shWords;
    uint hasSh;
    uint pad0;
};

StructuredBuffer<float4>      positions;
StructuredBuffer<uint>        shape;
StructuredBuffer<uint>        sh;
RWStructuredBuffer<float>     values;   // 20
ConstantBuffer<InspectParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void splatInspect(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint i = params.index;
    const float4 p = positions[i];
    const uint row = i * 4;
    const float4 q = decodeQuaternion(shape[row]);
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]),
                                    lowHalf(shape[row + 2])));
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]),
                               highHalf(shape[row + 3]));
    values[0] = p.x; values[1] = p.y; values[2] = p.z; values[3] = p.w;
    values[4] = scale.x; values[5] = scale.y; values[6] = scale.z;
    values[7] = q.x; values[8] = q.y; values[9] = q.z; values[10] = q.w;
    values[11] = base.x; values[12] = base.y; values[13] = base.z;
    for (uint k = 0; k < 6; ++k) {
        const uint word = params.hasSh != 0 ? sh[i * params.shWords + k / 2] : 0;
        values[14 + k] = (k % 2 == 0) ? lowHalf(word) : highHalf(word);
    }
}
