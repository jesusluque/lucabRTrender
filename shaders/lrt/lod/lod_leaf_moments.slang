// Copyright (c) 2026 lucabRTrender contributors.
//
// The finest merged level's moments, from the splats of each group.
module lod_leaf_moments;
import lrt.common.packing;
import lod_common;

StructuredBuffer<float4>   positions;   // Morton order
StructuredBuffer<uint>     shape;
StructuredBuffer<uint>     sh;
StructuredBuffer<uint>     starts;
RWStructuredBuffer<float>  moments;
ConstantBuffer<LodParams>  params;       // count: splats; groups

float halfAt(uint row, uint index) {
    const uint word = sh[row + index / 2];
    return (index & 1u) == 0u ? lowHalf(word) : highHalf(word);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lodLeafMoments(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.groups) {
        return;
    }
    const uint first = starts[g];
    const uint end = g + 1 < params.groups ? starts[g + 1] : params.count;
    const uint at = g * params.stride;
    for (uint k = 0; k < params.stride; ++k) {
        moments[at + k] = 0.0;
    }
    for (uint i = first; i < end; ++i) {
        const float4 p = positions[i];
        const uint row = i * 4;
        const float3 s = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
        const float4 q = decodeQuaternion(shape[row]);
        const float x = q.x, y = q.y, z = q.z, w = q.w;
        const float3x3 r = float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
        const float3x3 m = float3x3(r[0] * s, r[1] * s, r[2] * s);
        const float3x3 cov = mul(m, transpose(m));
        // The area it covers face on: its two longest axes.
        const float smallest = min(min(s.x, s.y), s.z);
        const float area = s.x * s.y * s.z / max(smallest, 1e-20);
        const float weight = p.w * area;
        moments[at] += weight;
        moments[at + 1] += weight * p.x;
        moments[at + 2] += weight * p.y;
        moments[at + 3] += weight * p.z;
        moments[at + 4] += weight * (cov[0][0] + p.x * p.x);
        moments[at + 5] += weight * (cov[0][1] + p.x * p.y);
        moments[at + 6] += weight * (cov[0][2] + p.x * p.z);
        moments[at + 7] += weight * (cov[1][1] + p.y * p.y);
        moments[at + 8] += weight * (cov[1][2] + p.y * p.z);
        moments[at + 9] += weight * (cov[2][2] + p.z * p.z);
        moments[at + 10] += weight * highHalf(shape[row + 2]);
        moments[at + 11] += weight * lowHalf(shape[row + 3]);
        moments[at + 12] += weight * highHalf(shape[row + 3]);
        for (uint half = 0; half < params.keep * 3; ++half) {
            moments[at + kMomentsHead + half] += weight * halfAt(i * params.shWords, half);
        }
    }
}
