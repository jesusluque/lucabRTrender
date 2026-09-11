// Copyright (c) 2026 lucabRTrender contributors.
//
// One stretched icosahedron per Gaussian, in the cloud's own space. A Gaussian
// below 1/255 opacity gets a degenerate proxy (every vertex at its centre),
// which no ray enters.
module rt_proxy;
import lrt.common.packing;
import rt_common;

static const float kPhi = 1.6180339887;
static const float kNorm = 0.5257311121;   // 1 / sqrt(1 + phi^2)
static const float3 kVertex[12] = {
    float3(-1, kPhi, 0), float3(1, kPhi, 0), float3(-1, -kPhi, 0), float3(1, -kPhi, 0),
    float3(0, -1, kPhi), float3(0, 1, kPhi), float3(0, -1, -kPhi), float3(0, 1, -kPhi),
    float3(kPhi, 0, -1), float3(kPhi, 0, 1), float3(-kPhi, 0, -1), float3(-kPhi, 0, 1) };
// Counter-clockwise seen from outside.
static const uint kFace[60] = {
    0, 11, 5,  0, 5, 1,  0, 1, 7,  0, 7, 10,  0, 10, 11,
    1, 5, 9,   5, 11, 4, 11, 10, 2, 10, 7, 6,  7, 1, 8,
    3, 9, 4,   3, 4, 2,  3, 2, 6,   3, 6, 8,   3, 8, 9,
    4, 9, 5,   2, 4, 11, 6, 2, 10,  8, 6, 7,   9, 8, 1 };

StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
RWStructuredBuffer<float4> vertices;   // count * 12; xyz read as RGB32Float, stride 16
RWStructuredBuffer<uint>   indices;    // count * 60, relative to the chunk
ConstantBuffer<RtParams>   params;

[shader("compute")]
[numthreads(256, 1, 1)]
void rtProxy(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = positions[i];
    const uint row = i * 4;
    const float alpha = p.w;
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    const float3x3 r = rotationOf(decodeQuaternion(shape[row]));
    const float k = alpha >= 1.0 / 255.0 ? sqrt(rtCutoff(alpha)) / kIcoInradius : 0.0;
    for (uint v = 0; v < kIcoVertices; ++v) {
        const float3 local = kVertex[v] * kNorm * scale * k;
        vertices[i * 12 + v] = float4(p.xyz + mul(r, local), 1.0);
    }
    for (uint f = 0; f < 60; ++f) {
        // Relative to the splat's chunk (params.base splats per chunk): each
        // chunk is its own bottom-level structure over a window of vertices.
        indices[i * 60 + f] = (i % params.base) * 12 + kFace[f];
    }
}
