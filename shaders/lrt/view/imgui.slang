// Copyright (c) 2026 lucabRTrender contributors.
//
// Dear ImGui's draw lists over a display texture. Its vertices and indices
// arrive whole in two buffers; a draw starts at its first index (the draw's
// start vertex) and adds its list's first vertex.
module imgui;

struct UiVertex {
    float2 position;   // ImGui's pixels, top left origin
    float2 uv;
    uint   colour;     // ImU32: R in the low byte, sRGB-encoded as the target
};

struct UiParams {
    float scaleX;
    float scaleY;
    float translateX;
    float translateY;
    uint  vertexBase;
    uint  idsIncludeStart;
    uint  pad0; uint pad1;
};

StructuredBuffer<UiVertex>  vertices;
StructuredBuffer<uint>      indices;
Texture2D<float4>           atlas;
SamplerState                atlasSampler;
ConstantBuffer<UiParams>    params;

struct UiOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 colour : COLOR0;
};

[shader("vertex")]
UiOut uiVertex(uint vertex : SV_VertexID, uint startVertex : SV_StartVertexLocation) {
    const uint corner = params.idsIncludeStart != 0 ? vertex : vertex + startVertex;
    const UiVertex v = vertices[indices[corner] + params.vertexBase];
    UiOut out;
    out.position = float4(v.position.x * params.scaleX + params.translateX,
                          v.position.y * params.scaleY + params.translateY, 0.0, 1.0);
    out.uv = v.uv;
    out.colour = float4(float(v.colour & 255u), float((v.colour >> 8) & 255u), float((v.colour >> 16) & 255u),
                        float((v.colour >> 24) & 255u)) / 255.0;
    return out;
}

[shader("fragment")]
float4 uiFragment(UiOut input) : SV_Target0 {
    return input.colour * atlas.Sample(atlasSampler, input.uv);
}
