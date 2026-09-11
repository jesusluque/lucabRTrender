// Copyright (c) 2026 lucabRTrender contributors.
//
// How far apart two renders are, in the units a person sees: 8-bit sRGB code
// values, the largest channel difference per pixel, as a 256-bin histogram
// summed per row of pixels. The CPU reads the histogram and nothing else.
module image_compare;

struct CompareParams {
    uint width;
    uint height;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float4>      a;
StructuredBuffer<float4>      b;
RWStructuredBuffer<uint>      histogram;   // height * 256
ConstantBuffer<CompareParams> params;

float encode(float linear) {
    const float c = saturate(linear);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

uint code(float linear) { return uint(round(encode(linear) * 255.0)); }

[shader("compute")]
[numthreads(64, 1, 1)]
void imageCompare(uint3 tid: SV_DispatchThreadID) {
    const uint y = tid.x;
    if (y >= params.height) {
        return;
    }
    const uint row = y * 256;
    for (uint k = 0; k < 256; ++k) {
        histogram[row + k] = 0;
    }
    for (uint x = 0; x < params.width; ++x) {
        const float4 pa = a[y * params.width + x];
        const float4 pb = b[y * params.width + x];
        uint worst = 0;
        for (uint c = 0; c < 3; ++c) {
            // Unpremultiplied colour would amplify noise where alpha is tiny;
            // premultiplied is what composites, so that is what is compared.
            const int d = int(code(pa[c])) - int(code(pb[c]));
            worst = max(worst, uint(abs(d)));
        }
        worst = max(worst, uint(abs(int(round(saturate(pa.w) * 255.0)) - int(round(saturate(pb.w) * 255.0)))));
        histogram[row + worst] += 1;
    }
}
