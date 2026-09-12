// Copyright (c) 2026 lucabRTrender contributors.
//
// Whether a path traced frame's per-pixel errors are independent of each other.
//
// If each pixel draws its own samples, the errors of neighbouring pixels are
// independent and the variance of a block mean falls as 1/blockSize: averaging
// a 2x2 block quarters it, a 4x4 sixteenths it. If pixels share sample sets --
// which is what a GF(2) mix of pixel and sample index would produce -- the
// errors are correlated and block averaging does not buy that reduction. The
// mean is unaffected either way, which is why the ladder's means looked sane
// while its draw-to-draw spread did not.
module block_error;

struct BlockParams {
    uint width;
    uint height;
    uint block;    // 1, 2 or 4
    uint pad0;
};

StructuredBuffer<float4>   a;         // the estimate
StructuredBuffer<float4>   b;         // the reference
RWStructuredBuffer<float>  moments;   // [sum of block means, sum of squares, blocks]
ConstantBuffer<BlockParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void blockError(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint bw = params.width / params.block;
    const uint bh = params.height / params.block;
    float sum = 0.0;
    float squares = 0.0;
    uint blocks = 0;
    for (uint by = 0; by < bh; ++by) {
        for (uint bx = 0; bx < bw; ++bx) {
            float mean = 0.0;
            for (uint y = 0; y < params.block; ++y) {
                for (uint x = 0; x < params.block; ++x) {
                    const uint at = (by * params.block + y) * params.width + bx * params.block + x;
                    const float3 d = a[at].rgb - b[at].rgb;
                    // The signed error of the block, channel mean, so errors
                    // cancel across a block exactly when they are independent.
                    mean += (d.r + d.g + d.b) / 3.0;
                }
            }
            mean /= float(params.block * params.block);
            sum += mean;
            squares += mean * mean;
            ++blocks;
        }
    }
    moments[0] = sum;
    moments[1] = squares;
    moments[2] = float(blocks);
}
