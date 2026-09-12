// Copyright (c) 2026 lucabRTrender contributors.
//
// Whether adaptive sampling's error estimate is truthful. A pixel stops when
// its own relative standard error falls below a target; if that estimate is
// right, the pixel's mean is within a few of those standard errors of a deep
// reference. Counted: converged pixels whose true error exceeds three standard
// errors (the reference's own noise folded in), and second moments that came
// out below the mean squared, which no sample set can produce.
module adaptive_check;

struct AdaptiveParams {
    uint  pixels;
    float referencePaths;   // the reference's paths a pixel, for its variance
    uint  pad0; uint pad1;
};

StructuredBuffer<float4>   colour;       // the adaptive mean
StructuredBuffer<float4>   reference;
StructuredBuffer<float4>   sum;          // colour * opacity summed; w = opacity summed = N when opaque
StructuredBuffer<float>    sumSquares;   // luminance^2 summed
StructuredBuffer<uint>     done;
StructuredBuffer<float4>   referenceSum;      // the reference's own moments, for the true spread
StructuredBuffer<float>    referenceSquares;
// [converged, beyond 3 own sigma, negative variance, min N, beyond and below, beyond 3 TRUE sigma, own sigma optimistic by 3x]
RWStructuredBuffer<uint>   counts;
RWStructuredBuffer<float>  worst;        // [largest error over sigma]
ConstantBuffer<AdaptiveParams> adaptive;

static const float3 kLum = float3(0.2126, 0.7152, 0.0722);

[shader("compute")]
[numthreads(1, 1, 1)]
void adaptiveCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint converged = 0;
    uint beyond = 0;
    uint negative = 0;
    uint minN = 0xFFFFFFFF;
    uint beyondBelow = 0;
    uint beyondTrue = 0;
    uint optimistic = 0;
    float largest = 0.0;
    float largestTrue = 0.0;
    for (uint i = 0; i < adaptive.pixels; ++i) {
        if (colour[i].w <= 0.0 || done[i] == 0) {
            continue;
        }
        ++converged;
        const float n = max(sum[i].w, 1.0);
        minN = min(minN, uint(n + 0.5));
        const float mean = dot(sum[i].rgb, kLum) / n;
        const float second = sumSquares[i] / n;
        const float spread = second - mean * mean;
        negative += spread < -1.0e-6 * max(second, 1.0e-12) ? 1u : 0u;
        const float variance = max(spread, 0.0) / n;
        // The reference's own variance at its depth, from the same per-sample
        // spread.
        const float referenceVariance = max(spread, 0.0) / adaptive.referencePaths;
        const float sigma = sqrt(variance + referenceVariance);
        const float actual = abs(dot(colour[i].rgb, kLum) - dot(reference[i].rgb, kLum));
        const float over = sigma > 0.0 ? actual / sigma : (actual > 0.0 ? 1.0e9 : 0.0);
        largest = max(largest, over);
        beyond += over > 3.0 ? 1u : 0u;
        beyondBelow += (over > 3.0 && dot(colour[i].rgb, kLum) < dot(reference[i].rgb, kLum)) ? 1u : 0u;
        // The true per-sample spread, from the reference's deeper moments: the
        // pixel's true standard error at its N, plus the reference's own.
        const float rn = max(referenceSum[i].w, 1.0);
        const float rmean = dot(referenceSum[i].rgb, kLum) / rn;
        const float rspread = max(referenceSquares[i] / rn - rmean * rmean, 0.0);
        const float sigmaTrue = sqrt(rspread / n + rspread / rn);
        const float overTrue = sigmaTrue > 0.0 ? actual / sigmaTrue : (actual > 0.0 ? 1.0e9 : 0.0);
        largestTrue = max(largestTrue, overTrue);
        beyondTrue += overTrue > 3.0 ? 1u : 0u;
        // And how optimistic the pixel's own estimate was against the truth.
        optimistic += sqrt(rspread / n) > 3.0 * sqrt(variance) ? 1u : 0u;
    }
    counts[0] = converged;
    counts[1] = beyond;
    counts[2] = negative;
    counts[3] = minN;
    counts[4] = beyondBelow;
    counts[5] = beyondTrue;
    counts[6] = optimistic;
    worst[0] = largest;
    worst[1] = largestTrue;
}
