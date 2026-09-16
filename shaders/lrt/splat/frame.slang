// Copyright (c) 2026 lucabRTrender contributors.
//
// What every stage of the splat rasteriser agrees on: the frame, the tile
// grid, how a splat's footprint is cut, and how a tile is tested against it.
//
// THE PIPELINE (lrt::render::TileRasterizer drives it)
//
//   project   per splat: EWA covariance, conic, colour, depth key, footprint,
//             and the exact number of tiles the footprint touches
//   sort      splats by depth (32-bit key)                  -- RadixSort
//   gather    tiles-touched in depth order, prefix sum      -- PrefixSum
//   emit      per splat in depth order: one (tile, splat) pair per tile
//   sort      pairs by tile, stably (tile bits only)        -- RadixSort
//   ranges    where each tile's pairs start and end
//   blend     per pixel: its tile's pairs, front to back
//
// Sorting splats by depth once and pairs by tile stably gives every tile its
// splats in depth order, ties in splat order: the same order a full 64-bit
// (tile, depth) sort gives, with a sort of N 32-bit keys and one of P keys of
// ~15 bits, instead of one of P 47-bit keys. Deterministic by construction.
//
// CONVENTIONS (openFXplayer's, measured and documented there)
//   - the kernel's frame: +z forward, x right, y up; screen origin bottom-left;
//     pixel centres at +0.5. The -Z camera of D36 is folded into the view.
//   - square pixels: focalY == focalX unless a window scale says otherwise.
//   - colour is blended in the encoded space the cloud was trained in, and the
//     finished pixel is linearised.
module frame;
import lrt.common.edit;

public static const uint kTile = 16;

public struct FrameParams {
    public uint  width;
    public uint  height;
    public uint  tilesX;
    public uint  tilesY;
    public float focalX;
    public float focalY;
    public float centreX;
    public float centreY;
    public float nearZ;
    public float farZ;
    public uint  orthographic;
    public uint  antialias;
    // Object -> view, rows of a 3x4.
    public float m00; public float m01; public float m02; public float m03;
    public float m10; public float m11; public float m12; public float m13;
    public float m20; public float m21; public float m22; public float m23;
    // The camera in the cloud's own space, for the harmonics' view direction.
    public float eyeX; public float eyeY; public float eyeZ;
    public uint  count;          // splats in this cloud / pairs / ...
    public uint  base;           // first projected slot of this cloud
    public uint  restPerColour;  // harmonics kept: 0, 3, 8, 15
    public uint  shWords;
    public uint  shLimit;        // highest degree to evaluate this frame
    public uint  tileBits;
    public uint  depthMode;      // 0 alpha-weighted mean, 1 first crossing
    public float depthThreshold;
    public float bgR; public float bgG; public float bgB; public float bgA;
    public uint  linearise;      // 1: finished pixel sRGB -> linear
    public uint  hasUnder;       // 1: an opaque layer (raster points) is under the splats
    public uint  relight;        // 1: light these splats rather than show them as baked
    public uint  lightCount;     // lights in `lights`, 0 when nothing is relit
    public uint  categoriesLo;   // this cloud's categories, for a light's link
    public uint  categoriesHi;
    public uint  shadowRays;     // 1: a relit splat is shadowed by the cloud's own proxies
    public uint  shadowLights;   // lights splat_shadow measured a factor for
    public float shadowOffset;   // where its shadow ray starts, in the splat's own sigmas
    public float shadowCut;      // and where the ray gives up
    public EditParams edit;      // this cloud instance's SplatEdit (common/edit.slang)
};

/// What the projection leaves for the blend, one record per splat, so a pair
/// costs the blend one memory access and not three.
public struct ProjRecord {
    public float4 screen;   // centre.xy, view z, cutoff power
    public float4 conic;    // a, b, c, alpha
    public float4 colour;   // rgb (encoded), 0
};

public float3 toView(FrameParams f, float3 p) {
    return float3(f.m00 * p.x + f.m01 * p.y + f.m02 * p.z + f.m03,
                  f.m10 * p.x + f.m11 * p.y + f.m12 * p.z + f.m13,
                  f.m20 * p.x + f.m21 * p.y + f.m22 * p.z + f.m23);
}

public float3x3 viewLinear(FrameParams f) {
    return float3x3(f.m00, f.m01, f.m02, f.m10, f.m11, f.m12, f.m20, f.m21, f.m22);
}

/// The squared Mahalanobis radius a splat is cut at: where its own
/// contribution falls below 1/255, and never past three sigma. The blend cuts
/// at the same radius, so no pixel outside a counted tile could have been
/// touched -- which is what keeps tile edges seamless.
public float cutoffPower(float alpha) {
    return min(2.0 * log(max(255.0 * alpha, 1.0001)), 9.0);
}

/// Minimum over a box of q(d) = a dx^2 + 2 b dx dy + c dy^2, for d measured
/// from the splat's centre. Exact: q is convex, so with the centre outside the
/// box the minimum lies on an edge, and on an edge it is a one-dimensional
/// quadratic minimised in closed form and clamped.
public float minPowerOverBox(float a, float b, float c, float2 lo, float2 hi) {
    if (lo.x <= 0.0 && hi.x >= 0.0 && lo.y <= 0.0 && hi.y >= 0.0) {
        return 0.0;
    }
    float best = 3.4e38;
    // x fixed at lo.x / hi.x, y free in [lo.y, hi.y]
    for (int e = 0; e < 2; ++e) {
        const float x = e == 0 ? lo.x : hi.x;
        const float y = c > 1e-12 ? clamp(-b * x / c, lo.y, hi.y) : lo.y;
        best = min(best, a * x * x + 2.0 * b * x * y + c * y * y);
    }
    for (int e = 0; e < 2; ++e) {
        const float y = e == 0 ? lo.y : hi.y;
        const float x = a > 1e-12 ? clamp(-b * y / a, lo.x, hi.x) : lo.x;
        best = min(best, a * x * x + 2.0 * b * x * y + c * y * y);
    }
    return best;
}

/// Does tile (tx, ty) hold a pixel centre inside the footprint?
public bool tileTouched(FrameParams f, float2 centre, float3 conic, float cutoff, uint tx, uint ty) {
    const float2 first = float2(float(tx * kTile) + 0.5, float(ty * kTile) + 0.5);
    const float2 last = float2(float(min((tx + 1) * kTile, f.width) - 1) + 0.5,
                               float(min((ty + 1) * kTile, f.height) - 1) + 0.5);
    return minPowerOverBox(conic.x, conic.y, conic.z, first - centre, last - centre) <= cutoff;
}

/// The footprint's axis-aligned extent, half-widths in pixels, from the 2D
/// covariance (the conic's inverse): sqrt(cutoff * sigma_xx), sqrt(cutoff * sigma_yy).
public float2 footprintExtent(float3 conic, float cutoff) {
    const float det = conic.x * conic.z - conic.y * conic.y;
    const float inv = det > 1e-20 ? 1.0 / det : 0.0;
    return sqrt(max(float2(cutoff * conic.z * inv, cutoff * conic.x * inv), 0.0));
}

/// The tile rectangle [lo, hi] (inclusive) of a footprint; lo > hi when it is off screen.
public void tileRect(FrameParams f, float2 centre, float2 extent, out uint4 rect) {
    const float x1 = centre.x - extent.x;
    const float y1 = centre.y - extent.y;
    const float x2 = centre.x + extent.x;
    const float y2 = centre.y + extent.y;
    if (x2 < 0.0 || y2 < 0.0 || x1 >= float(f.width) || y1 >= float(f.height)) {
        rect = uint4(1, 1, 0, 0);
        return;
    }
    rect = uint4(uint(max(x1, 0.0)) / kTile, uint(max(y1, 0.0)) / kTile,
                 uint(min(x2, float(f.width) - 1.0)) / kTile,
                 uint(min(y2, float(f.height) - 1.0)) / kTile);
}
