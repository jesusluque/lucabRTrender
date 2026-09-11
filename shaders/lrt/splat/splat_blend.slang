// Copyright (c) 2026 lucabRTrender contributors.
//
// Per pixel: its tile's splats, nearest first, until the pixel is opaque.
module splat_blend;
import lrt.common.color;
import frame;

StructuredBuffer<uint>      ranges;
StructuredBuffer<uint>      pairSplats;
StructuredBuffer<ProjRecord> proj;
// What is under the splats: a rasterised point layer (or nothing: bound to a
// placeholder with params.hasUnder == 0). Linear, premultiplied, bottom row first.
StructuredBuffer<float4>    underColour;
StructuredBuffer<float>     underDepth;
RWStructuredBuffer<float4>  colour;   // width * height, premultiplied, linear
RWStructuredBuffer<float>   depth;    // width * height
ConstantBuffer<FrameParams> params;

/// The walk, with `composite` a constant at each call site: the splat-only
/// entry point folds the disc and under-layer branches away. Both cost the
/// blend -- the hottest loop in the engine -- half as much again when they
/// are live in a frame that has neither.
void blendPixel(uint3 tid, bool composite) {
    const uint px = tid.x;
    const uint py = tid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    const uint tile = (py / kTile) * params.tilesX + px / kTile;
    const uint start = ranges[tile * 2];
    const uint end = ranges[tile * 2 + 1];
    const float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);

    const uint at = py * params.width + px;
    const bool under = composite && params.hasUnder != 0;
    const float underZ = under ? underDepth[at] : 0.0;
    const bool underHit = under && underZ > 0.0;

    float3 rgb = float3(0.0);
    float transmittance = 1.0;
    float depthSum = 0.0;
    float depthAt = -1.0;
    // What ends the walk from behind: the raster under-layer, or the first
    // opaque disc. Either is a linear colour composited under the splats'
    // (linearised) contribution -- the same arithmetic whichever route drew
    // the points, which is what lets the two routes agree.
    bool opaqueHit = false;
    float4 opaqueColour = float4(0.0);
    float opaqueZ = 0.0;
    for (uint k = start; k < end; ++k) {
        const ProjRecord record = proj[pairSplats[k]];
        const float4 screen = record.screen;
        if (underHit && screen.z >= underZ) {
            break;
        }
        const float2 d = pixel - screen.xy;
        const float4 conic = record.conic;
        const float power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
        if (power > screen.w) {
            continue;
        }
        if (composite && record.colour.w > 0.5) {
            opaqueHit = true;
            opaqueColour = float4(record.colour.rgb, 1.0);
            opaqueZ = screen.z;
            break;
        }
        const float alpha = min(0.99, conic.w * exp(-0.5 * power));
        if (alpha < 1.0 / 255.0) {
            continue;
        }
        const float weight = transmittance * alpha;
        rgb += weight * record.colour.xyz;
        depthSum += weight * screen.z;
        transmittance *= 1.0 - alpha;
        if (depthAt < 0.0 && 1.0 - transmittance >= params.depthThreshold) {
            depthAt = screen.z;
        }
        if (transmittance < 1e-4) {
            break;
        }
    }
    if (!opaqueHit && underHit) {
        opaqueHit = true;
        opaqueColour = underColour[at];
        opaqueZ = underZ;
    }

    const float coverage = 1.0 - transmittance;
    // Blended in the encoded space the cloud was trained in; the finished
    // splat contribution is made linear. Linearising each splat instead turned
    // letters red and skies electric in openFXplayer.
    float3 linearRgb = rgb;
    if (params.linearise != 0 && coverage > 1e-6) {
        linearRgb = srgbToLinear(rgb / coverage) * coverage;
    }
    float4 behind = float4(params.bgR, params.bgG, params.bgB, params.bgA);
    if (opaqueHit) {
        behind = opaqueColour;
        depthSum += transmittance * opaqueZ;
        if (depthAt < 0.0) {
            depthAt = opaqueZ;
        }
    }
    const float total = opaqueHit ? 1.0 : coverage;
    colour[at] = float4(linearRgb + transmittance * behind.rgb, coverage + transmittance * behind.w);
    depth[at] = params.depthMode == 0 ? (total > 1e-6 ? depthSum / total : 0.0) : max(depthAt, 0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlend(uint3 tid: SV_DispatchThreadID) {
    blendPixel(tid, false);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlendComposite(uint3 tid: SV_DispatchThreadID) {
    blendPixel(tid, true);
}
