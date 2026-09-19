// Copyright (c) 2026 lucabRTrender contributors.
//
// The read of a baked per-part visibility, a frame: one thread a gaussian,
// a factor a light, a product over the parts that are not its own, each
// asked in that part's own current frame. splat_visibility.slang bakes what
// this reads and says why it is baked that way.
module splat_visibility_read;
import lrt.common.packing;
import lrt.common.octahedral;
import lrt.light.lights;
import lrt.splat.splat_visibility_common;

StructuredBuffer<VisibilityPart> parts;
StructuredBuffer<uint>           texels;     // two f16 a word
StructuredBuffer<uint>           ambient;    // two f16 a word: a probe's mean over its directions

// ----------------------------------------------------------------------------
// 3. The read: one thread a gaussian, a factor a light, a product over parts.

struct FactorParams {
    uint  count;         // gaussians in this cloud
    uint  base;          // its first slot in `factors`
    uint  lights;        // lights a factor is measured for
    uint  partCount;
    uint  categoriesLo;  // this cloud's light linking
    uint  categoriesHi;
    uint  skinned;       // 1: parts move by `skinningXforms`; 0: the bind pose is the pose
    uint  pad0;
    // object -> world, rows of a 3x4.
    float w00; float w01; float w02; float w03;
    float w10; float w11; float w12; float w13;
    float w20; float w21; float w22; float w23;
    // The cloud's space into the skeleton's bind space, rows of a 3x4: what
    // the skinner applies before a joint's transform (`geomBindTransform`).
    // A posed position is A . X_k . B . rest, so rest is B^-1 . X_k^-1 . A^-1
    // . posed; A is the identity for a cloud drawn where its skeleton stands.
    float g00; float g01; float g02; float g03;
    float g10; float g11; float g12; float g13;
    float g20; float g21; float g22; float g23;
};

StructuredBuffer<float4>      positions;      // posed, what the frame draws
StructuredBuffer<uint>        ownPart;        // the part each gaussian is of
StructuredBuffer<float4>      skinningXforms; // 4 rows a joint, row-major, bind -> posed
StructuredBuffer<LightRecord> lights;
RWStructuredBuffer<float>     factors;
ConstantBuffer<FactorParams>  factorParams;

float3 toWorld(float3 p) {
    return float3(factorParams.w00 * p.x + factorParams.w01 * p.y + factorParams.w02 * p.z + factorParams.w03,
                  factorParams.w10 * p.x + factorParams.w11 * p.y + factorParams.w12 * p.z + factorParams.w13,
                  factorParams.w20 * p.x + factorParams.w21 * p.y + factorParams.w22 * p.z + factorParams.w23);
}

float3x3 inverse3(float3x3 m) {
    const float3 c0 = float3(m[0][0], m[1][0], m[2][0]);
    const float3 c1 = float3(m[0][1], m[1][1], m[2][1]);
    const float3 c2 = float3(m[0][2], m[1][2], m[2][2]);
    const float3 r0 = cross(c1, c2);
    const float3 r1 = cross(c2, c0);
    const float3 r2 = cross(c0, c1);
    const float det = dot(c0, r0);
    const float inv = abs(det) > 1.0e-20 ? 1.0 / det : 0.0;
    return float3x3(r0 * inv, r1 * inv, r2 * inv);
}

/// One texel of a part's field, as a value.
float texelAt(VisibilityPart part, uint3 cell, uint2 uv) {
    const uint probe = (cell.z * part.grid + cell.y) * part.grid + cell.x;
    const uint t = part.offset + probe * part.octave * part.octave + uv.y * part.octave + uv.x;
    const uint word = texels[t >> 1];
    return f16tof32((t & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

/// A probe's mean over every direction, for a light that has none.
float ambientTexel(uint part, VisibilityPart field, uint3 cell) {
    const uint probes = field.grid * field.grid * field.grid;
    const uint at = part * probes + (cell.z * field.grid + cell.y) * field.grid + cell.x;
    const uint word = ambient[at >> 1];
    return f16tof32((at & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

/// The ambient term at a point: trilinear over the probes. Outside the box
/// the part stops nothing.
float ambientAt(uint index, VisibilityPart part, float3 p) {
    const float3 unit = (p - part.boxMin.xyz) / max(part.boxMax.xyz - part.boxMin.xyz, 1.0e-20);
    if (any(unit < 0.0) || any(unit > 1.0)) {
        return 1.0;
    }
    const float gf = float(part.grid);
    const float3 at = clamp(unit * gf - 0.5, 0.0, gf - 1.0);
    const uint3 lo = uint3(floor(at));
    const uint3 hi = min(lo + 1, uint3(part.grid - 1));
    const float3 fr = at - float3(lo);
    float result = 0.0;
    [unroll] for (uint c = 0; c < 8; ++c) {
        const uint3 cell = uint3((c & 1u) != 0u ? hi.x : lo.x, (c & 2u) != 0u ? hi.y : lo.y, (c & 4u) != 0u ? hi.z : lo.z);
        const float wc = ((c & 1u) != 0u ? fr.x : 1.0 - fr.x) * ((c & 2u) != 0u ? fr.y : 1.0 - fr.y) *
                         ((c & 4u) != 0u ? fr.z : 1.0 - fr.z);
        result += wc * ambientTexel(index, part, cell);
    }
    return result;
}

/// The field at a point in a direction, trilinear over the probes and
/// bilinear over the map. Outside the box the part stops nothing.
float fieldAt(VisibilityPart part, float3 p, float3 w) {
    const float3 unit = (p - part.boxMin.xyz) / max(part.boxMax.xyz - part.boxMin.xyz, 1.0e-20);
    if (any(unit < 0.0) || any(unit > 1.0)) {
        return 1.0;
    }
    const float gf = float(part.grid);
    const float3 at = clamp(unit * gf - 0.5, 0.0, gf - 1.0);
    const uint3 lo = uint3(floor(at));
    const uint3 hi = min(lo + 1, uint3(part.grid - 1));
    const float3 fr = at - float3(lo);
    const float of = float(part.octave);
    const float2 ouv = clamp(octEncode(w) * of - 0.5, 0.0, of - 1.0);
    const uint2 ulo = uint2(floor(ouv));
    const uint2 uhi = min(ulo + 1, uint2(part.octave - 1));
    const float2 ufr = ouv - float2(ulo);
    float result = 0.0;
    [unroll] for (uint c = 0; c < 8; ++c) {
        const uint3 cell = uint3((c & 1u) != 0u ? hi.x : lo.x, (c & 2u) != 0u ? hi.y : lo.y, (c & 4u) != 0u ? hi.z : lo.z);
        const float wc = ((c & 1u) != 0u ? fr.x : 1.0 - fr.x) * ((c & 2u) != 0u ? fr.y : 1.0 - fr.y) *
                         ((c & 4u) != 0u ? fr.z : 1.0 - fr.z);
        const float v00 = texelAt(part, cell, ulo);
        const float v10 = texelAt(part, cell, uint2(uhi.x, ulo.y));
        const float v01 = texelAt(part, cell, uint2(ulo.x, uhi.y));
        const float v11 = texelAt(part, cell, uhi);
        result += wc * lerp(lerp(v00, v10, ufr.x), lerp(v01, v11, ufr.x), ufr.y);
    }
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void visFactors(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= factorParams.count) {
        return;
    }
    const uint slot = factorParams.base + i;
    const float3 posed = positions[i].xyz;
    const float3 world = toWorld(posed);
    for (uint k = 0; k < factorParams.lights; ++k) {
        float through = 1.0;
        const LightRecord light = lights[k];
        const bool linked = lightLinked(light.lightCategory, factorParams.categoriesLo, factorParams.categoriesHi) &&
                            (light.flags & kLightShadow) != 0;
        if (linked && light.kind == kLightDome) {
            // A dome comes from everywhere: what cuts it is each part's solid
            // angle from here, which the ambient term holds.
            const uint own = ownPart[i];
            for (uint part = 0; part < factorParams.partCount; ++part) {
                if (part == own) continue;
                const VisibilityPart field = parts[part];
                if (field.grid == 0u) continue;
                float3 p = posed;
                if (factorParams.skinned != 0) {
                    const uint joint = uint(field.boxMin.w);
                    const float4 r0 = skinningXforms[joint * 4];
                    const float4 r1 = skinningXforms[joint * 4 + 1];
                    const float4 r2 = skinningXforms[joint * 4 + 2];
                    const float3x3 back = inverse3(float3x3(r0.xyz, r1.xyz, r2.xyz));
                    const float3 bound = mul(back, posed - float3(r0.w, r1.w, r2.w));
                    const float3x3 bindLinear = float3x3(factorParams.g00, factorParams.g01, factorParams.g02,
                                                         factorParams.g10, factorParams.g11, factorParams.g12,
                                                         factorParams.g20, factorParams.g21, factorParams.g22);
                    p = mul(inverse3(bindLinear), bound - float3(factorParams.g03, factorParams.g13, factorParams.g23));
                }
                through *= ambientAt(part, field, p);
            }
        } else if (linked && light.kind != kLightDome) {
            const LightSample ls = sampleLight(light, world, float3(0.0, 0.0, 1.0), float2(0.5, 0.5));
            if (ls.valid) {
                // The direction in the cloud's own space, which is the space
                // the fields were baked in before any part moved.
                const float3x3 toWorldLinear = float3x3(factorParams.w00, factorParams.w01, factorParams.w02,
                                                        factorParams.w10, factorParams.w11, factorParams.w12,
                                                        factorParams.w20, factorParams.w21, factorParams.w22);
                const float3 wCloud = normalize(mul(inverse3(toWorldLinear), ls.wi));
                // A PART DOES NOT SHADOW ITSELF. Its field, read at one of
                // its own gaussians, is the crowd that gaussian stands in --
                // the same thing that made a shadow ray from a splat black,
                // one level up -- and it would darken every part from every
                // light. What a part's field is for is the other parts: the
                // wing on the body. The part's own local shadowing is the
                // term this gives up (docs/decisions.md).
                const uint own = ownPart[i];
                for (uint part = 0; part < factorParams.partCount; ++part) {
                    if (part == own) {
                        continue;
                    }
                    const VisibilityPart field = parts[part];
                    if (field.grid == 0u) {
                        continue;   // nothing stands on this part: it casts nothing
                    }
                    float3 p = posed;
                    float3 w = wCloud;
                    if (factorParams.skinned != 0) {
                        // Back through this part's transform: where the
                        // gaussian and the light would be if the part had not
                        // moved, which is where the field was baked -- the
                        // cloud's own rest space, so the bind transform the
                        // skinner applied first is undone last.
                        const uint joint = uint(field.boxMin.w);
                        const float4 r0 = skinningXforms[joint * 4];
                        const float4 r1 = skinningXforms[joint * 4 + 1];
                        const float4 r2 = skinningXforms[joint * 4 + 2];
                        const float3x3 linear = float3x3(r0.xyz, r1.xyz, r2.xyz);
                        const float3x3 back = inverse3(linear);
                        const float3 bound = mul(back, posed - float3(r0.w, r1.w, r2.w));
                        const float3 wBound = mul(back, wCloud);
                        const float3x3 bindLinear = float3x3(factorParams.g00, factorParams.g01, factorParams.g02,
                                                             factorParams.g10, factorParams.g11, factorParams.g12,
                                                             factorParams.g20, factorParams.g21, factorParams.g22);
                        const float3x3 unbind = inverse3(bindLinear);
                        p = mul(unbind, bound - float3(factorParams.g03, factorParams.g13, factorParams.g23));
                        w = normalize(mul(unbind, wBound));
                    }
                    through *= fieldAt(field, p, w);
                }
            }
        }
        factors[slot * factorParams.lights + k] = through;
    }
}

