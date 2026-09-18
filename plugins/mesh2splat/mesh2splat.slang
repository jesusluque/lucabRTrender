// Copyright (c) 2026 lucabRTrender contributors.
//
// A port of Electronic Arts' mesh2splat conversion, from its OpenGL pipeline
// to one compute kernel. The algorithm is theirs:
//
//   Copyright (c) 2024 Electronic Arts Inc. All rights reserved.
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are
//   met:
//   1. Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//   2. Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//   3. Neither the name of Electronic Arts, Inc. ("EA") nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//   THIS SOFTWARE IS PROVIDED BY ELECTRONIC ARTS AND ITS CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
//
//   https://github.com/electronicarts/mesh2splat
//
// WHAT THEIR PIPELINE DOES, AND WHAT THIS ONE DOES INSTEAD
//
// Theirs is a geometry shader and a fragment shader. The geometry shader takes
// a triangle, picks the axis its normal points along most, projects it onto
// that plane, and works out the Jacobian of the map from texture coordinates
// to space -- `J = V (UV)^-1`. From the Jacobian come the local frame and two
// of the three sizes: a gaussian is as wide as a texel is, measured in the
// world. The fragment shader then runs once per texel the projected triangle
// covers, samples the maps there, and appends a gaussian to a buffer with an
// atomic counter.
//
// There is no rasteriser here, so the fragments are walked: one thread a
// triangle, over the cells of the projection grid its bounding box covers.
// That is the same set of samples the rasteriser would have produced, at the
// same density -- `resolution` cells across the model's longest side is their
// render target's resolution -- and it keeps the atomic append, which is what
// makes the order of the output nobody's business.
//
// WHAT IS NOT THEIRS: transmission. Their fragment shader samples albedo,
// normal and metallic-roughness. A glass material (`transmission = 1`) would
// come out of it as an opaque white gaussian, so `transmission` and
// `transmissionColour` are taken here as a material constant, and the opacity
// and colour of every gaussian on that mesh answer to them. A gaussian tints;
// it does not refract, and the two are not the same thing.

/// Must match `Mesh2SplatUniforms` in Mesh2Splat.cpp exactly.
struct M2sParams {
    uint  triangles;
    uint  meshWidth;      // the mesh image: 6 float4 an entry, row major
    uint  meshStride;
    uint  recordPixels;   // float4 a splat: 4, or 6 with the PBR channels

    uint  dstWidth;       // where the records go
    uint  dstHeight;
    uint  dstStride;
    uint  maxSplats;

    float4 boundsMin;     // the model's box: xyz, and w unused
    float4 boundsExtent;  // its size, and w its longest side

    uint  resolution;     // cells across that longest side
    uint  maxCells;       // most cells one triangle may walk
    float sigmaX;         // 0.65 in their shader
    float sigmaY;

    float flatness;       // the third size: their log(1e-7)
    float opacity;
    float transmission;   // ours: 1 - this is what a gaussian keeps
    uint  useNormalMap;

    float4 materialColour;    // multiplied into the albedo; w unused
    float4 transmissionColour;

    float minOpacity;     // what a fully transmitting material still keeps
    uint  pad1;
    uint  pad2;
    uint  pad3;

    uint  hasAlbedo;
    uint  albedoWidth;
    uint  albedoHeight;
    uint  albedoStride;

    uint  hasNormal;
    uint  normalWidth;
    uint  normalHeight;
    uint  normalStride;

    uint  hasMetallicRoughness;
    uint  mrWidth;
    uint  mrHeight;
    uint  mrStride;
};

StructuredBuffer<float4>   mesh;        // per triangle: (position, u) and (normal, v), three corners
StructuredBuffer<float4>   albedo;
StructuredBuffer<float4>   normalMap;
StructuredBuffer<float4>   metallicRoughness;
RWStructuredBuffer<uint>   counters;    // [wanted, written, degenerate, dropped]
RWStructuredBuffer<float4> records;     // the output plane
ConstantBuffer<M2sParams>  params;

/// The counters start a render, and a buffer the host handed over holds
/// whatever it held before. Cleared in a dispatch of its own so that no thread
/// of the one that counts has to race the clear.
[shader("compute")]
[numthreads(1, 1, 1)]
void m2sClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    counters[0] = 0;
    counters[1] = 0;
    counters[2] = 0;
    counters[3] = 0;
}

/// Where one component of a record lands. A record is several entries long
/// and the row it starts on may not have room for all of them, so each is
/// addressed on its own rather than assuming they sit together.
uint recordAt(uint slot, uint component) {
    const uint entry = slot * params.recordPixels + component;
    const uint x = entry % params.dstWidth;
    const uint y = entry / params.dstWidth;
    return y * params.dstStride + x;
}

float4 meshEntry(uint index) {
    const uint x = index % params.meshWidth;
    const uint y = index / params.meshWidth;
    return mesh[y * params.meshStride + x];
}

/// A texture, wrapped, sampled as the four texels around the point. Row 0 is
/// v = 0: the host writes these images bottom row first, as every image this
/// engine carries is.
float4 sampleWrapped(StructuredBuffer<float4> texture, uint width, uint height, uint stride, float2 uv) {
    if (width == 0 || height == 0) {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    const float2 wrapped = uv - floor(uv);
    const float x = wrapped.x * float(width) - 0.5;
    const float y = wrapped.y * float(height) - 0.5;
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const uint x1 = uint((x0 + 1 + int(width)) % int(width));
    const uint y1 = uint((y0 + 1 + int(height)) % int(height));
    const uint xa = uint((x0 + int(width)) % int(width));
    const uint ya = uint((y0 + int(height)) % int(height));
    const float4 c00 = texture[ya * stride + xa];
    const float4 c10 = texture[ya * stride + x1];
    const float4 c01 = texture[y1 * stride + xa];
    const float4 c11 = texture[y1 * stride + x1];
    return lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
}

/// The rotation of a frame, as a quaternion in w, x, y, z -- the order the
/// records carry and the engine's decode expects.
float4 quaternionOfFrame(float3 x, float3 y, float3 z) {
    const float trace = x.x + y.y + z.z;
    float4 q;
    if (trace > 0.0) {
        const float s = sqrt(trace + 1.0) * 2.0;
        q = float4(0.25 * s, (y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s);
    } else if (x.x > y.y && x.x > z.z) {
        const float s = sqrt(1.0 + x.x - y.y - z.z) * 2.0;
        q = float4((y.z - z.y) / s, 0.25 * s, (y.x + x.y) / s, (z.x + x.z) / s);
    } else if (y.y > z.z) {
        const float s = sqrt(1.0 + y.y - x.x - z.z) * 2.0;
        q = float4((z.x - x.z) / s, (y.x + x.y) / s, 0.25 * s, (z.y + y.z) / s);
    } else {
        const float s = sqrt(1.0 + z.z - x.x - y.y) * 2.0;
        q = float4((x.y - y.x) / s, (z.x + x.z) / s, (z.y + y.z) / s, 0.25 * s);
    }
    const float length2 = dot(q, q);
    return length2 > 1.0e-20 ? q / sqrt(length2) : float4(1.0, 0.0, 0.0, 0.0);
}

/// Which pair of axes a triangle is walked in: the two the normal points
/// along least, which is their triplanar choice written as a lookup.
uint2 projectionAxes(float3 normal) {
    const float3 a = abs(normal);
    if (a.x >= a.y && a.x >= a.z) {
        return uint2(1, 2);   // x dominant: walk y and z
    }
    if (a.y >= a.z) {
        return uint2(0, 2);
    }
    return uint2(0, 1);
}

float axisOf(float3 v, uint axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

/// A point in the plane it is drawn in: relative to the model's box, over the
/// wider of that plane's two ranges, which is their `orthogonalUvs`.
float2 orthogonalUv(float3 p, uint2 axes, float range) {
    return float2((axisOf(p, axes.x) - axisOf(params.boundsMin.xyz, axes.x)) / range,
                  (axisOf(p, axes.y) - axisOf(params.boundsMin.xyz, axes.y)) / range);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sEmit(uint3 tid: SV_DispatchThreadID) {
    const uint triangle = tid.x;
    if (triangle >= params.triangles) {
        return;
    }
    const uint base = triangle * 6;
    const float4 p0 = meshEntry(base + 0);
    const float4 n0 = meshEntry(base + 1);
    const float4 p1 = meshEntry(base + 2);
    const float4 n1 = meshEntry(base + 3);
    const float4 p2 = meshEntry(base + 4);
    const float4 n2 = meshEntry(base + 5);

    const float3 a = p0.xyz;
    const float3 b = p1.xyz;
    const float3 c = p2.xyz;
    const float3 e1 = b - a;
    const float3 e2 = c - a;
    const float3 crossed = cross(e1, e2);
    const float area2 = length(crossed);
    if (!(area2 > 1.0e-20)) {
        InterlockedAdd(counters[2], 1u);
        return;   // no plane, no frame, nothing to place
    }
    const float3 geometric = crossed / area2;

    // THE PLANE THE TRIANGLE IS DRAWN IN, which is their triplanar choice:
    // the two axes its normal points along least, the position in them taken
    // relative to the model's box and divided by the wider of the two ranges.
    // That is their `orthogonalUvs`, and it is what their render target
    // rasterises -- so a cell of it is `range / resolution` of the world.
    const uint2 axes = projectionAxes(geometric);
    const float rangeU = axisOf(params.boundsExtent.xyz, axes.x);
    const float rangeV = axisOf(params.boundsExtent.xyz, axes.y);
    const float range = max(max(rangeU, rangeV), 1.0e-20);
    const float2 oa = orthogonalUv(a, axes, range);
    const float2 ob = orthogonalUv(b, axes, range);
    const float2 oc = orthogonalUv(c, axes, range);

    // The Jacobian of THAT map to space -- `J = V (O)^-1`, their
    // `computeUv3DJacobian` over `orthogonalUvs` and not over the texture's
    // coordinates. Its columns are how far a step of one unit of the
    // projection moves in the world; a step of one cell is that over the
    // resolution, and a gaussian is `sigma` of one cell.
    const float2 duv1 = ob - oa;
    const float2 duv2 = oc - oa;
    const float determinant = duv1.x * duv2.y - duv2.x * duv1.y;
    if (!(abs(determinant) > 1.0e-24)) {
        InterlockedAdd(counters[2], 1u);
        return;   // edge on to its own projection: no cell of it is covered
    }
    const float inverse = 1.0 / determinant;
    const float3 ju = (e1 * duv2.y - e2 * duv1.y) * inverse;
    const float3 jv = (e2 * duv1.x - e1 * duv2.x) * inverse;
    const float perCell = 1.0 / float(max(params.resolution, 1u));
    const float3 scale = float3(length(ju) * params.sigmaX * perCell,
                                length(jv) * params.sigmaY * perCell, params.flatness);

    // THE FRAME every gaussian of this triangle shares, and theirs: along the
    // triangle's longest edge, with the surface normal as the short axis and
    // the third made square to both. Not along the Jacobian's columns -- the
    // gaussian is a disc in the triangle's plane however that plane was
    // parametrised.
    float3 longest = e1;
    const float3 e3 = c - b;
    if (dot(e2, e2) > dot(e1, e1) && dot(e2, e2) > dot(e3, e3)) {
        longest = e2;
    } else if (dot(e3, e3) > dot(e1, e1) && dot(e3, e3) > dot(e2, e2)) {
        longest = e3;
    }
    const float3 tangent = normalize(longest);

    // The barycentric solve, in the plane the triangle is walked in.
    const float2 edge1 = ob - oa;
    const float2 edge2 = oc - oa;
    const float twice = edge1.x * edge2.y - edge2.x * edge1.y;
    const float inverseTwice = 1.0 / twice;
    const float2 lowest = min(oa, min(ob, oc));
    const float2 highest = max(oa, max(ob, oc));
    const float cell = perCell;
    const int2 first = int2(floor(lowest / cell));
    const int2 last = int2(floor(highest / cell));
    const uint2 span = uint2(max(last.x - first.x + 1, 1), max(last.y - first.y + 1, 1));
    const uint cells = min(span.x * span.y, max(params.maxCells, 1u));
    if (span.x * span.y > params.maxCells) {
        InterlockedAdd(counters[3], span.x * span.y - params.maxCells);
    }

    for (uint k = 0; k < cells; ++k) {
        const uint cx = k % span.x;
        const uint cy = k / span.x;
        const float2 at = (float2(float(first.x + int(cx)), float(first.y + int(cy))) + 0.5) * cell;
        const float2 relative = at - oa;
        const float beta = (relative.x * edge2.y - edge2.x * relative.y) * inverseTwice;
        const float gamma = (edge1.x * relative.y - relative.x * edge1.y) * inverseTwice;
        const float alpha = 1.0 - beta - gamma;
        if (alpha < 0.0 || beta < 0.0 || gamma < 0.0) {
            continue;   // the cell's centre is outside this triangle
        }
        const float3 position = a * alpha + b * beta + c * gamma;
        const float2 uv = float2(p0.w, n0.w) * alpha + float2(p1.w, n1.w) * beta +
                          float2(p2.w, n2.w) * gamma;
        float3 normal = normalize(n0.xyz * alpha + n1.xyz * beta + n2.xyz * gamma);
        if (!(dot(normal, normal) > 0.5)) {
            normal = geometric;
        }

        float3 colour = params.materialColour.xyz;
        if (params.hasAlbedo != 0) {
            const float4 texel = sampleWrapped(albedo, params.albedoWidth, params.albedoHeight,
                                               params.albedoStride, uv);
            colour *= texel.xyz;
        }
        float metallic = 0.1;
        float roughness = 0.5;
        if (params.hasMetallicRoughness != 0) {
            // glTF's packing, which is what their shader reads: blue is
            // metallic, green is roughness.
            const float4 texel = sampleWrapped(metallicRoughness, params.mrWidth, params.mrHeight,
                                               params.mrStride, uv);
            metallic = texel.z;
            roughness = texel.y;
        }
        float3 shading = normal;
        if (params.hasNormal != 0) {
            const float4 texel = sampleWrapped(normalMap, params.normalWidth, params.normalHeight,
                                               params.normalStride, uv);
            const float3 fromMap = normalize(texel.xyz * 2.0 - 1.0);
            const float3 bitangent = normalize(cross(normal, tangent));
            const float3 straight = normalize(cross(bitangent, normal));
            shading = normalize(straight * fromMap.x + bitangent * fromMap.y + normal * fromMap.z);
        }

        // Ours, not theirs: a transmitting material keeps less of its
        // opacity and takes its transmission colour. Not all of it, though:
        // `1 - transmission` is nothing at all for glass, and a gaussian with
        // no opacity is not a translucent gaussian but an absent one -- the
        // export drops it. So a fully transmitting material keeps
        // `minOpacity`, and what stands behind the glass is dimmed by that
        // much rather than seen through it. A tint, not a lens.
        const float keptOpacity = params.opacity * lerp(1.0, params.minOpacity, params.transmission);
        colour = lerp(colour, colour * params.transmissionColour.xyz, params.transmission);

        const float3 shortAxis = params.useNormalMap != 0 ? shading : normal;
        float3 alongU = tangent - shortAxis * dot(shortAxis, tangent);
        if (!(dot(alongU, alongU) > 1.0e-20)) {
            continue;   // the frame collapsed: no rotation to write
        }
        alongU = normalize(alongU);
        const float3 alongV = cross(shortAxis, alongU);
        const float4 rotation = quaternionOfFrame(alongU, alongV, shortAxis);

        // The slot is taken whether or not there is room for it, so that the
        // count of what was wanted survives a budget too small -- which is
        // what lets the host say how much room to give next time. Written is
        // counted separately, and their `u_maxGaussians` discard is the
        // `continue`.
        uint slot = 0;
        InterlockedAdd(counters[0], 1u, slot);
        if (slot >= params.maxSplats) {
            continue;
        }
        InterlockedAdd(counters[1], 1u);
        records[recordAt(slot, 0)] = float4(position, keptOpacity);
        records[recordAt(slot, 1)] = float4(scale, 0.0);
        records[recordAt(slot, 2)] = rotation;
        records[recordAt(slot, 3)] = float4(colour, 0.0);
        if (params.recordPixels >= 6) {
            records[recordAt(slot, 4)] = float4(shading, metallic);
            records[recordAt(slot, 5)] = float4(roughness, params.transmission, uv.x, uv.y);
        }
    }
}
