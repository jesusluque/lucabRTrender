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

    uint  countWidth;     // the per-triangle cell counts, one uint a triangle
    float metallic;       // the material's own, multiplied into the map where there is one
    float roughness;
    uint  countStride;

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

    uint  hasInfluences;  // the joints a corner is carried by, same shape as the mesh
    uint  pad2;
    uint  pad3;
    uint  pad4;
};

StructuredBuffer<float4>   mesh;        // per triangle: (position, u) and (normal, v), three corners
StructuredBuffer<float4>   albedo;
StructuredBuffer<float4>   normalMap;
StructuredBuffer<float4>   metallicRoughness;
StructuredBuffer<float4>   influences;  // 2 entries a corner: (joint, weight) twice
RWStructuredBuffer<uint>   counters;    // [wanted, written, degenerate, dropped]
RWStructuredBuffer<uint>   cellCounts;  // a triangle's covered cells, then where they start
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

/// A triangle's entry in the cell-count picture. Four to a pixel, addressed
/// the way every other picture here is.
uint countAt(uint triangle) {
    const uint pixel = triangle / 4;
    const uint x = pixel % max(params.countWidth, 1u);
    const uint y = pixel / max(params.countWidth, 1u);
    return (y * params.countStride + x) * 4 + (triangle % 4);
}

float4 meshEntry(uint index) {
    const uint x = index % params.meshWidth;
    const uint y = index / params.meshWidth;
    return mesh[y * params.meshStride + x];
}

/// The joints picture is the mesh picture's shape, entry for entry, which is
/// what saves it a second set of dimensions.
float4 influenceEntry(uint index) {
    const uint x = index % params.meshWidth;
    const uint y = index / params.meshWidth;
    return influences[y * params.meshStride + x];
}

/// THE JOINTS A GAUSSIAN IS CARRIED BY: the barycentric blend of the three
/// corners', which is what interpolating the skin means.
///
/// Twelve pairs go in and four come out, because four is what the cloud
/// carries: a joint already there gains the weight, an empty slot takes it,
/// and otherwise it displaces the lightest. The four are renormalised, so a
/// gaussian is carried entirely however its triangle was authored.
void blendInfluences(uint triangle, float3 bary, out float4 joints, out float4 weights) {
    joints = float4(0.0);
    weights = float4(0.0);
    for (uint corner = 0; corner < 3; ++corner) {
        const uint at = triangle * 6 + corner * 2;
        const float4 low = influenceEntry(at);
        const float4 high = influenceEntry(at + 1);
        float2 pairs[4];
        pairs[0] = low.xy;
        pairs[1] = low.zw;
        pairs[2] = high.xy;
        pairs[3] = high.zw;
        for (uint i = 0; i < 4; ++i) {
            const float weight = pairs[i].y * bary[corner];
            if (!(weight > 0.0)) {
                continue;
            }
            const float joint = pairs[i].x;
            bool placed = false;
            for (uint slot = 0; slot < 4; ++slot) {
                if (weights[slot] > 0.0 && joints[slot] == joint) {
                    weights[slot] += weight;
                    placed = true;
                    break;
                }
            }
            if (placed) {
                continue;
            }
            uint lightest = 0;
            for (uint slot = 1; slot < 4; ++slot) {
                if (weights[slot] < weights[lightest]) {
                    lightest = slot;
                }
            }
            if (weight > weights[lightest]) {
                joints[lightest] = joint;
                weights[lightest] = weight;
            }
        }
    }
    const float sum = weights.x + weights.y + weights.z + weights.w;
    weights = sum > 0.0 ? weights / sum : float4(1.0, 0.0, 0.0, 0.0);
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
/// EVERYTHING A TRIANGLE DECIDES FOR EVERY GAUSSIAN THAT STANDS ON IT,
/// worked out in one place so that the pass that counts and the pass that
/// writes cannot disagree about a single bit.
///
/// Why there are two passes at all: the slot used to be taken with
/// `InterlockedAdd`, and this file said so -- "the order of the output is
/// nobody's business". Two runs over the same mesh gave the same gaussians in
/// a different order, so "gaussian `i` is the same gaussian" was false before
/// any animation was in the picture. Counting first and then writing at
/// `where this triangle starts + k` makes the order the mesh's own, which is
/// what lets a gaussian be followed from one frame to the next -- and it is
/// also what makes a budget too small keep the *first* splats rather than
/// whichever ones won a race.
struct M2sTriangle {
    bool   valid;          ///< a plane, and a Jacobian over its projection
    float4 p0, n0, p1, n1, p2, n2;
    float3 a, b, c;
    float3 geometric;
    float3 tangent;
    float3 scale;
    float2 oa;
    float2 edge1, edge2;
    float  inverseTwice;
    float  cell;
    int2   first;
    uint2  span;
    uint   cells;          ///< walked, after the per-triangle bound
    uint   beyond;         ///< what that bound left out
};

M2sTriangle m2sTriangleOf(uint triangle) {
    M2sTriangle t;
    t.valid = false;
    t.beyond = 0;
    t.cells = 0;
    t.span = uint2(1, 1);
    t.first = int2(0, 0);
    t.cell = 1.0;
    t.oa = float2(0.0);
    t.edge1 = float2(0.0);
    t.edge2 = float2(0.0);
    t.inverseTwice = 0.0;
    t.geometric = float3(0.0, 0.0, 1.0);
    t.tangent = float3(1.0, 0.0, 0.0);
    t.scale = float3(0.0);

    const uint base = triangle * 6;
    t.p0 = meshEntry(base + 0);
    t.n0 = meshEntry(base + 1);
    t.p1 = meshEntry(base + 2);
    t.n1 = meshEntry(base + 3);
    t.p2 = meshEntry(base + 4);
    t.n2 = meshEntry(base + 5);

    t.a = t.p0.xyz;
    t.b = t.p1.xyz;
    t.c = t.p2.xyz;
    const float3 e1 = t.b - t.a;
    const float3 e2 = t.c - t.a;
    const float3 crossed = cross(e1, e2);
    const float area2 = length(crossed);
    if (!(area2 > 1.0e-20)) {
        return t;   // no plane, no frame, nothing to place
    }
    t.geometric = crossed / area2;

    // THE PLANE THE TRIANGLE IS DRAWN IN, which is their triplanar choice:
    // the two axes its normal points along least, the position in them taken
    // relative to the model's box and divided by the wider of the two ranges.
    // That is their `orthogonalUvs`, and it is what their render target
    // rasterises -- so a cell of it is `range / resolution` of the world.
    const uint2 axes = projectionAxes(t.geometric);
    const float rangeU = axisOf(params.boundsExtent.xyz, axes.x);
    const float rangeV = axisOf(params.boundsExtent.xyz, axes.y);
    const float range = max(max(rangeU, rangeV), 1.0e-20);
    t.oa = orthogonalUv(t.a, axes, range);
    const float2 ob = orthogonalUv(t.b, axes, range);
    const float2 oc = orthogonalUv(t.c, axes, range);

    // The Jacobian of THAT map to space -- `J = V (O)^-1`, their
    // `computeUv3DJacobian` over `orthogonalUvs` and not over the texture's
    // coordinates. Its columns are how far a step of one unit of the
    // projection moves in the world; a step of one cell is that over the
    // resolution, and a gaussian is `sigma` of one cell.
    const float2 duv1 = ob - t.oa;
    const float2 duv2 = oc - t.oa;
    const float determinant = duv1.x * duv2.y - duv2.x * duv1.y;
    if (!(abs(determinant) > 1.0e-24)) {
        return t;   // edge on to its own projection: no cell of it is covered
    }
    const float inverse = 1.0 / determinant;
    const float3 ju = (e1 * duv2.y - e2 * duv1.y) * inverse;
    const float3 jv = (e2 * duv1.x - e1 * duv2.x) * inverse;
    const float perCell = 1.0 / float(max(params.resolution, 1u));
    const float alongU = length(ju) * params.sigmaX * perCell;
    const float alongV = length(jv) * params.sigmaY * perCell;
    // THE THIRD SIZE IS A FRACTION OF THE OTHER TWO, not a length.
    //
    // mesh2splat writes 1e-7 there, a number in the model's own units, while
    // the two sizes across the surface are in cells -- so how thin a gaussian
    // is depends on how big the model happens to be. The chess pawn is 66 mm
    // across and traced fine; the Khronos fox is a hundred units long, which
    // makes the same 1e-7 fifteen hundred times more extreme, and the ray
    // tracer -- which integrates density along the ray rather than projecting
    // an ellipse as the rasteriser does -- saw a ghost where the rasteriser
    // saw a fox. A fraction is the same number for every model.
    t.scale = float3(alongU, alongV, min(alongU, alongV) * params.flatness);

    // THE FRAME every gaussian of this triangle shares, and theirs: along the
    // triangle's longest edge, with the surface normal as the short axis and
    // the third made square to both. Not along the Jacobian's columns -- the
    // gaussian is a disc in the triangle's plane however that plane was
    // parametrised.
    float3 longest = e1;
    const float3 e3 = t.c - t.b;
    if (dot(e2, e2) > dot(e1, e1) && dot(e2, e2) > dot(e3, e3)) {
        longest = e2;
    } else if (dot(e3, e3) > dot(e1, e1) && dot(e3, e3) > dot(e2, e2)) {
        longest = e3;
    }
    t.tangent = normalize(longest);

    // The barycentric solve, in the plane the triangle is walked in.
    t.edge1 = ob - t.oa;
    t.edge2 = oc - t.oa;
    const float twice = t.edge1.x * t.edge2.y - t.edge2.x * t.edge1.y;
    t.inverseTwice = 1.0 / twice;
    const float2 lowest = min(t.oa, min(ob, oc));
    const float2 highest = max(t.oa, max(ob, oc));
    t.cell = perCell;
    t.first = int2(floor(lowest / t.cell));
    const int2 last = int2(floor(highest / t.cell));
    t.span = uint2(max(last.x - t.first.x + 1, 1), max(last.y - t.first.y + 1, 1));
    const uint wanted = t.span.x * t.span.y;
    t.cells = min(wanted, max(params.maxCells, 1u));
    t.beyond = wanted > params.maxCells ? wanted - params.maxCells : 0u;
    t.valid = true;
    return t;
}

/// The k-th cell of a triangle's walk: where it is in barycentric
/// coordinates, and whether its centre is inside the triangle at all.
bool m2sCellInside(M2sTriangle t, uint k, out float3 bary) {
    const uint cx = k % t.span.x;
    const uint cy = k / t.span.x;
    const float2 at = (float2(float(t.first.x + int(cx)), float(t.first.y + int(cy))) + 0.5) * t.cell;
    const float2 relative = at - t.oa;
    const float beta = (relative.x * t.edge2.y - t.edge2.x * relative.y) * t.inverseTwice;
    const float gamma = (t.edge1.x * relative.y - relative.x * t.edge1.y) * t.inverseTwice;
    const float alpha = 1.0 - beta - gamma;
    bary = float3(alpha, beta, gamma);
    return !(alpha < 0.0 || beta < 0.0 || gamma < 0.0);
}

/// How many gaussians this triangle stands, one thread a triangle.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCount(uint3 tid: SV_DispatchThreadID) {
    const uint triangle = tid.x;
    if (triangle >= params.triangles) {
        return;
    }
    const M2sTriangle t = m2sTriangleOf(triangle);
    if (!t.valid) {
        InterlockedAdd(counters[2], 1u);
        cellCounts[countAt(triangle)] = 0;
        return;
    }
    if (t.beyond > 0) {
        InterlockedAdd(counters[3], t.beyond);
    }
    uint inside = 0;
    for (uint k = 0; k < t.cells; ++k) {
        float3 bary;
        if (m2sCellInside(t, k, bary)) {
            ++inside;
        }
    }
    cellCounts[countAt(triangle)] = inside;
}

/// Where each triangle's gaussians start: the exclusive running sum of the
/// counts, written back over them.
///
/// One workgroup, three phases: each thread adds up a slice, thread zero runs
/// the two hundred and fifty six slice totals into a running sum, and each
/// thread walks its slice again laying its own running sum down. A serial
/// scan over the whole mesh in one thread was the other option and is what
/// this avoids: the pawn has 42892 triangles and the fox will have more.
static const uint kScanThreads = 256;
groupshared uint gScanPartial[kScanThreads];

[shader("compute")]
[numthreads(256, 1, 1)]
void m2sScan(uint3 gid: SV_GroupThreadID) {
    const uint thread = gid.x;
    const uint total = params.triangles;
    const uint slice = (total + kScanThreads - 1) / kScanThreads;
    const uint from = min(thread * slice, total);
    const uint to = min(from + slice, total);

    uint sum = 0;
    for (uint i = from; i < to; ++i) {
        sum += cellCounts[countAt(i)];
    }
    gScanPartial[thread] = sum;
    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        uint running = 0;
        for (uint t = 0; t < kScanThreads; ++t) {
            const uint here = gScanPartial[t];
            gScanPartial[t] = running;
            running += here;
        }
        counters[0] = running;                              // wanted
        counters[1] = min(running, params.maxSplats);       // written
    }
    GroupMemoryBarrierWithGroupSync();

    uint running = gScanPartial[thread];
    for (uint i = from; i < to; ++i) {
        const uint here = cellCounts[countAt(i)];
        cellCounts[countAt(i)] = running;
        running += here;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sEmit(uint3 tid: SV_DispatchThreadID) {
    const uint triangle = tid.x;
    if (triangle >= params.triangles) {
        return;
    }
    const M2sTriangle t = m2sTriangleOf(triangle);
    if (!t.valid) {
        return;
    }
    const float3 a = t.a;
    const float3 b = t.b;
    const float3 c = t.c;
    const float4 p0 = t.p0;
    const float4 n0 = t.n0;
    const float4 p1 = t.p1;
    const float4 n1 = t.n1;
    const float4 p2 = t.p2;
    const float4 n2 = t.n2;
    const float3 geometric = t.geometric;
    const float3 tangent = t.tangent;
    const float3 scale = t.scale;

    // Where this triangle's gaussians go, which the scan settled. A budget
    // too small then keeps the first of them, in the mesh's own order, and
    // not whichever ones won a race.
    uint slot = cellCounts[countAt(triangle)];

    for (uint k = 0; k < t.cells; ++k) {
        float3 bary;
        if (!m2sCellInside(t, k, bary)) {
            continue;   // the cell's centre is outside this triangle
        }
        if (slot >= params.maxSplats) {
            return;
        }
        const float alpha = bary.x;
        const float beta = bary.y;
        const float gamma = bary.z;
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
        // The material's own values, and the map where there is one: glTF
        // multiplies the two and so does this, which is also what leaves a
        // material with no map at the values it authored. Theirs defaulted to
        // (0.1, 0.5) and ignored the material, so every conversion came out
        // of the same plastic.
        float metallic = params.metallic;
        float roughness = params.roughness;
        if (params.hasMetallicRoughness != 0) {
            // Blue is metallic, green is roughness: their packing.
            const float4 texel = sampleWrapped(metallicRoughness, params.mrWidth, params.mrHeight,
                                               params.mrStride, uv);
            metallic *= texel.z;
            roughness *= texel.y;
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

        // Ours, not theirs: transmission. Their conversion has no channel for
        // it, so glass came out as opaque white. What goes into the record is
        // what the material says -- its opacity, its colour taken towards the
        // transmission colour, and the transmission itself in a channel of
        // its own. What a renderer makes of that is the renderer's: a
        // translucent material is not a transparent one, and lowering the
        // opacity here would say it was.
        const float keptOpacity = params.opacity;
        colour = lerp(colour, colour * params.transmissionColour.xyz, params.transmission);

        const float3 shortAxis = params.useNormalMap != 0 ? shading : normal;
        float3 alongU = tangent - shortAxis * dot(shortAxis, tangent);
        if (!(dot(alongU, alongU) > 1.0e-20)) {
            continue;   // the frame collapsed: no rotation to write
        }
        alongU = normalize(alongU);
        const float3 alongV = cross(shortAxis, alongU);
        const float4 rotation = quaternionOfFrame(alongU, alongV, shortAxis);
        records[recordAt(slot, 0)] = float4(position, keptOpacity);
        records[recordAt(slot, 1)] = float4(scale, 0.0);
        records[recordAt(slot, 2)] = rotation;
        records[recordAt(slot, 3)] = float4(colour, 0.0);
        if (params.recordPixels >= 6) {
            records[recordAt(slot, 4)] = float4(shading, metallic);
            records[recordAt(slot, 5)] = float4(roughness, params.transmission, uv.x, uv.y);
        }
        if (params.recordPixels >= 8 && params.hasInfluences != 0) {
            float4 joints;
            float4 weights;
            blendInfluences(triangle, bary, joints, weights);
            records[recordAt(slot, 6)] = float4(joints.x, weights.x, joints.y, weights.y);
            records[recordAt(slot, 7)] = float4(joints.z, weights.z, joints.w, weights.w);
        }
        ++slot;
    }
}
