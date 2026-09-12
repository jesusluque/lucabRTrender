// Copyright (c) 2026 lucabRTrender contributors.
//
// Hydra instancing on the device: an instancer level's transforms from its
// primvars, levels composed, and the records every technique reads -- for as
// many instances as a scene has, with nothing per instance on the host.
//
// One level, as Storm composes it (hdSt/shaders/instancing.glslfx) and
// hdEmbree does (plugin/hdEmbree/instancer.cpp), in column-vector form:
//   instancerTransform * translate(t) * rotate(q) * scale(s) * instanceTransform
// with q = (ix, iy, iz, real), GfQuat's own layout, and instanceTransform a
// row-major GfMatrix4d read as its transpose. Levels nest outward:
//   final[i * inner + j] = parent[i] * level[j]
// and a record's object to world is final * the prototype's own transform.
module instancing;
import lrt.common.packing;
import scene_types;

struct InstancingParams {
    uint  count;          // outputs
    uint  inner;          // compose: instances per parent instance
    uint  translations;   // stream kinds: 0 none, 1 float, 2 half, 3 double
    uint  rotations;
    uint  scales;
    uint  transforms;
    uint  mesh;           // records
    uint  primId;
    uint  first;          // records: the sets' first record
    uint  flags;
    uint  sets;           // records: how many sets
    uint  pad0;
    float c0r; float c0g; float c0b; float c0a;   // records: displayColor, opacity
    // a 3x4 in rows: the level's instancer transform, or (records) the prototype's own
    float m00; float m01; float m02; float m03;
    float m10; float m11; float m12; float m13;
    float m20; float m21; float m22; float m23;
    // records: world to view
    float v00; float v01; float v02; float v03;
    float v10; float v11; float v12; float v13;
    float v20; float v21; float v22; float v23;
};

StructuredBuffer<uint>              indices;          // level: which primvar element each instance takes
StructuredBuffer<uint>              translationWords;
StructuredBuffer<uint>              rotationWords;
StructuredBuffer<uint>              scaleWords;
StructuredBuffer<uint>              transformWords;
StructuredBuffer<float4>            parentRows;       // compose: 3 per parent instance
StructuredBuffer<float4>            levelRows;        // compose: 3 per level instance
RWStructuredBuffer<float4>          rows;             // level, compose: 3 per output
RWStructuredBuffer<InstanceRecord>  records;

/// An instanced mesh: its instances' chain rows are a run of `setRows`, and
/// its records a run after the scene's single instances (GpuScene.cpp).
struct SetRecord {
    float4 prototype0; float4 prototype1; float4 prototype2;   // the prototype's own transform, 3x4 rows
    float4 colour;       // displayColor rgb, displayOpacity
    uint   first;        // its first instance among the sets
    uint   count;
    uint   mesh;
    uint   primId;
    uint   flags;        // bit 0: double sided
    uint   categoriesLo; // what a light's link is tested against
    uint   categoriesHi;
    uint   pad0;
};
StructuredBuffer<SetRecord>         sets;             // records: in order of `first`
StructuredBuffer<float4>            setRows;          // records: 3 per instance, every set's chain
ConstantBuffer<InstancingParams>    params;

/// A float64 from its two words (little-endian), rounded to float -- Metal
/// has no double.
float fromDouble(uint lo, uint hi) {
    const uint biased = (hi >> 20) & 0x7FFu;
    const float sign = (hi >> 31) != 0 ? -1.0 : 1.0;
    if (biased == 0) {
        return 0.0 * sign;   // zero, and denormals no float can hold
    }
    if (biased == 0x7FFu) {
        return sign * 3.402823e38;
    }
    // The top 24 mantissa bits are what a float keeps anyway.
    const float mantissa = 1.0 + (float(hi & 0xFFFFFu) * 4294967296.0 + float(lo)) / 4503599627370496.0;
    return sign * ldexp(mantissa, float(int(biased) - 1023));
}

float streamValue(StructuredBuffer<uint> words, uint kind, uint index) {
    if (kind == 2) {
        const uint word = words[index / 2];
        return (index & 1u) == 0 ? lowHalf(word) : highHalf(word);
    }
    if (kind == 3) {
        return fromDouble(words[index * 2], words[index * 2 + 1]);
    }
    return bit_cast<float>(words[index]);
}

float3x4 paramRows(bool view) {
    return view ? float3x4(params.v00, params.v01, params.v02, params.v03, params.v10, params.v11, params.v12,
                           params.v13, params.v20, params.v21, params.v22, params.v23)
                : float3x4(params.m00, params.m01, params.m02, params.m03, params.m10, params.m11, params.m12,
                           params.m13, params.m20, params.m21, params.m22, params.m23);
}

float4x4 full(float3x4 m) {
    return float4x4(m[0], m[1], m[2], float4(0.0, 0.0, 0.0, 1.0));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void instancerLevel(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint e = indices[i];
    float4x4 m = full(paramRows(false));
    if (params.translations != 0) {
        const float3 t = float3(streamValue(translationWords, params.translations, e * 3),
                                streamValue(translationWords, params.translations, e * 3 + 1),
                                streamValue(translationWords, params.translations, e * 3 + 2));
        m = mul(m, float4x4(1, 0, 0, t.x, 0, 1, 0, t.y, 0, 0, 1, t.z, 0, 0, 0, 1));
    }
    if (params.rotations != 0) {
        const float4 q = float4(streamValue(rotationWords, params.rotations, e * 4),
                                streamValue(rotationWords, params.rotations, e * 4 + 1),
                                streamValue(rotationWords, params.rotations, e * 4 + 2),
                                streamValue(rotationWords, params.rotations, e * 4 + 3));
        const float x = q.x, y = q.y, z = q.z, w = q.w;
        const float4x4 r = float4x4(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0,
                                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0,
                                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0,
                                    0, 0, 0, 1);
        m = mul(m, r);
    }
    if (params.scales != 0) {
        const float3 s = float3(streamValue(scaleWords, params.scales, e * 3),
                                streamValue(scaleWords, params.scales, e * 3 + 1),
                                streamValue(scaleWords, params.scales, e * 3 + 2));
        m = mul(m, float4x4(s.x, 0, 0, 0, 0, s.y, 0, 0, 0, 0, s.z, 0, 0, 0, 0, 1));
    }
    if (params.transforms != 0) {
        // Row-major GfMatrix4d read as its transpose: column c of the column
        // matrix is row c of the stored one.
        float4x4 x;
        for (uint r = 0; r < 4; ++r) {
            for (uint c = 0; c < 4; ++c) {
                x[r][c] = streamValue(transformWords, params.transforms, e * 16 + c * 4 + r);
            }
        }
        m = mul(m, x);
    }
    rows[i * 3] = m[0];
    rows[i * 3 + 1] = m[1];
    rows[i * 3 + 2] = m[2];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void instancerCompose(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint i = k / params.inner;
    const uint j = k % params.inner;
    const float4x4 parent = full(float3x4(parentRows[i * 3], parentRows[i * 3 + 1], parentRows[i * 3 + 2]));
    const float4x4 level = full(float3x4(levelRows[j * 3], levelRows[j * 3 + 1], levelRows[j * 3 + 2]));
    const float4x4 m = mul(parent, level);
    rows[k * 3] = m[0];
    rows[k * 3 + 1] = m[1];
    rows[k * 3 + 2] = m[2];
}

/// The set holding instance `i` (counted among the sets): a binary search
/// over the sets' firsts.
uint setOf(uint i) {
    uint lo = 0;
    uint hi = params.sets;
    while (hi - lo > 1) {
        const uint mid = (lo + hi) / 2;
        if (sets[mid].first <= i) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// Every instanced set's records in one dispatch: instance i's chain times
/// its set's prototype, to view, with the normal matrix.
[shader("compute")]
[numthreads(256, 1, 1)]
void instanceRecords(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const SetRecord set = sets[setOf(i)];
    const float4x4 chain = full(float3x4(setRows[i * 3], setRows[i * 3 + 1], setRows[i * 3 + 2]));
    const float4x4 world = mul(chain, full(float3x4(set.prototype0, set.prototype1, set.prototype2)));
    const float4x4 view = mul(full(paramRows(true)), world);
    // The normal matrix: the inverse transpose of view's 3x3, by cofactors.
    const float3 a = view[0].xyz;
    const float3 b = view[1].xyz;
    const float3 c = view[2].xyz;
    const float3 r0 = cross(b, c);
    const float3 r1 = cross(c, a);
    const float3 r2 = cross(a, b);
    const float det = dot(a, r0);
    const float inverse = abs(det) > 1e-30 ? 1.0 / det : 0.0;
    InstanceRecord record;
    record.view0 = view[0];
    record.view1 = view[1];
    record.view2 = view[2];
    record.normal0 = float4(r0 * inverse, 0.0);
    record.normal1 = float4(r1 * inverse, 0.0);
    record.normal2 = float4(r2 * inverse, 0.0);
    record.world0 = world[0];
    record.world1 = world[1];
    record.world2 = world[2];
    record.colour = set.colour;
    record.mesh = set.mesh;
    record.primId = set.primId;
    record.instanceId = i - set.first;
    record.flags = set.flags;
    record.categoriesLo = set.categoriesLo;
    record.categoriesHi = set.categoriesHi;
    record.mask = 0xFF;   // still: in every shutter slice
    record.pointsOffset = 0;
    records[params.first + i] = record;
}
