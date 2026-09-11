// Copyright (c) 2026 lucabRTrender contributors.
//
// A primvar's values onto the device as one float4 per element, its indices
// (if it is indexed) resolved there too: element i is values[indices[i]].
// Float, half or double components, one to four of them; what is missing
// reads (0, 0, 0, 1).
module primvar_expand;
import lrt.common.packing;

struct ExpandParams {
    uint count;        // elements out
    uint components;
    uint kind;         // 1 float, 2 half, 3 double
    uint indexed;
    uint values;       // elements in the stream
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>       words;
StructuredBuffer<int>        indices;
RWStructuredBuffer<float4>   expanded;
ConstantBuffer<ExpandParams> params;

float fromDouble(uint lo, uint hi) {
    const uint biased = (hi >> 20) & 0x7FFu;
    const float sign = (hi >> 31) != 0 ? -1.0 : 1.0;
    if (biased == 0) {
        return 0.0 * sign;
    }
    if (biased == 0x7FFu) {
        return sign * 3.402823e38;
    }
    const float mantissa = 1.0 + (float(hi & 0xFFFFFu) * 4294967296.0 + float(lo)) / 4503599627370496.0;
    return sign * ldexp(mantissa, float(int(biased) - 1023));
}

float valueAt(uint index) {
    if (params.kind == 2) {
        const uint word = words[index / 2];
        return (index & 1u) == 0 ? lowHalf(word) : highHalf(word);
    }
    if (params.kind == 3) {
        return fromDouble(words[index * 2], words[index * 2 + 1]);
    }
    return bit_cast<float>(words[index]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void primvarExpand(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const int element = params.indexed != 0 ? indices[i] : int(i);
    float4 v = float4(0.0, 0.0, 0.0, 1.0);
    if (element >= 0 && uint(element) < params.values) {
        for (uint c = 0; c < min(params.components, 4u); ++c) {
            v[c] = valueAt(uint(element) * params.components + c);
        }
    }
    expanded[i] = v;
}
