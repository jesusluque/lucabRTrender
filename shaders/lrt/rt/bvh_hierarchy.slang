// Copyright (c) 2026 lucabRTrender contributors.
//
// The hierarchy from sorted Morton codes, every internal node on its own
// (Karras 2012, "Maximizing parallelism in the construction of BVHs, octrees,
// and k-d trees"). Equal codes are told apart by leaf index, as the paper
// suggests, so duplicates build a valid tree.
module bvh_hierarchy;
import bvh_common;

StructuredBuffer<uint>     sortedKeys;
StructuredBuffer<uint>     sortedValues;
RWStructuredBuffer<uint>   children;
RWStructuredBuffer<uint>   leaves;
ConstantBuffer<BvhParams>  params;

uint leadingZeros(uint v) {
    return v == 0 ? 32 : 31 - firstbithigh(v);
}

/// The length of the common prefix of leaves i and j (index-extended keys).
int delta(int i, int j) {
    const int n = int(params.count);
    if (j < 0 || j >= n) {
        return -1;
    }
    const uint a = sortedKeys[i];
    const uint b = sortedKeys[j];
    if (a != b) {
        return int(leadingZeros(a ^ b));
    }
    return 32 + int(leadingZeros(uint(i) ^ uint(j)));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bvhHierarchy(uint3 tid: SV_DispatchThreadID) {
    const uint u = tid.x;
    if (u >= params.count) {
        return;
    }
    // Leaves: the particle each sorted slot holds.
    leaves[params.leafBase + u] = sortedValues[u];
    if (u + 1 >= params.count) {
        return;
    }
    const int i = int(u);
    // Direction of the range this node covers.
    const int d = delta(i, i + 1) - delta(i, i - 1) >= 0 ? 1 : -1;
    const int deltaMin = delta(i, i - d);
    // Upper bound on the range's length, then the exact other end.
    int lengthMax = 2;
    while (delta(i, i + lengthMax * d) > deltaMin) {
        lengthMax *= 2;
    }
    int length = 0;
    for (int t = lengthMax / 2; t >= 1; t /= 2) {
        if (delta(i, i + (length + t) * d) > deltaMin) {
            length += t;
        }
    }
    const int j = i + length * d;
    // Where the range splits.
    const int deltaNode = delta(i, j);
    int split = 0;
    int step = length;
    do {
        step = (step + 1) / 2;
        if (delta(i, i + (split + step) * d) > deltaNode) {
            split += step;
        }
    } while (step > 1);
    const int gamma = i + split * d + min(d, 0);
    const uint left = min(i, j) == gamma ? (kLeafBit | uint(gamma)) : uint(gamma);
    const uint right = max(i, j) == gamma + 1 ? (kLeafBit | uint(gamma + 1)) : uint(gamma + 1);
    children[(params.nodeBase + u) * 2] = left;
    children[(params.nodeBase + u) * 2 + 1] = right;
}
