// Copyright (c) 2026 lucabRTrender contributors.
//
// What every radix pass agrees on.
//
// THE SORT
//
// LSD radix, eight bits a pass, over (key, value) pairs; keys of 32 bits, or
// 64 split into lo/hi. Stable, deterministic and portable:
//
//   1. histogram  -- one invocation per chunk counts its chunk's digits
//   2. totals     -- one invocation per digit sums that digit over chunks
//   3. starts     -- one invocation per digit: where the digit begins overall,
//                    then where each chunk's share of it begins
//   4. scatter    -- one invocation per chunk walks its chunk in order and
//                    writes each pair to its digit's next slot
//
// Every write lands at an index no other invocation writes: histograms and
// starts are rows per chunk, scatter destinations are disjoint by
// construction. No shared group memory and no atomic counters -- the two
// things openFXplayer's per-tile sort needed and Vulkan and D3D do not have.
// Walking chunks in order and chunks' shares in chunk order is what makes the
// sort stable, and stability plus in-order emission is what makes a splat
// frame deterministic.
module radix_common;

public static const uint kDigits = 256;

public struct RadixParams {
    public uint count;
    public uint chunkSize;
    public uint chunkCount;
    public uint shift;        // bit position of this pass's digit, 0..56
    public uint wide;         // 1 when keys are 64-bit (keysHi is live)
    public uint pad0;
    public uint pad1;
    public uint pad2;
};

/// This pass's digit of a key.
public uint digitOf(uint lo, uint hi, uint shift, uint wide) {
    if (wide == 0 || shift + 8 <= 32) {
        return (lo >> shift) & 0xFF;
    }
    if (shift >= 32) {
        return (hi >> (shift - 32)) & 0xFF;
    }
    return ((lo >> shift) | (hi << (32 - shift))) & 0xFF;
}
