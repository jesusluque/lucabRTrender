// Copyright (c) 2026 lucabRTrender contributors.
//
// The compute route's BVH: a linear BVH (Karras 2012) over one cloud's
// particles, built on the device -- boxes and Morton codes, a radix sort, the
// hierarchy from the sorted codes, boxes refitted bottom-up -- and traversed
// by rt_bvh_render.slang for devices whose ray tracing is not in hardware.
//
//   internal node i (0 .. n-2), root 0     boxes[2 i]: lo, boxes[2 i + 1]: hi
//                                          children[2 i], [2 i + 1]: child
//   child                                  bit 31 set: leaf k, else internal
//   leaf k (0 .. n-1), in Morton order     leaves[k]: the particle's index in its cloud
//
// Offsets into the combined buffers are per cloud: nodes at `nodeBase`,
// leaves at the cloud's particle base.
module bvh_common;

public static const uint kLeafBit = 0x80000000u;

public struct BvhParams {
    public uint  count;       // particles in the cloud
    public uint  nodeBase;    // first internal node in the combined buffers
    public uint  leafBase;    // first leaf in the combined buffers (the cloud's particle base)
    public uint  pad0;
    public float boundsLoX; public float boundsLoY; public float boundsLoZ; public float pad1;
    public float boundsHiX; public float boundsHiY; public float boundsHiZ; public float pad2;
};
