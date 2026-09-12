// Copyright (c) 2026 lucabRTrender contributors.
//
// NanoVDB grids read on the device through PNanoVDB (OpenVDB's own
// portable reader, included as HLSL: its buffer is a StructuredBuffer<uint>
// and its functions are what Slang compiles). Several grids share one
// buffer, each at a 32-byte aligned word offset; a leaf is addressed by its
// index in the tree's leaf array, which is how the bounds and majorant
// kernels walk a grid without traversing it. PNanoVDB's own types stay
// inside this module; what it exports names grids and leaves by offset.
module nanovdb;

#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"

/// A grid: its word offset in the buffer.
public struct NvGrid {
    public uint word;
};

/// A leaf: its byte offset in the buffer.
public struct NvLeaf {
    public uint byteOffset;
};

pnanovdb_grid_handle_t handleOf(NvGrid g) {
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = g.word * 4u;
    return grid;
}

pnanovdb_leaf_handle_t handleOf(NvLeaf l) {
    pnanovdb_leaf_handle_t leaf;
    leaf.address.byte_offset = l.byteOffset;
    return leaf;
}

pnanovdb_root_handle_t rootOf(StructuredBuffer<uint> buf, NvGrid g) {
    return pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, handleOf(g)));
}

public NvGrid nvGrid(uint wordOffset) {
    NvGrid g;
    g.word = wordOffset;
    return g;
}

public uint nvLeafCount(StructuredBuffer<uint> buf, NvGrid g) {
    return pnanovdb_tree_get_node_count_leaf(buf, pnanovdb_grid_get_tree(buf, handleOf(g)));
}

/// Leaf `i` of the tree's leaf array (float grids: 8x8x8 voxels, 512 values).
public NvLeaf nvLeaf(StructuredBuffer<uint> buf, NvGrid g, uint i) {
    const pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, handleOf(g));
    const uint leafSize = PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_FLOAT, leaf_size);
    pnanovdb_address_t at = pnanovdb_address_offset64(tree.address, pnanovdb_tree_get_node_offset_leaf(buf, tree));
    at = pnanovdb_address_offset(at, i * leafSize);
    NvLeaf leaf;
    leaf.byteOffset = at.byte_offset;
    return leaf;
}

/// A leaf's origin in index space; its voxels span [origin, origin + 8).
public int3 nvLeafOrigin(StructuredBuffer<uint> buf, NvLeaf leaf) {
    const pnanovdb_coord_t c = pnanovdb_leaf_get_bbox_min(buf, handleOf(leaf));
    return int3(c.x, c.y, c.z);
}

/// Whether voxel `n` (0..511, z fastest) of a leaf is active.
public bool nvLeafActive(StructuredBuffer<uint> buf, NvLeaf leaf, uint n) {
    return pnanovdb_leaf_get_value_mask(buf, handleOf(leaf), n);
}

/// Voxel `n`'s value in a float leaf.
public float nvLeafValue(StructuredBuffer<uint> buf, NvLeaf leaf, uint n) {
    return pnanovdb_read_float(buf, pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_FLOAT, buf, handleOf(leaf), n));
}

/// The value at index coordinate `ijk`, through an accessor made for the call.
public float nvValueAt(StructuredBuffer<uint> buf, NvGrid g, int3 ijk) {
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootOf(buf, g));
    pnanovdb_coord_t c;
    c.x = ijk.x;
    c.y = ijk.y;
    c.z = ijk.z;
    const pnanovdb_address_t at = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, c);
    return pnanovdb_read_float(buf, at);
}

/// Whether the voxel at `ijk` is active.
public bool nvActiveAt(StructuredBuffer<uint> buf, NvGrid g, int3 ijk) {
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootOf(buf, g));
    pnanovdb_coord_t c;
    c.x = ijk.x;
    c.y = ijk.y;
    c.z = ijk.z;
    return pnanovdb_readaccessor_is_active(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, c);
}
