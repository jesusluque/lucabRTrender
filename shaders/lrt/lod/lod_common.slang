// Copyright (c) 2026 lucabRTrender contributors.
//
// LEVEL OF DETAIL, BUILT ON THE DEVICE
//
// Splats sorted by Morton code. The octree cell at level r holding a splat is
// its code's top 3r bits, so every cell's splats are one contiguous run of the
// sorted order: a level's cells ("groups") are where that prefix changes, found
// by a boundary pass and a prefix sum, never by walking a tree.
//
// Each group carries moments of the splats under it (Kerbl et al. 2024, "A
// Hierarchical 3D Gaussian Representation"): with w = opacity x the area of
// the splat's two longest axes,
//   W = sum w     S1 = sum w mu     S2 = sum w (Sigma + mu mu^T)     C = sum w colour
// Moments add, so a level's groups are their children's sums -- built from the
// finest merged level up -- and each group becomes one Gaussian:
//   mu = S1 / W,  Sigma = S2 / W - mu mu^T (eigen-decomposed into rotation and
//   scales),  colour = C / W,  opacity = min(W / its own two-axis area, 0.99).
//
// A group is drawn when its cell projects to at most `threshold` pixels and
// its parent's does not; a splat is drawn when the finest merged level's cell
// holding it does not. A child cell lies inside its parent, so its projected
// size (edge over nearest distance) is never larger: along any line down the
// tree the test flips once, and every place is drawn at exactly one level.
// Threshold 0 draws every splat and nothing merged: the cloud as it is.
//
// STREAMING
//
// The splats come in chunks -- runs of the Morton order -- that may or may not
// be on the device; the merged levels always are. A finest-level group whose
// cell wants its splats, but some of whose splats are in a chunk not loaded,
// is drawn merged instead (its own Gaussian: the nearest resident ancestor of
// those splats), and those chunks are asked for.
module lod_common;

public static const uint kLevels = 10;   // 30-bit codes

/// Moments per group: W, S1 (3), S2 (6: xx xy xz yy yz zz), C base colour
/// (3), C rest (3 per basis kept, rgb per basis).
public static const uint kMomentsHead = 13;

public struct LodParams {
    public uint  count;        // splats, or groups at this level
    public uint  level;
    public uint  groups;       // groups at this level
    public uint  fineGroups;   // groups at the finer level this one is built from
    public uint  keep;         // rest basis functions per colour
    public uint  shWords;
    public uint  stride;       // moments per group
    public uint  base;         // gather: first output slot
    public float boundsLoX; public float boundsLoY; public float boundsLoZ;
    public float extent;      // the bounds' largest edge: cells are cubes
    // cut
    public float eyeX; public float eyeY; public float eyeZ;
    public float pixelsPerUnit;   // focal in pixels at distance 1 (perspective) or per unit (ortho), times the transform's scale
    public float threshold;       // pixels a drawn cell may span
    public uint  orthographic;
    public uint  coarsest;        // 1 when this is the coarsest stored level
    public uint  offset;          // chunks: the chunk's first splat within its buffers
    public uint  chunkSplats;     // splats per chunk
    public uint  chunks;
    public uint  splats;          // the whole cloud's splats
    public uint  pad0; public uint pad1;
};

public uint levelShift(uint level) {
    return 3 * (kLevels - level);
}
