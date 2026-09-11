# Decisions

Why the engine is the way it is, with the measurements behind each choice.
Numbers are from an Apple M5 Pro (Metal) unless a section says otherwise.

## Ray tracing Gaussians

`lrt::render::GaussianRayTracer`, kernels in `shaders/lrt/rt/`.

### What is drawn

Each particle is evaluated in 3D at its peak response along the ray, with the
rasteriser's opacity-aware cut (`min(2 ln(255 alpha), 9)`). Particles are
blended front to back ordered by where they peak along the ray. The colour of
a particle is the rasteriser's: its harmonics evaluated for the direction from
the eye to its centre, once per frame (`rt_shade.slang`).

It deliberately differs from the rasteriser in three ways:

- No EWA approximation. The rasteriser projects with the local affine map;
  the ray tracer is exact for any lens.
- No 0.3 px screen-space dilation.
- Order by peak along the ray, not by centre depth.

### Ground truth

`ReferenceRenderer::renderPeaks` is the ray tracer's GPU reference. It
evaluates every particle for every pixel and sorts exactly by peak, with no
BVH, segments or carry. The ray tracer is held to it at p99 of at most one
8-bit sRGB code value, with at most 0.05% of pixels over two
(`tests/render/test_ray_tracing.cpp`).

Against the rasteriser, the tolerance only holds where the two definitions
agree:

| Scene | p99 tolerance |
|---|---|
| Orthographic, sparse, large particles (EWA is exact) | 2 |
| Perspective, sparse, particles of a few pixels | 4 |

Dense scenes differ by tens of code values. That difference is ordering, not
error. As particles grow, the perspective gap grows with them: p99 is 0 at
sizes 0.03–0.05, 2 at 0.1–0.15, and 20 at 0.3–0.5.

### Two routes, one integrator

Both routes offer entries to the same `Integrator`
(`rt_integrate.slang`), which records, evaluates, orders, blends and
segments:

- **Hardware.** Every particle is a stretched icosahedron (3DGRT's proxy) in
  a device BVH, queried with inline `RayQuery`. Metal and Vulkan.
- **ComputeBvh.** A Karras LBVH built on the GPU and traversed in compute.
  The build is Morton codes, radix sort, the hierarchy, then refit until a
  pass changes nothing. Traversal meets each particle's cut ellipsoid
  directly. It runs on every device, CUDA included.

Measured on train_7k (742k splats) at 1080p:

| Route | Frame | Build |
|---|---|---|
| Hardware | 435 ms | 731 ms |
| ComputeBvh | 343 ms | 168 ms |

On Metal, a non-opaque candidate costs a round trip out of hardware
traversal. An opaque closest-hit query over the same 15M proxy triangles
takes 6 ms, and merely enumerating every candidate takes 255 ms. `Auto`
therefore picks ComputeBvh on Metal. On Vulkan it picks Hardware, which has
not been measured. For comparison, the tile rasteriser draws the same frame
in 13 ms.

### What made it fast

In order, all measured on the same frame:

1. **Starting point: a 3DGRT k-buffer, 2.52 s.** It kept 16 hits sorted
   during traversal and restarted traversal from the 16th entry.
2. **Close segments where entries are dropped, 1.94 s.** Commit the farthest
   dropped entry, and stop committing only when an entry lands exactly in
   slot 16.
3. **Unsorted record, evaluate after traversal, 658 ms.** Per candidate, only
   record the distance, primitive and instance. Evaluate and sort after
   traversal ends. The record holds 256 entries; the sweep across sizes was
   64 → 845 ms, 128 → 657, 192 → 558, 256 → 530, 384 → 548 and 512 → 553.
4. **Precomputed data, no change (652 ms).** Per-particle frames and
   per-frame colours are cleaner, but did not move the time.
5. **Leave the record arrays uninitialised: −48 ms.**
6. **Sort an index permutation, not four arrays: −49 ms.**
7. **Counting sort into 128 buckets before insertion sort: −85 ms** (compute
   route). Insertion sort alone did about 2500 moves per ray. The bucket
   sweep was 32 → 362 ms, 64 → 351, 128 → 344 and 256 → 342.

Tried and rejected:

- **Sorted keys beside the indices.** No change.
- **Carrying peak and alpha from the compute route's leaf test into the
  record.** Slower, 344 → 389 ms.
- **Sharing code through struct methods.** This cost 11% until the scalars
  used per candidate moved into a separate small `Cursor` local. With the
  split, the shared version runs at the same speed as the hand-inlined one.

### Segments, order and carry

A traversal records up to 256 entries. When a ray enters more proxies than
that, the record closes at the nearest entry it had to let go. The next
traversal starts a hair before that point (a relative 1e-5) and recognises by
ID what the overlap reports again.

Restarting a hair *after* the boundary lost particles, as measured against
the reference. Starting exactly at it works on Metal, and the overlap keeps
it working on an intersector that does not report one triangle at one
distance twice.

A particle always peaks inside its proxy, so particles peaking beyond the
closing entry are carried into the next segment. Only when more than 64 are
carried are the nearest blended early; that is the only approximation.
Blending in entry order instead drew every proxy's silhouette as a seam.

A camera inside a particle's bound does not see that particle. Only entries
ahead of the ray start count, which is also what stops a restarted segment
from taking a particle twice.

### Bugs found on the way

These were fixed in `cmake/patches/slang-rhi-metal-acceleration-structures.patch`
or in the engine:

- **Freed structures crashed the next build.** Once any acceleration
  structure had been freed, slang-rhi's Metal backend put a nil into the
  device-wide structure array. The next build threw inside
  `NSArray initWithObjects:` and aborted the process. Freed slots now get
  empty placeholder structures.
- **Indexed builds read past their index window.** slang-rhi's Metal backend
  took `max(vertexCount, indexCount) / 3` as the triangle count. A chunk that
  indexed into a larger vertex buffer therefore read past its own indices.
  The count is fixed in the patch, and the engine now gives each chunk
  chunk-local vertex windows as well.
- **Winding.** Facing is decided in object space, so a mirroring instance
  transform must not flip the front face. Metal behaves this way, and the
  Vulkan and DXR specifications say the same.
- **Instances share particle IDs.** De-duplication has to key on the pair of
  particle and instance.
- **Flat particles.** The textbook ray–ellipsoid discriminant `b² − ac`
  cancels to nothing for very flat particles; it lost 176 of 400 thin layers.
  The compute route now computes the peak and the distance to it instead.

### Not done, not verified

- The OptiX pipeline route (CUDA). CUDA uses ComputeBvh instead.
- Vulkan and CUDA runs of either route. These need the Linux host.
- Rays that are not primary rays: shadows and reflections. The integrator
  takes any `RayDesc`, but nothing traces secondary rays yet.
