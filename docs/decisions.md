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

## SPZ

`io::readSpz`, `third_party/spz`, `shaders/lrt/scene/splat_encoding.slang`.

### Who does what

Niantic's reference reader (MIT, vendored as openFXplayer vendors it) only
decompresses. Version 2 and 3 files are gzip; version 4 is zstd.

The CPU arranges the quantised bytes into float records:

- A 24-bit fixed-point position is parsed into its integer.
- A smallest-three quaternion is split into two 16-bit halves, because a
  float holds 16 bits exactly and not 32.

The GPU decode does the rest:

- the fixed-point scale;
- `byte / 16 - 10` log scales;
- the DC term at SPZ's 0.15 scale;
- both quaternion packings;
- `(byte - 128) / 128` harmonics;
- the turn from SPZ's right-up-back to the right-down-front a 3DGS PLY is in:
  y and z of positions and rotations negate, and each harmonic basis takes
  the sign of its parity in y and z.

### How it is checked

Two tests check it:

- **Hand-written files** with exact decoded values: version 3 with degree-1
  harmonics, and version 2.
- **Niantic's packer against the PLY it packed.** A degree-3 cloud goes
  through Niantic's packer (versions 3 and 4) and is rendered against the
  same cloud read as a PLY.

Results of the second test, p99 in 8-bit sRGB code values:

| Harmonics rendered | p99 | Same comparison with bands 2 and 3 signs wrong |
|---|---|---|
| None | 3 | |
| Degree 1 (5-bit) | 4 | |
| Degree 2 (4-bit) | 10 | 136 |
| Degree 3 (4-bit) | 13 | 234 |

The difference that remains is quantisation. Sign errors are ruled out: the
deliberate control is an order of magnitude worse.

Degree-4 files load with the fourth band dropped, because the engine
evaluates up to degree 3.

## SOG

`io::readSog`, `shaders/lrt/scene/sog_decode.slang`, `scene::loadSplatFile`.

### Who does what

The CPU only unpacks:

- opens the zip, stored or deflated, or the directory beside a `meta.json`;
- decodes each WebP with libwebp to its raw RGBA bytes, never premultiplied,
  since these channels are indices;
- parses ranges and codebooks with nlohmann/json.

The GPU does the reconstruction:

- 16-bit positions in their signed log domain;
- version 2 codebooks and version 1 ranges;
- the smallest-three quaternion with its mode byte;
- the higher-harmonics palette.

It writes records in the engine's float encoding, which then take the same
validate and decode as every other format. Loading a SOG sends nothing back
to the CPU. The USD export, which consumes host records, reads them back
(`CloudLoader::records`).

### How it is checked

PlayCanvas's own converter wrote the fixtures in `tests/data/splats`. The
converter reorders splats and clusters harmonics, so the test compares
renders against the source PLY rather than splat by splat:

| Fixture | Render | p99 |
|---|---|---|
| `tiny`, version 2 | no harmonics | 5 (8-bit codebooks) |
| `sh3`, 64-entry palette | no harmonics | 1 |
| `sh3`, 64-entry palette | degree-3 harmonics | 1 |
| `sh3`, palette red and blue exchanged | degree-3 harmonics | 170 |

A first `sh3` of 200 splats had a 128-entry palette. Its k-means loss on
random harmonics gave p99 61, which proves nothing about decoding. The
fixture was cut to one palette entry per splat.

## SplatEdit and the lrt schemas

`shaders/lrt/common/edit.slang`, `render::SplatEdit`, `modules/usd/schemas`.

### One rule, every renderer

A SplatEdit is openFXplayer's, rule for rule: a box or sphere in the cloud's
own space, what happens to the splats inside it (keep, remove, grade), a
grade (tint, brightness, saturation about Rec.709 luma, opacity), and two
filters that apply wherever the volume is (minimum opacity, maximum scale).

The rule is written once and read by four renderers:

- the tile rasteriser;
- the rasteriser's GPU reference;
- the ray tracer, in its per-instance shade pass, carrying the edited opacity
  that the integrator then cuts by;
- the ray tracer's GPU reference.

An edit belongs to an instance, not to a cloud, so two instances of one
cloud can be edited differently.

Checks, p99 in 8-bit sRGB code values:

| Test | Result |
|---|---|
| Rasteriser vs its reference, keep, remove, inverted grade, filters | 0 |
| Ray tracer vs its reference, both routes, two differently edited instances | 0 |

### In USD

`LrtSplatEditAPI` is a codeless applied API schema. Its properties are
constant primvars `primvars:lrt:edit:*`, and constant primvars inherit down
the namespace. An edit authored on an Xform therefore stands over every
ParticleField below it, which is openFXplayer's Edit node over its subtree,
with no UsdImaging adapter to write. A Hydra render of such a stage matches
the direct render with the same edit at p99 1.

`LrtPointStyleAPI` declares the point primvars the delegate already read
(`lrt:sizeInPixels`, `lrt:edl`, `lrt:surfaceOffset`).

Both schemas are written by hand in usdGenSchema's output form, since this
OpenUSD build has no Python. They are installed beside hdLrt, so one
`PXR_PLUGINPATH_NAME` finds both.

No `LrtCameraWindowAPI`: a UsdGeomCamera already expresses openFXplayer's
window. Translate is the aperture offsets, scale is the apertures, and roll
is the camera's own rotation. The SceneText bridge maps to those.
`LrtStreamedAssetAPI` waits for the LOD work.

## Levels of detail

`modules/lod`, `shaders/lrt/lod`. The method is written out in
`lod_common.slang`.

### Built and cut on the device

- **Build.** Splats are sorted by 30-bit Morton code. An octree cell at
  level r is a run of equal top-3r-bit prefixes, found with a boundary pass
  and a prefix sum. Moments add (after Kerbl et al. 2024): the finest merged
  level is summed from the splats, and each coarser level from its children.
  Each group then becomes one Gaussian, with its covariance diagonalised by
  Jacobi sweeps in the shader.
- **Which levels are stored.** Levels from `coarsestLevel` down to the
  deepest level whose cell count is at most half the splat count.
- **Cut, per group, fully parallel.** A group is drawn when its cell projects
  to at most the threshold and its parent's cell does not. A splat is drawn
  when the finest merged level's cell does not. Projected size is edge over
  nearest distance, and a child cell lies inside its parent, so the test is
  monotone down the tree and every place is drawn at exactly one level.
- **What comes back to the CPU.** Only counts: one per level while building,
  and one read per instance per frame, which holds every part's count plus,
  when streaming, each chunk's need.

### Measured (M5 Pro)

**train_7k (742k splats).** Building took 47 ms and made 146k merged
Gaussians over levels 1 to 10. On a far view at 1080p:

| Threshold | Drawn | Cut | Render | Image |
|---|---|---|---|---|
| 0 (off) | 742k | | 12.6 ms | reference |
| 4 px | 727k | 2.3 ms | 11.4 ms | mean abs 1e-4 |
| 8 px | 83k (11%) | 1.3 ms | 3.3 ms | mean abs 3e-3 |

**Random-colour test clouds**, the worst case for merging:

| Threshold | Drawn | p99 |
|---|---|---|
| 2 px | 64% | 9 |
| 4 px | 19.5% | 29 |

**Exactness at threshold 0.** p99 is at most 1. The residue comes from depth
keys that tie and keep index order, which the Morton sort has changed.

**A cell with one splat.** It merges back into that splat, with covariance
equal to 1e-4.

### Chunks, `.lrtc` and streaming

- **Chunks.** The cloud's own splats are cut into chunks: runs of
  `chunkSplats` of the Morton order (65536 by default), so each chunk is a
  compact piece of space. The merged levels are small and always on the
  device; chunks may or may not be. A chunk on the device sits in a slot of a
  store, and the store's slots need not follow the chunks' order. Built in
  memory, chunk c is slot c and every chunk is there. The cut draws one run of
  consecutive slots per dispatch, which is a single run in that case.
- **The finest merged level decides for its splats.** A finest-level group
  whose cell wants splats draws them when every chunk holding them is on the
  device, and draws its own Gaussian otherwise. Its merged Gaussian is the
  nearest resident ancestor of those splats, so a missing chunk never leaves
  a hole. Each splat stores the index of its finest-level group, which
  replaces the Morton key at frame time, and reads the group's decision. The
  test no longer runs per splat, and every place is still drawn exactly once.
- **What a view wants.** `lodChunkNeeds` gives, per chunk, the largest
  projected edge among the cells that want its splats: 0 for none, otherwise
  in 1/16 px. It comes back in the same single read as the counts.
- **Why chunks follow the Morton order and not group boundaries.** Fixed-size
  chunks give uniform slots, and a store of uniform slots never fragments.
  Groups that straddle a chunk boundary only need to check more than one
  chunk, which is a short loop in the finest-level kernel.
- **`.lrtc` v1.**
  - Layout: a header in page 0; level and chunk tables; the finest level's
    group starts; each level's positions, shape, SH and cells; each chunk's
    positions, shape, SH and finest group.
  - Every block begins on a 4096-byte page, so a chunk can be mapped and
    faulted in alone. The bytes use the device's own packing, so reading is
    a copy.
  - The writer writes `name.partial` and then renames it, so a failed write
    never looks like a whole file.
  - `lrt convert in.ply out.lrtc`.
- **`StreamingPool`.**
  - The file is mapped. The levels go to the device when the pool opens, and
    the store starts empty.
  - After each cut the caller passes its needs to `want`. `update` queues the
    missing chunks, most wanted first, only as many as have a place to go.
  - A place is, in order of preference: a free slot; the slot of a chunk not
    wanted now, least recently wanted first; or the slot of a chunk wanted
    less than half as much. The half stops two chunks from swapping every
    frame.
  - Loader threads copy a chunk off the mapping, so the page faults happen on
    those threads and not in the frame. The next `update` uploads it and
    flips its resident flag.
  - `update(true)` waits for the queued loads; an offline render repeats it
    until a frame places nothing. `lrt render --stream-budget N` does exactly
    that, and `lrt bench` streams without waiting.

### Measured: streaming (M5 Pro)

**train_30k (1.05M splats, 16 chunks).**

- `lrt convert` to `.lrtc` takes 2.8 s in total and writes 150 MB, against
  266 MB for the PLY.
- At 1080p with `--lod 2`, the PLY built in memory, the `.lrtc` read whole
  and the `.lrtc` streamed into 4 slots write byte-identical EXRs. The stream
  settles in 2 cuts (48 ms).
- At `--lod 0.25` with 4 of 16 slots, 12 wanted chunks do not fit, and their
  places are drawn merged. Loading by priority instead of chunk order raised
  the visible splats from 155k to 170k.

**Tests, 20k splats in 20 chunks.**

- A store with its slots reversed gives the same cut and the same image,
  max 0.
- Dropping the chunks a view does not want changes nothing, max 0.
- Missing chunks are drawn merged.
- With 8 slots, the frame the pool settles on is p99 0 against the whole
  cloud.
- Turning the camera to the other side evicts 4 chunks and settles back to
  max 0.

### In USD: `LrtStreamedAssetAPI`

- **The schema.** It is codeless, like the others, and its properties are
  constant primvars:
  - `lrt:asset`: the `.lrtc` file.
  - `lrt:lod:threshold`: pixels.
  - `lrt:stream:budget`: splats, where 0 reads the file whole.
- **What it does to the prim.** Authored on a ParticleField, the asset stands
  in for the prim's own arrays, which may be left empty.
- **Where the file is opened.** The engine opens it in `commit`, on the render
  pass's thread. It opens it again only when the path or budget changes; a
  new threshold alone does not reopen it.
- **One cut per frame.** Every asset is cut in a single `CutSelector` call,
  which is why `LodInstance` carries its own threshold: a second call would
  overwrite the clouds the first returned.
- **Waiting for streams.** `lrt:settleStreams` is a render setting. It is
  false by default, so a viewport fills in over the frames that follow.
  `StageRenderer`, which makes images, sets it true and cuts and loads until
  nothing more is placed.
- **The ray-traced technique.** It draws an asset read whole as its whole
  cloud, and does not draw streamed assets. A cut changes every frame, and
  the tracer would rebuild every frame.
- **Checked.** A stage referencing a `.lrtc`, read whole and streamed into
  8 of 20 slots, renders through Hydra with max 0 against the same cut and
  stream done directly.

### Not done yet
- **The cut still waits twice a frame:** once for its counts, which come back
  in one read (reading them level by level had cost 3.9 against 2.3 ms), and
  once for the gather.
- **LOD with the ray tracer.** A cut that changes every frame would rebuild
  the structures every frame.

## Time: FrameClock and `lrt live`

`modules/sched` (genlock underneath) and `apps/lrt/src/CmdLive.cpp`.

### What it does

- **The clock.** `FrameClock` runs free on this machine's clock, or follows a
  PTP master as a genlock `PtpClock` slave. Frame N of a rate begins at the
  instant ST 2059-1 gives it, computed from the TAI epoch, never summed. Two
  nodes following one master therefore agree on it without talking to each
  other.
- **Timecode.**
  - It is the UTC time of day, counted from the first frame that begins at or
    after midnight.
  - At 30000/1001 and 60000/1001 it is drop-frame: SMPTE 12M's labels ;00–;01
    (;00–;03 at 59.94) are skipped every minute not a multiple of ten. Other
    rates, 24000/1001 included, count non-drop.
  - `framesFromTimecode` inverts the count, and a test checks the round trip.
- **`lrt live stage.usd [--ptp host --port N] --rate R --frames N --at
  HH:MM:SS:FF -o out.####.exr`.**
  1. It waits for a lock.
  2. It renders one warm-up frame, because the first render loads the stage
     and compiles shaders.
  3. It waits for each frame's instant, renders the stage at that frame's USD
     time, and hands the EXR to a writer thread.
  - **Missed frames.** Frames whose instant passes during a render are
    skipped, not drawn late.
  - **What each EXR carries.** `timeCode` (SMPTE 12M BCD, OpenEXR's type),
    `framesPerSecond` (rational), `lrt:taiNs`, `lrt:frameIndex`,
    `lrt:usdTime`, `lrt:wakeLateMs` and `lrt:clock`.
- **`--at`.** It names the timecode at which `--start` plays. Without it,
  each node counts USD time from when it happened to start, so two nodes
  would draw different times for the same instant: correct frames, wrong
  content. With the same `--at`, they draw the same time.

### Waking on time

`std::this_thread::sleep_for` on macOS overran by 7 ms on average and 10 ms at
worst, measured, whatever the thread's QoS.

- **macOS.** `platform::sleepPrecisely` gives the thread a time-constraint
  (real-time) policy only while it sleeps. It must not keep it while it
  renders, because a thread that overruns its computation budget is demoted.
  The overrun drops to 36 µs at worst.
- **Linux.** The thread's timer slack is set to 1 ns instead.
- **The last 100 µs.** `FrameClock::waitFor` spends them yielding, and it
  rereads the clock because a PTP correction may have moved it.

### Measured (M5 Pro, loopback master `genlock-cli master --port 3190`)

- **Tests.** Free run: the latest of 10 wakes came 9 µs after its alignment
  point. Following the master: the clock reads 0.14–0.41 ms off it, the
  software-timestamp error on a loaded machine.
- **Two `lrt live` nodes on `train_7k` at 640x360 and 25 fps.**
  - The nodes started seconds apart and shared one GPU and one `--at`.
  - Both drew frames 44728571775 to 44728571799, from 16:07:14:00, at
    identical USD times.
  - 0 frames were skipped, the latest wake was 12 µs late, and renders took
    17 ms.
  - Writing EXRs inside the loop had cost 6 skipped frames in 15.

### Not done

- **Output.** It is EXR files only; nothing is sent to a video output or
  over the network.
- **Scanout.** Software cannot phase-lock a display (genlock's README says
  why); an SDI card is what would.
- **Linux PTP.** Untested here. Its kernel software timestamps should narrow
  the error.

## Complete USD: toolchain (M0)

This is the first milestone of the plan to render all of USD: geometry,
materials, lights, cameras, animation, curves, volumes and render settings.
The work proceeds in two techniques, an interactive raster and a path tracer,
with MaterialX feeding a Slang generator and `lrt view` as the viewer.

### What changed and why

- **OpenUSD 26.08 with MaterialX 1.39.5 and OpenVDB 10.1 (with NanoVDB)**,
  in `~/tools/usd-26.08-mx` via `scripts/build-usd.sh`.
  - **MaterialX:** 1.39.5 is the first release with a Slang shader generator
    (`MaterialXGenSlang`, on by default). `MATERIALX_SLANG_RHI_SOURCE_DIR`
    stays unset, since MaterialX's own Slang renderer would bring a second
    slang-rhi into the process.
  - **Build fix:** CMake 4 refuses c-blosc's `cmake_minimum_required`, so the
    script exports `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
  - **Switching over:** the new prefix sat beside the old one until the engine
    passed 54/54 against it. Only then did the presets move.
- **Open Image Denoise 2.5.1, built from source by `scripts/build-oidn.sh`.**
  - **Why not the release binaries:** they ship their own `libtbb.12`, a
    second TBB beside USD's under the same soname.
  - **GPU devices only:** Metal here, CUDA on Linux. A denoiser that could
    fall back to the CPU is a CPU fallback.
  - **Sharing the queue:** `technique::Denoiser` opens OIDN on the engine's
    own Metal command queue.
  - **Result:** `lrt info` reports `denoiser OIDN 2.5.1 on Metal` and
    `tbb libraries 1`. The `single_tbb` test holds that count.
- **The aofx SDK is pinned.**
  - `aofx_sdk_manifest` hashes every header in
    `modules/aofx/sdk/include/aofx` against `tests/aofx/sdk_manifest.txt`.
  - The headers differ from openFXplayer's only in the doc comments corrected
    here; `kAbiVersion` is 22 in both.
  - Re-recording the manifest is allowed only when openFXplayer's SDK moved
    the same way.
- **Four places where the CPU did arithmetic on data, now on the device.**
  - **ParticleField and Points arrays:** Sync keeps the `VtValue`s Hydra hands
    it, float or half, with no copy. The commit uploads their bytes, and
    `scene/streams.slang` interleaves them into records. The per-element
    interleave loops and the half-to-float conversions on the host are gone.
    Half attributes draw as their float twins at p99 1.
  - **Hydra render buffers:** `usd/aov_convert.slang` fills them. It converts
    to the buffer's format and turns view z into the host projection's
    [0, 1], one thread per output word. Rows stay bottom first, as Storm and
    hdEmbree lay out Hydra buffers (M2 found the flip this first did). The per-pixel loops in
    `RenderBuffer::WriteColour` and the render pass are gone.
  - **StageRenderer:** it reads the engine's targets directly, so
    `StageImage.depth` is now view z, as `lrt render` writes it.
- **A latent configure bug.**
  - **Symptom:** in a fresh build directory, Catch2 was never fetched.
  - **Cause:** `include(Dependencies)` ran before `include(CTest)` defined
    `BUILD_TESTING`.
  - **Fix:** CTest is now included first.
- **Deferred to M3:** GLFW and Dear ImGui arrive with `lrt view`, their only
  user, rather than as unused dependencies now.

## Complete USD: GPU foundations (M1)

What the next milestones build on, in `modules/gpu`.

- **`Texture` and `Sampler`.**
  - They own their slang-rhi objects and report failures as `Result`.
  - Each subresource is uploaded with a single command and read back only for
    output and tests.
  - Engine images stay buffers, because kernels index them. A texture is for
    what needs one: material images with mips, render targets, depth.
- **`MipGenerator` (`algo/mips.slang`).** slang-rhi has no mip generation.
  - Each texel of a level is the area-weighted mean of the texels above it.
  - An odd edge of 2n + 1 folds into n with weights (n − x, n, x + 1) / (2n + 1).
  - Every source texel therefore gives exactly n/(2n + 1) of itself to the
    level below, so the chain keeps level 0's mean.
  - Measured on 64², 37×23, 1×9 and 128×5: the means agree to 2e-6, and the
    37×23 chain drifts only in the seventh decimal.
- **`RasterKernel`.** A vertex and fragment pipeline bound by name.
  - Draws pull their data from StructuredBuffers, with no vertex buffers or
    input layouts, as the point rasteriser already did.
  - A draw that binds something of its own gets a fresh root object. Draws
    that bind nothing share the pass's (`RasterPass::bind`), added in M2.
  - Checked: two triangles over the left half of clip space cover exactly
    w/2 × h pixels.
- **`RayTracingKernel`.** A pipeline plus its shader table, for OptiX and
  Vulkan RT.
  - On Metal, slang-rhi has no pipelines, only inline RayQuery in compute, so
    this reports `Unsupported` and the engine traces with ComputeKernels there.
- **`ShaderLibrary` extensions.**
  - **Link-time constants:** a module declares
    `extern static const uint kName;` and is linked against a generated
    exports module. Each distinct set of values gets its own program.
  - **Generated modules:** `loadSource` compiles modules that exist only as
    source (materials). Loading an existing name with different source is
    refused.
- **Persistent shader cache (`DiskShaderCache`).**
  - One file per compiled program, holding its key and its data, written to a
    temporary name and renamed into place.
  - Location: `$LRT_SHADER_CACHE`, or `lucabRTrender/shaders` under the
    platform's cache directory.
  - The key comes from slang-rhi and includes the linked program's hash, so
    edited shaders miss the cache.
  - The whole suite dropped from 89 s to 26 s on a warm cache.
- **Image comparisons (`render/ReferenceRenderer`).**
  - **`compareHdr`:** per-pixel relative difference in a logarithmic histogram
    (eight bins per octave), and relMSE Kahan-summed per row. It is for
    radiance above 1 and dark noise, where 8-bit code values say nothing.
    Checked: a against 1.1·a gives p99 0.0964 against an exact 1/11.
  - **`countDifferent`:** counts the differing entries of two uint buffers,
    per chunk, then reduces the counts. It is meant for ID AOVs.

## Complete USD: geometry and visibility (M2)

`UsdGeomMesh`, PointInstancer and native instancing, drawn through Hydra
with ids, depth, normals and primvars as render outputs. There are three
routes to visibility, and all of them agree with Storm.

### On the device, in Hydra's order

- **`geom::MeshBuilder`.**
  - Hydra's topology is triangulated in `HdMeshUtil`'s fan order, with holes
    and left-handed orientation.
  - Smooth normals use `Hd_SmoothNormals`' formula: cross products per
    corner, scattered to points through a radix sort.
  - Primvars of every interpolation are expanded per triangle corner.
    Indexed primvars are resolved on the device, and doubles are decoded
    from their two words.
  - Checked:
    - Areas agree with the faces'.
    - A height field's normals come within 1.1e-7 rad of a brute-force sum.
    - A sphere's normals come within 1.1e-6 rad of radial.
- **`world::GpuScene`.**
  - Every mesh sits in shared pools (positions, indices, corners, faces,
    primvar values) with one 64-byte record each.
  - Every drawn copy has a 176-byte `InstanceRecord`: object to view and its
    normal matrix, object to world, look, ids and the double-sided flag.
  - The pools are repacked only when the mesh set changes, which is what
    `generation()` counts.
- **`world::Instancing`.**
  - Each level is composed as Storm composes it:
    `instancer * T * R * S * instanceTransform`, nested
    `parent[i] * level[j]`, and read from float, half or double primvars.
  - A chain is recomposed only when an instancer in it changes.
  - Instanced sets are pooled on the device, and one dispatch writes every
    set's records: a thread binary-searches its set. This used to be a
    dispatch per set, 32 ms of command recording for Kitchen_set_instanced's
    1462 sets; it is now 0.4 ms.

### Three routes, one visibility buffer

Each route writes (instance + 1, triangle) per pixel. Shading and AOVs
rebuild the hit from those two numbers with Möller–Trumbore in view space
(`technique/surface.slang`), so the routes shade alike.

- **`VisibilityRaster`.**
  - Reversed infinite Z, with depth `near / z` in D32Float.
  - One draw per mesh with `instanceCount`, because Metal has no indirect
    draws.
  - Every draw shares one root object. A draw's mesh and instances arrive
    as its start vertex and start instance.
  - On Metal, `vertex_id` and `instance_id` already include those starts.
    Slang's Vulkan and D3D output subtracts them. `Caps::drawIdsIncludeStart`
    records the difference, and `tests/gpu/test_textures.cpp` measures it.
- **`VisibilityTrace`.**
  - A BLAS per mesh over the pools, rebuilt when they are repacked. A TLAS
    per frame.
  - A kernel writes the instance descriptors from the records, in the
    backend's layout: 64 B generic/D3D12/Vulkan, 80 B OptiX, 68 B Metal.
- **`VisibilityBvh`.**
  - A Karras LBVH per mesh and one over the instances, from the splat ray
    tracer's build kernels.
  - It is for devices without ray tracing hardware.
- **Single-sided meshes keep their front only**, as in Storm.
  - Raster: `SV_IsFrontFace`, flipped when the transform mirrors.
  - Hardware rays: cull flags, with double-sided instances opting out.
  - BVH: `det < 0` in object space.
- **Which route.**
  - `lrt:visibility` (`lrt stage --visibility`) selects `automatic`,
    `raster`, `rays` or `bvh`.
  - Automatic takes rays where the device has ray queries, else raster,
    else the BVH. Rays win on this machine at every size measured (below).
- **Layers.** Meshes and points are composited by view z into the opaque
  layer the splat rasteriser draws over.

### Hydra outputs

- **Render outputs.** primId, instanceId and elementId (Int32, cleared to
  −1), Neye and normal (Float32Vec3), `primvars:NAME`, colour and depth.
- **Conversion.** `usd/aov_convert.slang` converts on the device.
- **Row order.** Buffers are bottom row first, which is Storm's and
  hdEmbree's layout. M0 had flipped them; Storm showed it.

### How it is checked

All comparisons are kernels, and the numbers are from the last run.

- **Analytic square.** 8281 pixels covered, with 0 coverage and 0 colour
  mismatches, and depth exact. Through Hydra: 8464 pixels, all exact.
- **Instancing.**
  - Six instances of one mesh against six meshes: identical colour and
    depth bits.
  - Six nested instances against six authored: relMSE 1.2e-9. The residue
    is half-precision rotations.
  - A PointInstancer against authored transforms: relMSE 3.4e-9.
- **The routes against each other.**
  - On a bumpy grid with twelve squares (single-sided, double-sided,
    mirrored), 0 of 23654 interior pixels differ between raster and either
    ray route.
  - Through Hydra, on a scene with instancing, culling and a mirrored mesh,
    the three routes differ in 15 and 6 id words out of 230400. Those words
    lie along a grazing edge.
- **Culling.** A single-sided square shows 6723 pixels from the front and 0
  from the back, mirrored or not.
- **Layers.** Splats and points behind an opaque wall change 0 pixels. In
  front of it they change 24137, on all three routes.
- **Storm as the oracle** (`lrt_storm_oracle_tests`, in its own process).
  - Setup: Kitchen_set at 480×270, compared on primId segmentation,
    coverage, depth and Neye.

    | Route | Coverage | Segmentation | Depth, worst | Neye > 6/255 |
    |---|---|---|---|---|
    | raster | 0 differ | 18 of 57219 | 1.2e-6 | 17 |
    | rays | 4 differ | 41 of 57236 | 6.3e-6 | 17 |
    | bvh | 4 differ | 36 of 57233 | 6.3e-6 | 17 |

  - **Why segmentation.** Storm numbers prims differently from the engine,
    so ids are compared as a segmentation: a pixel whose 3×3 neighbourhood
    is one prim in one image must be one prim in the other.
  - **Why Neye in bytes.** Storm writes Neye into UNorm8, where negative
    components clamp. Ours is compared in that space.
  - **Storm renders single-sampled.**
    - With multisampling, Metal cannot resolve an R32Sint target. The Metal
      validation layer asserts it, and without the layer the id buffers come
      back with their upper 16 bits unwritten.
    - OpenUSD reads `HDX_MSAA_SAMPLE_COUNT` as its libraries load, so ctest
      sets it in the environment and the test refuses to run without it.

### Measured (M5 Pro, release)

- **Method.** `lrt stage --frames 40 --visibility <route>`, median frame
  after the first.
- **What a frame includes.** Hydra sync, visibility, headlight shading, the
  splat pass (empty here) and reading colour and depth back.
- **Camera.** The oracle's: eye (500, −350, 350), focal 20.

| Scene | Size | raster | rays | bvh |
|---|---|---|---|---|
| Kitchen_set (1788 meshes) | 480×270 | 18.0 ms | 4.8 ms | 8.9 ms |
| Kitchen_set | 1920×1080 | 31.7 ms | 19.2 ms | 29.2 ms |
| Kitchen_set_instanced (1462 sets) | 480×270 | 15.2 ms | 5.4 ms | 9.0 ms |
| Kitchen_set_instanced | 1920×1080 | 29.1 ms | 19.2 ms | 28.6 ms |

- **First frame.** 3.2–4.3 s with raster or rays, 5.5–6.5 s with the BVH.
  That is the stage load, mesh builds and shader compiles on a cold cache.
- **Where raster's time goes.** Recording 1800 draws costs the host about
  8 ms, even with one root object: slang-rhi writes render state and looks
  up binding data per draw.

### Not done, not verified

- **Draw count.** Raster pays per draw, and single-instance meshes could
  share draws.
- **The TLAS is rebuilt every frame**, not refit.
- **Other backends.** Vulkan and D3D12 start-location semantics are
  unverified. So are the OptiX descriptor layout on real hardware and
  visibility on CUDA, which has no raster.
- **Storm oracle coverage.** One camera on one stage. Negative Neye
  components are not compared, because Storm clamps them.
- **Arrives with later milestones.** geomSubsets and materials (M4).
  Deformation and motion, which need BLAS refit (M7). Subdivision, curves
  and implicit surfaces (M8).

## Complete USD: lrt view (M3)

`lrt view stage.usd` is a window onto a stage through the engine's Hydra
delegate.

- **Cameras.** A free camera: orbit with the left button, pan with the
  middle button or shift, dolly with the right button or wheel, and F to
  frame. The stage's own cameras can be picked too.
- **Choices in the panels.**
  - Technique: raster or rt.
  - Mesh visibility route.
  - Output: colour, depth, prim, instance and element ids, Neye, normal.
  - View transform, display, exposure and render scale.
- **Stage.** A tree of the stage, and a click to pick the prim under the
  mouse.

### Frames stay on the device

- **Drawing.** `StageRenderer::draw` executes Hydra and reads nothing back.
- **Hydra buffers.** They are converted only when mapped: the render pass
  leaves each one a fill that runs on its first `Map`. A host that shows an
  output on the device never pays for it on the host. `lrt stage` and tests,
  which map, read what they did before.
- **Display.**
  - `StageRenderer::displaySource` hands the frame's colour or an AOV
    (`Engine::aovView`) to `technique::DisplayTransform`.
  - The transform writes the window's surface texture directly: BGRA8Unorm
    with storage usage, so `framebufferOnly` is off.
  - Any render scale; each output pixel shows the source pixel under it.
- **Panels.**
  - Dear ImGui 1.92.9 (`ImGuiBackendFlags_RendererHasTextures`) draws over
    the display through `view::ImGuiRenderer`, on the same device.
  - Each frame its lists go up as two buffers. A draw pulls vertices through
    32-bit indices from its start vertex and binds its texture and scissor.
  - ImGui tessellates on the CPU. That is the one place this viewer does
    arithmetic on the host, and it is chrome, not scene data.
- **Picking.** `StageRenderer::pick` reads one pixel's two id words and
  resolves the rprim to its USD prim through `HdPrimOriginSchema`.
- **Framing.** `GpuScene::worldBounds` folds every instance's world box on
  the device. Clouds add their decoded boxes through their prims'
  transforms.
- **Window.** GLFW 3.4 with no client API; slang-rhi makes the Metal surface.
  `platform::matchLayerToBacking` sets the layer's contents scale through the
  Objective-C runtime, so drawables map one to one on Retina screens.

### Display transform

- **View transforms.** Standard, and AgX in Wrensch's analytic fit of
  Sobotka's: inset, log2 over [−12.47, 4.03] stops, a sixth-order sigmoid,
  outset, then 2.2.
- **Displays.** sRGB, Rec.709 (BT.1886, a pure 2.4 power) and Display P3
  (P3-D65 primaries with sRGB's transfer).
- **Other outputs.** Depth is a log grey from near to far, ids are hashed
  colours (−1 is the background), and vectors are shown as rgb·½+½.
- **Checked** (`tests/technique/test_display.cpp`).
  - A generated ramp from 2⁻¹⁰ to 2⁶, with hues and partial coverage over a
    background, goes through six view, display and exposure combinations.
  - Each output is compared per pixel with the formulas written again in
    another kernel: the P3 matrix derived from chromaticities, the sigmoid as
    powers, exposure as exp.
  - Worst difference 3.5e-6.

### How it is checked

- **Smoke test** (`lrt_view_tests`). A hidden window draws a square stage
  for four frames. A kernel counts the snapshot's lit pixels (49538 at
  480×320).
  - It skips where GLFW cannot initialise or a window cannot open.
  - A hidden window's drawables come at about 100 ms each; a shown window's
    at the display's rate.
- **Picking and bounds** through `StageRenderer` on the primvars stage:
  - Pixels over each mesh pick `/PerFace` and `/PerCorner`; an empty pixel
    picks nothing.
  - The bounds come to (−2, −1.5, −5)–(2, 1.5, −5), the authored points.
- **Snapshot.** `lrt view --frames N --snapshot out.exr` writes the last
  frame as shown, panels included: a float texture read back for output.
  Looked at for Kitchen_set.

### Measured (M5 Pro, release)

- **Setup.** `lrt view --frames 200` in a 1600×900 window with a free
  camera, raster technique and automatic (ray) visibility.
- **Draw.** Kitchen_set 7.08 ms, Kitchen_set_instanced 7.15 ms (medians).
  That covers Hydra and the engine.
- **Frame.** 10.0 ms for both, which is the display's vsync, not the
  engine.

### Not done

- **Display.** ACES 2.0, OCIO and EDR output (RGBA16Float with extended
  range) are not implemented.
- **Picking under instancing.** It names the prototype's prim and the
  instance number, not the instance proxy's path. The viewport does not
  highlight the selection.
- **Stage tree.** It lists prims and marks native instances. It does not
  walk into instance proxies.
- **Time.** The time slider sets the stage time; animation itself is M7.
- **Platforms.** Linux and Windows windows are untested. X11 is wired
  through GLFW's native handle; Wayland is not.

## Complete USD: textures and materials (M4)

A mesh no longer shows its displayColor: it shows the material bound to it,
compiled from MaterialX into Slang and evaluated on the device.

### Textures

- **Reading.** Hio decodes a file's bytes on the CPU and nothing else: the
  bytes are uploaded raw and a kernel decodes them (v up, since Hydra's rows
  run the other way).
- **Mips** are a kernel, since slang-rhi generates none. An sRGB texture is
  decoded, filtered and encoded again, so a mip's mean is the mean of the
  level above it in light, not in code values. Views are made with the sRGB
  format its samplers want (a slang-rhi patch: a full-range view ignored the
  format it was asked for).
- **UDIM** is an indirection table: a tile that is missing leaves the node's
  default, and the graph says so rather than sampling black.
- **The table.** One `ParameterBlock` of 1024 texture slots and its
  samplers, deduplicated. Metal takes it as an argument buffer; a device
  with bindless will take the same interface.
- **Filtering is per target.** A footprint is sampled with its gradients
  where a compute entry point may ask for them, and otherwise from the level
  the wider side of the footprint lands on -- CUDA has no `SampleGrad` in
  compute. The choice is a `__target_switch` in the shader, not a build
  flag. On Metal, where both exist, they pick the same levels (0, 1, 2, 3
  for footprints of 1, 2, 4 and 8 texels) and the same samples.
- **Colour spaces.** MaterialX `srgb_texture` is sRGB and anything else is
  raw; `UsdUVTexture`'s `sourceColorSpace` is auto, raw or sRGB, auto
  meaning sRGB for 8-bit images.
- **Checked** (`tests/material/test_texture_store.cpp`): a decoded texture
  is the file to the last bit (0 of 3404 components differ), a mip chain's
  1x1 mean is the level 0 mean (0.49616 against 0.49804 raw, 0.30570
  against 0.30499 through sRGB), and UDIM tiles resolve or report missing.

### The lobe library

- **Lobes.** Oren-Nayar and its energy-compensated form (EON), Burley,
  translucent, dielectric (reflection, transmission, both), conductor,
  generalized Schlick with an F82 tint, and sheen in both the Imageworks and
  the Zeltner forms. Each has `eval`, `sample` and `pdf`; microfacets sample
  the visible normal distribution, transmission follows Walter, and sheen's
  albedo comes from an LTC fit.
- **The stack.** A material's lobes are built into a `LobeStack` and
  sampled with one-sample MIS, so a graph of any depth costs one sample.
- **Checked** (`tests/material/test_lobes.cpp`), all on the device: a
  chi-squared of sampled directions against the pdf (393.2 on 399 degrees of
  freedom for the diffuse lobes, 379.0 to 461.0 elsewhere), the pdf's
  integral against the fraction of samples drawn, and a white furnace where
  the albedo sampled and the albedo integrated uniformly agree to 3%.

### MaterialX into Slang

- **The generator** derives from MaterialX's own `SlangShaderGenerator`.
  Every node keeps its genglsl or genslang implementation except the ones
  that cannot mean here what they mean in a rasteriser:
  - the **surface** node, which has no light loop: its BSDF graph runs once,
    pushing lobes, and what it weights them by becomes the material's stack;
  - the **BSDF and EDF** nodes, which push lobes instead of responding to a
    light (`shaders/lrt/material/mx/`, declared in
    `lrt_genslang_closures.mtlx`);
  - the **image** nodes, which sample the texture table;
  - **heighttonormal**, which needs a screen derivative (below).
- **A BSDF value is a weight per built lobe**, not a response: `mix`,
  `layer`, `add` and `multiply` combine those weights the way genglsl
  combines responses, so a value used twice is two weightings of one lobe,
  not two lobes.
- **Uniforms are not baked in.** Every input is read from a float blob, and
  the module is named by a hash of its source: materials that differ only in
  values share one compiled module. `MaterialCompiler::parameters` lays a
  material's values, its textures' ids and its primvars' scene slots into
  that blob.
- **Sizes.** UsdPreviewSurface 580 lines and 23 blob words, standard_surface
  821 and 59, OpenPBR 945 and 55, glTF PBR 575 and 36, an unlit texture
  graph 242 and 22 (one texture, one primvar).
- **Checked against MaterialX itself.** The same graphs compile a second
  time in a reference variant whose closures are MaterialX's own genglsl
  responses; a kernel evaluates both for 65536 light directions. Worst
  component difference 3.2e-5 over eleven graphs, from a single
  `oren_nayar_diffuse_bsdf` to standard_surface with metalness, coat and
  sheen.
  - One difference is deliberate and aligned in the test: genglsl's layering
    scales the base by the top's Fresnel at the half vector, the lobes by
    the Fresnel at the view direction, which is what a sampler can carry.
  - A transmission-only scatter leaves the throughput at 1 in genglsl.

### Shading a frame, and who else evaluates a material

- **One generated module** (`technique::MaterialPrograms`) imports every
  compiled material and dispatches on a material row's function. Shading
  imports it to build a lobe stack; the visibility passes import it to ask
  whether a sample is there at all. It is named after the set it dispatches
  to, so a frame that shows the same materials compiles nothing.
- **The row** is the instance's (`InstanceRecord.flags >> 8`), unless the
  triangle is in a GeomSubset that binds one of its own
  (`triangleSubsets`, `subsetRows`).
- **The light** is still the headlight: a unit light from the eye, as
  `HeadlightShading` drew unshaded meshes. Scene lights are M5.

### Cutouts

MaterialX resolves `opacityThreshold` itself, so a UsdPreviewSurface that
has one leaves opacity at 0 or 1; what is left is deciding who evaluates it.
Shading cannot, because a sample cut away has to let what is behind it
through, so visibility does:

- the rasteriser draws with a generated fragment shader that discards;
- the two ray routes carry the ray on past the sample, up to sixteen times;
- only rows flagged as cutouts pay for the evaluation, and a frame with none
  runs the plain passes.

### Bump

`heighttonormal` -- and so `bump`, which is `heighttonormal` into
`normalmap` -- differences the height in screen space. A material here is
evaluated in a compute kernel, which has no `dFdx` (Slang has no such
identifier at all), so the difference comes from the thread's quad.

- **Which threads a quad holds was measured.** On this device the lanes are
  handed out along the group's rows, so four consecutive lanes were four
  pixels of one row and every vertical derivative was wrong -- all 15939
  threads of a test dispatch.
- **So the kernels walk their pixels in quad order** (`lrtQuadPixel`): each
  quad of four lanes covers a 2x2 block, and then bit 0 of the lane is x and
  bit 1 is y.
- **The limit is the quad.** Where one of its four threads shades something
  else -- a silhouette -- or leaves early, the derivative is of whatever it
  did evaluate. Measured: one pixel of a square's 360-pixel edge ring.
- `normalmap` needed nothing: the tangent frame (dP/du orthonormalised
  against the normal) was already in `MaterialInputs`.

### In Hydra

- The delegate has a material sprim, and asks for the `mtlx` and the
  universal render contexts.
- A `HdMaterialNetwork2` becomes a MaterialX document through `hdMtlx`,
  after two rewrites: the USD shading nodes are renamed to their nodedefs
  (`UsdPreviewSurface` to `ND_UsdPreviewSurface_surfaceshader`, and so on),
  and `UsdPrimvarReader` nodes become `geompropvalue` (varname to geomprop,
  fallback to default, result to out), whose primvar the generator needs as
  a constant.
- Materials compile when the render thread commits, and the primvars a
  material reads are added to the scene's primvar slots.

### How it is checked

- **Through USD** (`tests/usd/test_usd.cpp`): a MaterialX graph textured by
  an image, a UsdPreviewSurface with a UsdUVTexture read through a
  UsdPrimvarReader, a UsdPreviewSurface without specular, and a GeomSubset
  whose material shades its faces and the mesh's the rest. The analytic
  square (coverage, depth and colour) has 0 mismatches.
- **Raster against rays, materials included**: max 0.
- **Cutouts**: a square cut away shows the square behind it exactly as if it
  were alone (8464 pixels, 0 coverage and 0 colour mismatches) in all three
  routes; the same opacity above its threshold is not cut; and a square half
  cut by a texture's alpha is the same image whichever route drew it (max
  0).
- **Bump**: a height linear in u gives MaterialX's normal,
  (-k * scale / 16, 0, 1) normalised, over all 7921 interior pixels of a
  square; and the quad derivatives of a field linear in the pixel are its
  gradient for every thread of a 161 x 99 dispatch.

### Measured (M5 Pro, release)

- **Method.** `lrt view --frames 200 --size 1600x900`, draw median, as M3
  measured its frames: Hydra sync and drawing, no readback.

| Scene | Route | Draw | With the headlight (M3) |
|---|---|---|---|
| Kitchen_set | rays | 9.27 ms | 7.08 ms |
| Kitchen_set_instanced | rays | 9.36 ms | 7.15 ms |
| Kitchen_set | raster | 21.29 ms | -- |

  Raster's distance from rays is the one M2 measured: recording 1788 draws
  costs the host about 8 ms, which materials do not change.

- **What a material costs.** One UsdPreviewSurface (metallic 0.2, clearcoat
  0.5) over a quad filling 1600x900 draws in 29.9 ms. That is the lobe
  stack, not the textures: four lobes built and evaluated per pixel, in
  registers sized for sixteen.
- **What a cutout costs.** The same quad with an `opacityThreshold` that
  cuts nothing draws in 60.7 ms: the material is evaluated twice, once by
  visibility to decide the sample is there and once by shading. Nothing is
  carried between them.
- **First frame** (`lrt stage`, which compiles when the render thread
  commits): 0.33 s for that one material, and 1.74 s when a cutout pass has
  to be generated as well. Compiling is synchronous, and this is what that
  costs.

### Not done, not verified

- **Storm as an oracle for materials is not possible on this Mac.** Storm's
  own MaterialX shaders fail to compile in this build (undeclared `u_env*`
  in the generated MSL, with a lighting state and a dome light present), so
  the MaterialX TestSuite comparison is written but hidden
  (`[.][usd][gpu][oracle][storm-materialx]`).
- **Compiling is synchronous.** A material compiles when the render thread
  commits it, which stalls the first frame that shows it; the plan's
  placeholder and background compile are not done.
- **Transparency is not blended.** An opacity below 1 without a threshold
  weights the sample's colour but does not let what is behind it through:
  that is the path tracer's, M6.
- **Nodes whose genglsl uses a screen derivative do not compile** unless
  they have a genslang implementation here, which only `heighttonormal` has:
  `aastep` and the hextile nodes would fail on `dFdx`.
- **Displacement and volume terminals** are read and ignored.
- **Layering** uses the top's throughput at the view direction; directional
  albedo tables are not computed.
- **The lobe stack is not optimised.** Every material carries sixteen build
  lobes through registers whatever it uses, and a cutout evaluates its
  material a second time rather than keeping what visibility already found.
  Both are measured above and both are worth revisiting once lights (M5)
  settle what shading needs to keep.

## Complete USD: lights (M5)

A mesh is lit by what the stage authored: UsdLux lights reach the engine
through Hydra, and shading samples each one where it stands.

### What a light is

- **A record per light**, in world space and in the units USD authored
  (`modules/light`). What a record becomes is derived in the shader, not on
  the host: exposure, the blackbody of a colour temperature (Krystek's fit of
  the Planckian locus, normalised to luminance 1), and the area a `normalize`
  divides by.
- **Five kinds**: distant with an angular diameter, sphere, disk, rectangle
  and dome. A sphere of radius 0 and a sun of angle 0 are delta lights and
  carry no density.
- **Shaping** is the cone and its softness. IES profiles and cylinder lights
  are not read.

### How one is sampled

- **The cone it subtends** for a distant light and a sphere, uniformly in
  solid angle, which is the density a plane's closed-form irradiance is
  written against.
- **Its own surface** for a disk and a rectangle, uniformly in area, turned
  into a solid-angle density by the distance and the cosine at the light.
- **The surface being shaded** for a dome, cosine weighted: the light comes
  from the hemisphere above the surface, and sampling the whole sphere throws
  half the samples below the horizon -- 7.1% of noise against 0.02% on a
  plane under a constant dome, at the same count.
- **`lightPdf`** gives that density for any direction, not just for the
  sample drawn. The path tracer will weigh hits by it (M6); here it is what
  the chi-square compares against.

### Shadows

Where the device traces rays, a light that casts one is occluded by whatever
lies between the point and the sample. The ray leaves along itself as well as
along the normal, so its origin does not depend on a sign, and by a distance
that grows with the scene -- which costs contact: an occluder within that
offset is not seen. The structure is the scene's own, built for whatever
route drew the frame, and read after the visibility pass rather than before,
since the rays route rebuilds it there.

### The dome

A dome carries a lat-long image through the same texture table the materials
sample, mapped around the light's own axes. It is not a layer: it has no
depth, and giving it one would make the background read as covered, so it is
painted where the frame drew nothing, opaque, after everything else.

A dome follows its image's own brightness. The warp descends the mip chain
the texture store already built, choosing among a cell's children by
luminance, and its density needs no walk at all: a lat-long texel covers
2 pi^2 sin(theta) du dv, and a texel's share is its luminance over the
image's total, which the 1x1 level holds as an average -- so

    pdf = luminance / (average * 2 pi^2 * sin(theta))

Nothing is precomputed, and the choice between warping and sampling around
the surface is made by how much the image varies: the 1x1 level gives the
mean and a middle level the spread. A flat sky is better served by the
cosine, which the numbers below say plainly.

It took two bugs to get there, and the chi-square binned in the image itself
-- where the warp works, so no grid artefact could be blamed -- found both:

- **Splitting left from right and then top from bottom off one level** does
  not give the four children their own probabilities: z 61543 over a million
  samples. An explicit choice among four weights fixed it.
- **Splitting an axis that has no resolution left.** With that fixed a square
  image passed at once (z 0.84) while a 64 x 32 one still read z 1164589: a
  lat-long chain reaches one row while it still has columns, and from there a
  cell has two children rather than four, so probability was being handed to
  texels that are not there. Each level now splits only the axes that still
  divide.

What was never wrong, measured rather than assumed: the chain telescopes to
0.7% (a parent against its four children), uv survives a turn through a
direction exactly (0 of 4032), and every sample agrees with `lightPdf` --
that last one passed throughout, which is the lesson: a per-sample check
compares a density with itself and cannot see a sampler drawing the wrong
distribution.

- **Verified**: z 0.84 with a square image and 2.41 with a 2:1 one, a million
  samples each, against the density integrated over the same bins.
- **Against the cosine**: on a plane under a flat sky the warp is 20% out
  where the cosine is 0.08% at the same count, since a uv-uniform sample
  crowds the poles and drops the cosine. Which is why the rule picks by
  variation, and why a dome with a sun in it is the warp's case, not this
  one.

The three instruments that settled this stay: the chi-square bins a dome in
its image, a cone in its own solid angle and an area light on its own
surface; a round trip checks the mapping without statistics; and a pyramid
check measures whether the chain telescopes at all.

### The cylinder, added with M6

UsdLux's `CylinderLight`: the lateral surface of a cylinder along the prim's
x, of `radius` and `length`, emitting outward, one-sided. Sampled uniformly
on that surface (an angle about the axis, a height along it), with the pdf
of any area light, `d^2 / (cos * area)`; `lightPdf` finds where a direction
meets it by the ray's closest approach to the axis rather than the quadratic's
`b^2 - 4ac`, which cancels catastrophically beside a tangent ray -- measured:
7826 of a million samples disagreeing with their own pdf beyond 1e-3 with the
quadratic, 2378 with the closest-approach form, and what remained was the
conditioning of `1/cos` itself, derived in the check and guarded below a
cosine of 0.017, where the light arriving is of order 0.02% of the whole.
Checked four ways:

- **Chi-square in its own support** (angle by height on the surface, cells
  facing away expecting nothing): z 1.30, the pdf integrating to 0.4512
  against 0.4524 drawn, 0 samples disagreeing with `lightPdf`.
- **The sampler alone**, at three points, against dense quadrature of the
  form-factor integral: 0.04% to 0.15% apart.
- **A Lambert plane under it**, against two closed forms that share nothing
  -- 512 one-sided strips under Lambert's edge formula, and the quadrature --
  which agree with each other to 0.2%: 0 of 2209 pixels beyond 3% at 65536
  light samples. Not 4096 like the flat lights: a one-sided curved emitter
  rejects half its samples and varies over the rest, and the first run read
  3103 of 8281 pixels beyond 2% -- which was noise (sigma ~2% at 4096, 0.9%
  at 32768, 0.4% at 131072), not the bias it looked like, and the sampler
  alone is what said so.
- **Through Hydra**, the same.

Found on the way and fixed: shading's second random number was one LCG step
of the first, tying every sample pair to a lattice. It did not bias the
lights that were checked, but it is the path tracer's PCG chain now.

And one the closing plan found by reading: each light's cumulative share of
the frame's power was accumulated in a host loop -- the one piece of CPU
arithmetic on scene data left in the tree. It is a kernel now
(`light_prefix.slang`), the total is the last record's share where
`chooseLight` reads it, nothing comes back to the host, and the table takes
the shader library to make it. Checked by a kernel that writes the power a
second time from the record alone: five lights of mixed kinds, exposures and
`normalize`, 0 shares missing their power.

### IES profiles, added with M6

UsdLux's `ShapingAPI` IES: `shaping:ies:file`, `angleScale`, `normalize`.
`io::readIes` reads an LM-63 file as authored -- the vertical and horizontal
angle lists and the candela table, the multiplier, the photometric type --
and nothing is normalised, resampled or mirrored on the host: each of those
is arithmetic on the data. The frame's profiles are concatenated into one
values buffer with a record each (`IesRecord`), and the shader samples the
table where it samples the light: the emission direction in the light's own
axes, straight down its -Z at theta 0 as UsdLux orients a profile, bilinear
between the authored nodes, with the horizontal range folded by the symmetry
its last angle declares (one angle: rotational; 90: quadrant; 180: bilateral).
It modulates radiance only, never the density, so every pdf and every
chi-square stands as it was.

- **`angleScale`** as UsdLux defines it: positive divides theta, negative
  scales from 180 degrees, zero is none.
- **`normalize`** divides by the profile's power. UsdLux says the intensity is
  "scaled by the overall power of the IES profile ... integrating the
  luminous intensity over all solid angle patches", which leaves the constant
  open; here the power is that integral over 4 pi, so a normalised profile
  has mean intensity one over the sphere and a uniform profile is unchanged.
  The integral is a statistic, so `ies_prepare` computes it on the device at
  commit, over the patches the angle lists define, folded by the symmetry.

Checked three ways:

- **Every node returns its own candela**: 0 of 37 off, worst 2.2e-5 relative.
- **Between nodes, the closed form**: a profile of 1000 cos^4(theta) authored
  at five-degree nodes, sampled at 4096 directions against the formula. The
  error of piecewise-linear interpolation is bounded by h^2/8 max|f''| --
  0.0873^2 / 8 * 4000 = 3.8 -- and the worst read 3.77.
- **Through Hydra**, a cutoff profile (one to 20 degrees, zero from 25) on a
  small sphere light over the plane: inside the cone the frame is the frame
  without the profile, word for word (0 of 2785 pixels differ), and outside
  it is black (0 of 1488 lit). The band between is the profile's own ramp,
  20 to 25, spread by the sphere's angular radius of 1.43 degrees: a first
  check that skipped three degrees about the cutoff read 1626 lit pixels,
  all within 26.3 degrees and all the ramp's, and was wrong, not the light.

Not done: photometric types B and A are read but sampled as C; `TILT=<file>`
is treated as none; the splat relighting samples a light's centre without
its profile.

### Light instancing, added with M6

A light under an instancer is placed as a mesh under one: the delegate
walks the instancer chain above the sprim (`_UpdateInstancer`, the same
loop `Mesh.cpp` runs), and the engine composes it on the device with
`world::Instancing::compose`, unchanged, once per change of any level. The
light module sits below world, so `light::Light` carries the composed rows
raw -- `instanceRows`, a buffer of 3 float4 rows an instance, and
`instanceCount` -- and `LightTable::set` does bookkeeping only: it copies
the prototype's record once per instance and `light_instances.slang`
rewrites each copy's rows as the instance's rows times the prototype's own,
before `light_prefix` accumulates the power over the whole table. A light
whose instancer has not arrived is not drawn, as a mesh in that state is
not.

Checked twice, both exact. In the technique: a rect light that itself
rotates, under an instancer that rotates and takes four elements out of
order, against the four lights authored at `instancer * element * prototype`
-- `lightInstanceCheck` finds 0 of 6 records differing in rows, size or
cumulative power (a product in the wrong order shows, since both factors
rotate). Through Hydra: a `PointInstancer` whose prototype is a sphere light
at three positions over the plane, against the three lights authored one by
one, relMSE 0 -- Hydra delivers instanced lights in this install, which is
the half light linking is missing.

### In Hydra

The delegate takes sphere, disk, rect, distant, dome and cylinder lights as
sprims, and reads a light's IES profile where `ShapingAPI` authors one. A
light's samples per pixel are a render setting, `lrt:lightSamples`, reachable
from `StageRenderer` and from `lrt view --light-samples`: one is what an
interactive frame takes, and a comparison against a closed form asks for
enough that what is left is the light and not the noise.

### How it is checked

- **Closed forms that share no code with the renderer**
  (`lambert_irradiance.slang`): a Lambert plane under a sphere, a disk (as a
  512-gon), a rectangle (Lambert's formula over its edges), a sun and a dome,
  and a point light behind a square occluder whose umbra is exactly what it
  projects. 0 of 8281 pixels beyond 2% in each: worst 0.26% for the sphere,
  1.6% disk, 1.3% rect, 0.02% dome, and exact for the sun and the umbra.
- **A chi-square per light**, a million samples each, binned in the frame
  that matches the light's support -- a cone's own solid angle, an area
  light's own surface -- against `lightPdf` integrated over each bin: sphere
  z -0.23, disk 0.14, rect -0.27, sun -0.29, dome -1.61, every pdf
  integrating to 1.0000.
  - It carries a per-sample pass too: the density a sample reports against
    the density its own direction has. That is what MIS depends on, it needs
    no histogram, and it is what proved the large statistics were the binning
    rather than the sampling (worst disagreement 3e-7).
- **Through Hydra**: a UsdLuxSphereLight over a Lambert plane, 0 of 8281
  pixels beyond 2% with its centre at 0.03200 against the closed form's
  0.03200; and a dome light's image lighting the same plane to 0.08%, its
  background reading 0.6039 against the 0.6038 its PNG decodes to from sRGB.
  Each of those renders the same stage with no light first and checks it
  against the analytic headlight, exact to 1e-5, so the lit comparison is
  about the light and not the material.

### Bugs these found

- **The shading normal never faced the viewer.** `materialInputsAt` computed
  the backface and did not flip, so a front-facing square handed materials a
  normal pointing away. Shading hid it, since the lobes build their frame
  around the view direction; a shadow ray could not, and every ray hit the
  surface it left.
- **Shading traced against a freed structure.** It took the acceleration
  structure's pointer while preparing the frame, and in the rays route the
  visibility pass rebuilds it there -- releasing the one shading still
  pointed at. Two readings were wrong before that one, and measurement killed
  both.

### Measured (M5 Pro, release)

- **Method.** `lrt view --frames 200 --size 1600x900`, draw median.
- **Scene.** Kitchen_set with four lights (a dome, a rectangle and two
  spheres), authored beside it: the asset itself carries no UsdLux prim.

| What | Draw |
|---|---|
| Four lights, one sample, rays | 27.77 ms |
| The same, compute BVH | 35.14 ms |
| The same, raster | 40.94 ms |
| The same without shadows, rays | 26.69 ms |
| No lights at all (the headlight), rays | 10.61 ms |

- **Samples per light**, rays: 1 gives 27.79 ms, 4 gives 77.72 ms, 16 gives
  275.40 ms. Linear in samples times lights, since every light is sampled at
  every pixel: what a light BVH and MIS are for.

### Not done, not verified

- **A light can be chosen instead of visited.** Shading either loops over
  every light at every pixel -- exact, and the default, because at one sample
  it is the quieter of the two -- or draws one light a sample in proportion to
  its power (`lrt:chooseLights`, `lrt view --choose-lights`), dividing the
  density of that choice back out. The choice is what stops a pixel's cost
  growing with the number of lights; what it costs is noise a frame has to
  average away. There is still no light BVH, which is what the choice would
  need to stay cheap at thousands of lights.
  - **Checked by three sphere lights in the same place**, of intensity 1, 2
    and 3: one light of six times the power, analytically, with a
    distribution over them that is not uniform -- which a single light can
    never exercise. Both ways, 0 of 8281 pixels beyond 3%, worst 0.21% for
    the loop and 0.26% for the choice.
  - **Measured** on Kitchen_set with four lights, 1600x900, draw medians: the
    loop takes 28.49, 79.61 and 282.21 ms at 1, 4 and 16 samples per light;
    the choice takes 11.87, 12.18 and 13.22 ms, over a 10.61 ms frame with no
    lights at all. Flat, because the cost of a sample is small beside the
    frame it sits in -- which is also why the loop is affordable at one
    sample and the default.
- **No MIS.** Lights are sampled, the material is not sampled back at them.
  That is the path tracer's, M6.
- **The two dome densities are not combined.** A dome is sampled either by
  its image or around the surface, whichever its variation calls for, and
  never both with MIS weighing between them: that is the path tracer's, M6.
- **Light linking works in the engine and not through USD.** The engine's
  half is exact: an instance carries a 64-bit mask of its categories (the
  record grew to 192 bytes, and a set record spends two spare words to carry
  the same mask through the instances the device writes), a light carries the
  category it lights, and shading skips a light the surface does not carry --
  checked by counters rather than a tolerance, two squares of different
  categories with 6150 pixels each, the linked one wholly lit and the other
  exactly zero, both lit when the light has no collection.
  - **What does not arrive is the scene index's half**, measured in this
    order rather than guessed: `HdsiLightLinkingSceneIndex` is registered
    from a point every host reaches -- a registry function alone never runs,
    since a host that builds the delegate itself never goes through plug's
    discovery -- and it is appended to the chain (traced); it is given ten
    light types and five geometry types, so its defaults are not the
    obstacle; the stage's collection transports correctly, but only in
    *expression mode* (`membershipExpression='/Left'` reaches the light's
    collections data source, where relationship mode sends UsdLux's default
    `~//*.*`); and the mesh carries a `categories` data source while the
    light carries `lightLink` -- both empty, before the stage is synced and
    after, with the filter inserted first in the chain and last. Whatever
    makes that filter mark a prim is not happening here, and its
    implementation is headers only in this install. The USD case is written
    and hidden (`[.][usd][gpu][mesh][lights][linking]`) with that list in it.
  - **Found at M9, with OpenUSD's sources on the machine.** The filter
    builds its collection cache in `_PrimsAdded`, from the added-prim
    notices that pass through it, and nowhere else. `StageRenderer` handed
    the stage to `UsdImagingCreateSceneIndices`, which populates on the spot,
    before this renderer's filters were appended: a filtering scene index
    made after its input populated never hears of the prims already there,
    and the render index then read them through `GetPrim` from an empty
    cache. Nothing on the list above could have seen it, because every
    data source was right. The chain is now built empty, inserted, and given
    the stage after. The USD case is no longer hidden: the square in the
    light's collection lit over its 6150 pixels, the other drawn and 0 lit.
- **Shadow linking is honoured in the trace.** A light with a shadow link
  walks its ray on past whatever does not carry that category, as a cutout
  walks past what its opacity removed, up to sixteen times; a light without
  one keeps the cheap first-hit query. Checked both ways against the closed
  form: with the link naming the occluder's category the umbra is exactly
  where it projects, and with it naming another the plane is lit as if
  nothing were there -- 0 pixels of 7440 away from the closed form either
  way. From USD a shadow link comes through the same filter as a light
  link, which was missing for the reason found at M9 (above). Checked from
  USD too ("a UsdLux light's shadowLink collection decides what casts its
  shadow"): a sphere light and an occluder, both above the frame, over a
  floor. With the collection left whole the occluder's shadow changes 9108
  words of 24576 under raster and 12396 under rt (no bounces); with
  `collection:shadowLink` naming `/Floor` alone the frame is bit for bit
  the frame without the occluder, 0 words, under both.
- **Light instancing arrived with M6**, below. (So did the cylinder and IES
  profiles.)
- **Splats are relit where their prim asks**, and baked everywhere else.
  `LrtSplatLightingAPI` (`primvars:lrt:splat:relight`, a constant primvar, so
  it is inherited) turns a cloud over to the scene's lights: the albedo is the
  harmonics' constant term, the normal is the splat's shortest axis turned
  towards the eye, and light linking reaches a cloud by the same bit it
  reaches a mesh.
  - **What it is not**: one sample at each light's centre, no shadow ray, no
    second sample, and a normal a splat never had. It is for a capture that
    has to sit under different light, and wrong wherever the capture's own
    light was the point -- which is why baked is the default.
  - **What made it possible**: the light module is two, a core that reads no
    texture and `lights_image` on top. A dome's image comes through the
    material texture table, and the splat projection lives below material in
    the module order, so before the split the lights were simply out of its
    reach.
  - **Checked** three ways at once: with relighting off the frame is
    identical to the one that never had the feature (max 0), with it on the
    picture changes (max 185 over 9216 pixels), and a light whose collection
    does not include the cloud lights none of it (max 18 against the relit
    frame).
  - **Not measured.** What relighting costs against showing what was baked has
    no number here: it wants a stage with both a cloud and lights, and there
    is no splat asset on this machine to build one from -- the clouds the tests
    use are synthesised in memory and never reach the command line.
- **Contact shadows** closer than the ray's offset are missed, and a cutout
  material still stops a shadow ray where its opacity would have let it
  through.

## Complete USD: the path tracer (M6)

Paths over the same visibility buffer the raster shading reads, so the two
can be told apart by exactly one thing: the bounces -- and, with a lens, by
the tracer's own primary rays. The milestone's five checks are each in this
section: the white furnace (the closed emissive shell, exact to 1.7e-7 over
0 to 6 bounces), the path tracer against the raster shading where the bounce
contributes nothing, the error falling as one over root N (exponent -0.499),
splats alone under `rt` through the ray tracer, and OIDN lowering the error
(3.65e-4 to 7.17e-5). What was folded into it from M5's deferrals -- the
cylinder, IES profiles, light instancing, the power prefix on the device --
is in the lights section.

### What it does

- **The same surface, the same material, the same light.** A hit is rebuilt by
  `material_surface.slang`, its material evaluated into the same lobe stack,
  and its direct light gathered by next event estimation with the light chosen
  by power -- all of it the machinery M5 left behind.
- **No MIS, on purpose.** Next event estimation covers the analytic lights of
  the light table and sampling the material covers emissive geometry: disjoint
  sets, so neither strategy is weighed against the other. There was a power
  heuristic here, and it was wrong -- see below.
- **The bounce** samples the material (`stackSample`), traces where it points,
  shades what it lands on, and carries that surface's emission and direct
  light back through the path's throughput.
- **Accumulation** is a running mean of each sample's colour times its
  opacity: a call adds its samples to a sum and says how many the frame holds,
  which is what a progressive render needs.
- **The sampler** folds pixel, stream, absolute path index, bounce and
  dimension through a PCG hash one after another. The path index is absolute
  (`accumulated + sample`), so a frame gathered in one pass and in many draws
  the same samples: 256 paths in one pass against 64 in each of four differ by
  relMSE 3.9e-15, accumulation order alone.
- **Where the device does not trace**, there is no bounce to trace: the kernel
  is generated without one and gathers direct light alone.

### Where it runs from

The `rt` technique used to trace splats and return, which left every mesh and
every material out of a traced frame. Now it returns early only where there is
nothing to compose under: a frame of splats alone is still `GaussianRayTracer`
writing the whole image, at the tolerances `test_ray_tracing` already held it
to. With meshes in the frame the surfaces are path traced and the splats
composed over them by the rasteriser, because the tracer takes no `under`
layer -- splats inside the rays is still to be written.

A path traced surface gets an acceleration structure whatever the lights do,
since its bounce is a ray. Shading needs one only where a light casts a shadow,
and the same structure serves both; without that, a traced frame would trace
against nothing and no test would say so.

`lrt:pathSamples` is how many paths a pixel a pass gathers and
`lrt:pathBounces` how many bounces each takes after the first hit, one of each
by default -- what an interactive frame affords.

### Gathering a frame over several passes

`lrt:pathTotal` is the paths a pixel at which a frame is finished; one, the
default, never accumulates, so nothing that worked before behaves differently.
Above one, drawing the same frame again adds its paths to the running mean and
the pass reports itself unconverged until the total is reached, which is what
makes a viewport quieten down while it is left alone.

What counts as "the same frame" is the engine's to decide, since it is the only
place that sees both the camera and the scene: it remembers the camera element
by element (`Mat4` has no comparison of its own), the frame's size, the samples
and bounces, and a revision. The revision is what says the scene itself moved
-- `commit` raises it whenever it uploads anything, and so does every setting
that changes what a path would find, each of those only when the value really
changes, so a host that re-sends the same settings every frame does not reset
the mean. Moving the finish line is the exception: `lrt:pathTotal` leaves what
has been gathered still valid.

The render buffer reports convergence from the engine too. It used to answer
"always converged", which was true while there was no progressive mode and
would now let a host stop asking for the rest of a frame the pass had not
finished.

### How it is checked

- **A white furnace** under an imageless dome, path traced against unweighted
  NEE: p99 relative 0.0000, max 0.0003 at 4096 light samples against 4096
  paths. The one the MIS weight failed.
- **A closed emissive shell** reads its geometric series exactly: 0 of 3072
  pixels beyond 1e-4 at 0, 1, 2, 3 and 6 bounces, worst 1.7e-7. The plan's
  first check, and the only one on more than one bounce.
- **One bounce against the raster's direct light**, in a scene with nothing
  for a bounce to find: p99 1 and max 1, with no pixel beyond 2, over 4096
  accumulated paths. That is the plan's check, and it holds the two
  estimators to each other rather than to a tolerance of their own.
- **The error falls as 1/sqrt(N)**, measured without a reference: pairs of
  independent estimates at 64 to 1024 paths, six pairs a point, the median
  of their mean squared difference, and a least-squares exponent on the
  square root: -0.499, window -0.42 to -0.58 calibrated as above. The
  plan asked for a 64k spp reference; measured, that is ~80 s for one scene
  at the other cases' resolution, and depth of reference is not what verifies
  a law -- the ratio between points is, and a reference only adds a floor.
- **The sampler**, three ways: the same 256 paths in one, four and sixteen
  passes agree to 3.9e-15; no two of 3072 pixels share a sample sequence or a
  sample set, at the seeds the ladder used; and block-averaged errors fall as
  independent errors do (4.42x and 15.94x for 2x2 and 4x4).
- **A frame of splats alone through `rt` is `GaussianRayTracer`'s**, the
  plan's fourth check: the Hydra case that renders a cloud through the
  delegate's `lrt:technique` against the ray tracer called directly has held
  it at p99 1 and max 2 since M0, and the engine's early return is what keeps
  it true.
- **The bounce carries light from a second surface.** The check above proves
  the bounce takes nothing away where there is nothing to find -- which is
  also exactly what an unbound acceleration structure would look like, and
  that test binds none. So a wall stands along a plane's edge, turned to face
  it, and the same frame is held at nought bounces against itself at one, over
  the same seeds: the direct term is identical, so what is left between them is
  the bounce alone. p99 41 and max 73 over 4688 pixels. The control is the
  scene without the wall, where the two come out at max 0 -- identical frames,
  which is what makes the difference the bounce and not the noise. (It read
  p99 41, max 73 while the bounce carried rho pi / cos; p99 10, max 13 now.)
- **The traced technique over a mesh, through Hydra.** A plane under one sphere
  light has nothing for a bounce to find, so the traced frame and the raster
  frame are two estimators of the same direct light: p99 1 and max 1 at 1024
  paths, with the surface drawn exactly where the surface is (8281 pixels
  covered, no coverage mismatch). Only the coverage is read from
  `squareMismatches` there: it compares colour against albedo times the cosine
  to the eye, which is the headlight's answer and not a lit scene's -- reading
  its colour count as a verdict on a light would have been reading the wrong
  oracle, and the sphere light's closed form is what the M5 case checks with
  `lrt/test/lambert_irradiance`.
- **The accumulation, over the whole chain**: settings, render pass, mean.
  Eight passes of four paths hold 4, 8, 12, 16, 20, 24, 28 and 32 and then
  report converged; a camera somewhere else drops back to 4 and unconverged,
  and changing the bounce count cuts 16 back to 4. That second half is what
  says the revision is armed rather than decorative: a revision nothing raised
  would go on averaging over a scene that had changed, and no image would look
  wrong enough to say so.

### What was wrong, and what was not

Two defects, found by reading before any new test was run, and then measured.

- **The MIS weight was one-sided and lost half the dome.** `gatherLight`
  weighed every non-delta light sample by the power heuristic against the
  material's pdf -- but the strategy it was sharing with never covered an
  analytic light: sampling the material collects emission from geometry, a
  table light has none, and a bounce ray that escapes broke without gathering
  the dome it passed through. The estimator was scaled down and nothing paid
  the remainder. For an imageless dome and a Lambert lobe the two densities
  are the same function (`cos/pi`), so the weight was exactly one half.
  Predicted and then measured: a white furnace, path traced against
  MaterialShading's unweighted NEE, read p99 relative **0.5453** before and
  **0.0000** (relMSE 1.3e-11, max 0.0003) after. Every earlier path check had
  passed because it used a sphere of radius 0.4 at distance 2, where the
  weight is 0.9984 -- invisible at p99 1 in eight bits. That is the "two
  estimators wrong in the same way" the bounce test warns about, met in the
  flesh.
- **The accumulation was a product of means.** `mean(colour) * mean(alpha)`
  rather than `mean(colour * alpha)`: identical while every sample is opaque,
  which was every test, and biased the moment opacity varied.

**Two more in the bounce, found by the first check that ever exercised more
than one.** A closed emissive shell -- every point emits E and reflects rho,
seen from inside -- must read E (1 + rho + ... + rho^N) after N bounces, and
with cosine sampling of a Lambert lobe each bounce's weight over pdf is rho
with no variance at all, so the check is exact, not statistical. It read 5 pi
times the series at one bounce. Two causes: `throughput *= weight / pdf`,
where `LobeSample.weight` is by its own contract already `f |cos| / pdf`, so
the bounce carried rho pi / cos instead of rho; and `shadeHit` rebuilt the
bounce's hit by calling `shadeAt`, which re-intersects the *camera's* ray with
the triangle the bounce found -- a point not on the bounce ray at all, with
barycentrics and a normal to match. The hit is now rebuilt from the ray
query's committed barycentrics (`surfaceFromWeights`, which `surfaceAt` now
shares), with the bounce's own direction deciding which side it arrived at.
Shell: 0 of 3072 pixels beyond 1e-4 at 0, 1, 2, 3 and 6 bounces, worst 1.7e-7.
A box was tried first and leaks at its edges -- the tracer's origin offset
`p + (n + wi) * 1e-3 scale` puts a ray leaving a face beside an edge outside
the box, where the next face is a back face and culled: one sample in sixteen
short in 25 pixels, worst exactly rho^2/(1 + rho + rho^2)/16 -- which is the
test's geometry, not the integrator's, and the reason the shell is a sphere.

**And the 1/sqrt(N) anomaly, which those two explain.** The ladder had read
exponents of -0.43 to -0.45 through five different instruments, and one point
had spread 3.4x between draws. Fourteen hypotheses were killed by measurement
first, in this order: the running mean (read: algebraically right); the hash
(read: a full avalanche); overlapping seeds (computed: disjoint); the split
into passes (the invariant `1x256 = 4x64 = 16x16` holds to 3.9e-15); the MIS
weight and the premultiply (fixed: the ladder did not move); pixels sharing a
sample *sequence* (fingerprints sorted: 0 of 3072); pixels sharing a sample
*set* in another order (a commutative fingerprint: 0 of 3072); neighbours'
errors correlated (block variance fell 4.42x at 2x2 and 15.94x at 4x4, against
4 and 16); the sampler (replaced by the PCG chain: -0.435 before and after);
the metric (relMSE's `1/(b^2 + 1e-2)` weight has a heavy tail: a plain mean
square instead); the reference (dropped: two independent estimates at the same
N have `E[(a - b)^2] = 2 Var`, no floor by construction); the mean over pairs
(the median); and the box's own seams. Each of those was worth doing and none
was the cause. The cause was the double division: rho pi / cos has a finite
mean under cosine sampling but an infinite second moment, so the bounce's
contribution had infinite variance and a mean of N of them does not tighten as
1/sqrt(N). The ladder was right to complain and the shell found why. With the
bounce fixed the same ladder fits **-0.499**. The 3.4x spread was four draws
of a heavy-tailed statistic read as a switch; the window is calibrated to the
measured scatter and still rejects no convergence, a floor and a linear law.

### The denoiser, and what it took to hand OIDN a buffer

`technique::Denoiser` denoises now. OIDN's `RT` filter runs on the engine's
own Metal queue, over the engine's own buffers, and nothing crosses to the
host. What that took, in the order it was found:

- **OIDN shares only Metal buffers with hazard tracking**, and slang-rhi makes
  none: its Metal backend forbids `MTLHazardTrackingModeTracked` on every
  resource and orders its own work. So each image goes through a staging
  buffer the platform makes tracked -- `platform::newTrackedMetalBuffer`, one
  Objective-C call in `core/Platform`, beside `matchLayerToBacking`, which is
  the precedent and the rule -- wrapped for slang-rhi with `Buffer::wrap` and
  shared with OIDN once.
- **slang-rhi's Metal `copyBuffer` does nothing with a wrapped buffer on
  either side.** Silently: 4095 of 4096 words untouched both ways, while a
  kernel reads and writes the same buffer exactly (0 of 4096). Measured by a
  test that stays in the tree (`a tracked Metal buffer is read and written by
  kernels, and not by slang-rhi's blit`). The staging copies are therefore a
  kernel, `buffer_copy.slang`, word by word.
- **OIDN writes three of a pixel's four floats.** The alpha in a fresh private
  buffer is whatever was there, and `compareHdr` compares four channels: the
  first working run read relMSE 0.15 against a reference at 3.7e-4 for the
  noisy input, which is what garbage alpha looks like. The output staging is
  seeded from the input, so the alpha that comes back is the input's.
- On CUDA the engine's buffers are shared directly (`oidnNewSharedBuffer` on
  the pointer); no staging, no copies.

**Checked**: sixteen paths against a 4096-path reference, relMSE 3.65e-4
noisy, **7.17e-5** denoised with the first hit's albedo and normal, 6.27e-5
without. The plan's fifth check. On a flat Lambert plane with one wall the
unguided filter does a little better; that is printed, not asserted, since
nothing says the guides must win on such a scene. The test skips where OIDN is
not built or the device will not open it -- never a CPU fallback.

From the engine, `lrt:denoise` runs it over a path traced frame once the
frame has gathered `lrt:pathTotal` -- every frame when the total is one -- in
place over the mean, after the frame's batch and never inside it, since OIDN
submits work of its own and waits. Checked through Hydra by the gate: with the
setting and the total reached, 8748 of 27648 words of the frame change; with
the setting and the total not reached, none.

Not done here: un-premultiplying the colour before the filter and
re-premultiplying after (the scene's opacity is 1 everywhere a test looks).

### Adaptive sampling

A pixel keeps its luminance's second moment beside its sum, and stops taking
paths once the relative standard error of its mean -- `sqrt((E[l^2] - E[l]^2)
/ N) / mean` -- falls below a target after at least `minSamples` paths. The
decision is a kernel of its own after each pass (`pathDecide`), one thread a
pixel, which also counts the covered and the stopped pixels by atomics; the
trace kernel skips a stopped pixel. The frame is gathered when every covered
pixel has stopped or `lrt:pathTotal` is reached, whichever first
(`lrt:pathAdaptive`, `lrt:pathError`).

What matters is not that it stops but that the estimate is truthful, and the
check was built to separate two questions: whether the *means* are right, and
whether each pixel's *own error estimate* is. Against a 4096-path reference,
with the reference's own moments kept for the true per-sample spread:

- **The means are right.** 10 of 4212 stopped pixels beyond three true
  standard errors at a minimum of 16 paths (0.24%, worst 4.0 sigma) and 15 at
  a minimum of 64 (0.36%, worst 4.5) -- three sigma leaves 0.27% by chance.
- **A pixel's own estimate is optimistic where it has not yet seen what is
  rare.** Under a bright bounce that a pixel meets in one path in a hundred,
  its first N paths may all miss it, and the spread they show is the direct
  light's alone, a hundred times too small. Measured before any remedy: 526 of
  4212 stopped pixels beyond three of their own sigma, worst 133, every one
  of them below the reference. The remedy is the standard one: the variance a
  pixel stops on is the larger of its own and the mean of its 3x3
  neighbours', since a neighbour that did see the event stands in. After it:
  100 beyond three of their own sigma (2.4%), and 14 (0.3%) with an estimate
  more than threefold optimistic against the truth. Raising the minimum to 64
  does not move that much (96 and 8): the residual is pixels whose whole
  neighbourhood missed the event, and no per-pixel statistic can see it. That
  is the method's known weakness, and the test bounds it at what it measures.
- Second moments never fall below the mean squared (0 of 4212), and the
  1/sqrt(N) ladder pins `adaptive = false`, since a sampler built to beat the
  law would break its window from the other side.

Through Hydra, an image with `lrt:pathAdaptive` at 10% and a total of 100000
gathers 16 paths a pixel and reports itself converged.

### Three things the ground did not turn out to be

Measured while surveying, and worth writing down because each one changes what
the rest of M6 has to build:

- **`technique::Denoiser` is a presence check, not a denoiser.** It has
  `create` and `description` and nothing else: OIDN is available, not applied.
  Denoising is to be written, not wired.
- **`ReferenceRenderer` is a reference for splats**, projecting and blending
  clouds and points. The plan's "error against a 64k spp GPU reference falls
  as 1/sqrt(N)" cannot lean on it: the path tracer will have to accumulate its
  own reference.
- **`rt_integrate.slang` is a splat integrator**, with an ordered record per
  ray and overlap windows. It is what "splats in rays" will reuse, and it is
  not a skeleton for a surface path tracer.

### The camera's lens

Exposure, the diaphragm and radial distortion, all from `UsdGeomCamera`
through `HdCamera`. Exposure scales the composed frame by `2^stops` once,
after the domes -- everything the camera sees, and no AOV -- checked exact
through Hydra (the frame with exposure 1 authored is the plain frame doubled
on the device, 0 of 27648 words apart).

The other two are not a parameter away: a ray through the aperture, or a
distorted one, no longer passes through the pixel's centre, so it cannot
ride on the visibility buffer the path tracer shades from. When the
projection carries a lens radius (`focalLength / (2 fStop)`, focal in
`HdCamera`'s scene units) or a `k1`/`k2`, the tracer casts its own primary
ray a sample (`shadeLensSample`): the pixel's ray, its x/y scaled by
`1 + k1 r^2 + k2 r^4` in ndc radius, then bent by a thin lens -- every ray
through the pixel meets the pixel's ray at the depth in focus, and leaves
the lens from a point drawn uniformly on its disc -- traced with the same
query the bounces use and shaded by `shadeHit` from where it met the
triangle. A lens ray that finds nothing is a transparent sample, counted.
The aux carry the first sample's hit. The engine's path state carries the
lens, so changing it restarts the accumulation.

Checked against closed forms that share nothing with the tracer
(`dof_check.slang`):

- **In focus is the pinhole.** A uniformly lit Lambert plane at the focus
  distance, its edge off centre, lens radius 0.2: relMSE 0 against the
  pinhole frame, bit for bit -- every lens ray through a pixel meets that
  pixel's ray there.
- **Out of focus is a circular segment.** The plane twice as far: its edge
  is blurred by the lens disc projected, a uniform disc of
  `lensRadius |z - f| / (z f) focal` pixels (5.73 here), so a pixel at signed
  distance d from the edge reads the fraction of that disc on the lit side,
  `(R^2 acos(-d/R) + d sqrt(R^2 - d^2)) / (pi R^2)`. At 4096 lens rays a
  pixel, 0 of 4235 pixels within three radii of the edge beyond 4% of the
  profile (worst 2.2%; the coverage's standard error is at most 0.8%).
- **Distortion moves the edge to the pixel.** `k1 0.5, k2 -0.2`, the edge
  off centre so the radial term shows: in each of 121 rows the lit pixels
  are exactly those whose distorted ray meets the plane on the lit side --
  0 rows off at all, all 121 moved by the distortion.
- **Through Hydra**: `fStop 8, focusDistance 5` on the square at 5 gives the
  pinhole frame exactly; `focusDistance 2.5` and `lensDistortion:k1 0.3`
  each change it (relMSE 0.44 and 6.8). The frame is 160 wide on purpose:
  at 161 the pixel centres lay on the square's triangle seam, and one lens
  ray in four converging exactly on the seam fell through it.

**A bug the Hydra check found, in the sun.** A distant light with an angle
handed its intensity out as the disc's radiance, so the irradiance it laid
was `intensity * solid angle`: UsdLux's default 0.53 degree sun lit a plane
6.7e-5 of its intensity, 15000 times short, while a sun of angle 0 -- the
one the closed forms had been checking -- was right. Now the disc's radiance
is `intensity / solid angle`, and the closed form takes a cap's vector
irradiance, `pi sin^2(a)` along its axis, over that solid angle: the 0.2
radian sun reads 0 of 8281 pixels beyond 2%, worst 0.01%.

### Not done

- There is no `HdRenderThread`: the pass draws on the thread that executes
  it. `StageRenderer::render` does draw until the path traced frame holds its
  total (checked: a total of 32 at 4 a pass leaves 32 gathered), so an image
  from the CLI is a gathered one; a viewport is the host's to keep asking for.
- **Splats inside the path tracer's rays.** A splats-only stage under `rt`
  goes whole to `GaussianRayTracer` (checked, with `test_ray_tracing`'s
  tolerances); with meshes the splats are composited over the path traced
  surfaces by depth. A bounce ray does not see them: `rt_integrate.slang`
  owns its pixel and assumes a primary ray, and a splat's contribution along
  a secondary ray is an integral through its Gaussian that no route here
  evaluates yet.
- **Points as spheres**, spiked and not built. slang-rhi's Metal backend
  does build acceleration structures over AABBs
  (`AccelerationStructureBuildInputType::ProceduralPrimitives`, a
  `BoundingBoxGeometryDescriptor` each) and OptiX does too; `Spheres` and
  `LinearSweptSpheres` it refuses on Metal. So a sphere primitive is an AABB
  with a custom intersection under `RayQuery`, on both devices. What it needs
  beyond that is a second primitive type in the visibility buffer and in
  `surface.slang` -- the same thing M8's curves add, which is where it goes.
- The lens under `raster`. Depth of field and distortion are the path
  tracer's (above): the raster route shades the visibility buffer's hit,
  which is the pixel centre's, and a viewport under `raster` draws a pinhole
  whatever the camera authors.
- Lens distortion beyond the radial terms: `lensDistortion:center`, `anaSq`,
  `asym` and `scale` are read by `HdCamera` and not applied.
- The plan's per-milestone `lrt bench` condition is retired, in CLAUDE.md as
  well: `lrt bench` times splat files and never rendered a stage, so the
  condition had been unmet since meshes arrived. Medians of `lrt stage
  --frames` and `lrt view --frames` are what is recorded, where there is
  something to compare against.
- **Real MIS**, for when the two strategies overlap: mesh lights. It needs a
  "does this direction reach light k, and with what radiance" beside
  `lightPdf`, which does not exist. Until then the disjointness above is the
  argument, and a weight here would be the defect again.
## Complete USD: animation and movement (M7)

### Deformation in place, and refit instead of rebuild

Until this, a mesh whose points changed was a new `GpuMesh`, and a new mesh
in the set meant `GpuScene::repack` -- every pool reallocated and copied,
every bottom-level structure and every LBVH built again -- once a frame for
anything animated. The plan named a latent bug here (a mesh deformed in
place would have left the structures stale, since they key on
`generation()` alone); the tree never deformed in place, so the bug never
fired, and the cost stood in for it.

Now a mesh carries a **topology key** (`MeshInput::topology`, kept by the
caller: the engine gives a prim a new key when Hydra marks its topology
dirty and keeps it otherwise). `GpuScene::update` takes a mesh set in which
every slot holds the same mesh or one of the same key and layout (counts,
subsets, primvar names, interpolations, components and counts) as a
**deformation**: the new positions and primvar values are copied over the
old in the pools, the record's box is rewritten, `positionsRevision()` and
that mesh's `meshRevision(k)` rise, and `generation()` does not. Anything
else repacks as before.

What is built on the pools follows the revision. `RayTracingScene` builds
its bottom levels `AllowUpdate` and keeps one scratch of the largest update
size; a mesh whose revision moved is refit in place
(`AccelerationStructureBuildMode::Update`, source and destination the same
structure), one submit each since they share the scratch. `BvhScene` keeps
each mesh's build parameters, recomputes its leaves' boxes from the pool's
positions and settles the internal boxes over the same tree (`bvh_refit`,
the passes already written for the build): the tree keeps the shape the
old positions gave it, so its boxes get looser and never wrong, until the
next repack reshapes it. Both take a `refit` flag whose false leaves a
deformed mesh's structure as it was -- there for the check below, not for a
caller.

**Checked** by what a deformation changes, in `test_visibility` on the
bumpy grid built three times under one key: after a deformation the scene's
generation stands and its positions revision is 1; rays and the compute
walker see the triangles the rasteriser sees (0 of the interior pixels
differ, the rasteriser reading the pool directly); `bvh_check.slang` finds
0 of 2047 LBVH nodes whose box misses a child's (a leaf's box being the one
its triangle's positions make now). With the refit skipped on purpose the
same comparisons say so -- 5378 and 1918 pixels differ, 1143 nodes miss a
child -- and the next build with the refit allowed catches up to 0 again.
Through Hydra, a sheet with time-sampled points: at the second time Hydra
hands new points and the same topology, the engine keeps the key, the
generation stands and the revision rises (`StageRenderer::meshGeneration`
and `meshPositionsRevision`), 28042 of 30000 pixels change, and the three
routes agree on the deformed frame to the same 4 edge pixels the flat one
allows.

**Not done here**: the top level is still rebuilt every call (it is small,
and instances move every frame); a mesh with changed topology still
repacks every pool, not only its own.

### Motion blur: the shutter in buckets, and time samples from Hydra

**What a bucket is.** Metal has no acceleration structure with motion in
it, and a ray query cannot be handed a time; so the shutter is cut into
`buckets` slices (1 to 8), and each slice gets what the scene looks like
at its centre. One top-level structure holds every slice at once: a moving
instance appears once per slice, its instance mask one bit (`1 << b`) and
its transform interpolated to the slice's time; a still instance appears
once, answering to every bit. A path draws a time per sample, takes the
slice it falls in, and every ray of that path -- primary, shadow, bounce
-- traces with that slice's mask. Eight bits of mask are why eight is the
most. Between the two shutter samples everything is linear: a transform's
rows (exact for a translation, an approximation for a turn) and a point.

**Where it lives.** `MeshInstance::motion` (`MeshMotion`: the transform at
the shutter's open and close, and meshes built from the points there when
it deforms, under the instance's own topology key). `GpuScene::update`
takes the bucket count; when something moves the records get a copy per
slice after the frame's own (`motion.slang`: `motionRecords` writes them
on the device, view and normal matrices included), `tlasFirst`/`tlasCount`
say which records the structure holds, and the first `instanceCount` stay
what the rasteriser and the compute BVH draw -- the frame at the frame's
time, without blur. A deforming mesh's positions are laid out once per
slice in the pool (`pointsStride` apart, `positionsLerp` between its two
meshes), its bottom level is built once per slice over that slice's
positions, and its records carry `pointsOffset` so the surface is rebuilt
from the slice's positions (`InstanceRecord.mask` and `pointsOffset` took
the record's two pads). `RayTracingScene` keeps one handle entry a slice a
mesh, a still mesh's all the same, and `instance_descs` takes the slice
from the record's mask. The path tracer casts its own primary rays under
motion, as it does under a lens.

**Checked against the staircase the buckets make** (`motion_check.slang`):
a uniformly lit plane whose edge slides 57 pixels along x over the
shutter, so a pixel's coverage is the fraction of slices at whose centre
the edge is past it. At 1024 samples a pixel and eight slices, 0 of 7381
pixels beyond 8% of the staircase (worst 5.2%; the coverage's standard
error is at most 1.6%) -- the same with the plane's points sliding under a
still transform, which exercises the per-slice positions and bottom levels;
the same at two slices; and with no motion under eight slices the frame is
the still frame bit for bit.

**Through Hydra.** The camera's `shutter:open`/`shutter:close` reach the
delegate through `HdLrtRenderParam` -- set by the pass from the `HdCamera`
it draws, and by `StageRenderer::aim` from the stage ahead of the first
Sync, since Sync runs before the pass and a shutter learnt there is a
frame late; when the pass finds it changed it marks every rprim's
transform and points dirty. With a shutter open for a while, `HdLrtMesh`
samples the transform (`SampleTransform`) and the points (`SamplePrimvar`)
about its open and close as well as at the frame, and the engine builds the
shutter's meshes under the same topology key. What Hydra hands back are the
**authored samples that bracket the shutter, at their own times** -- a
stage with samples at frames 0 and 1 drawn at 0.5 under a shutter of a
quarter frame either way returns the samples at -0.5 and +0.5 -- not the
values at the shutter's ends; so each sample's time travels with it
(`MeshMotion::timeStart`/`timeEnd`, `MeshTransforms`, the points' times)
and the device places each bucket's centre between them (`bucketFactor`).
A first version took the two samples for the shutter's ends and blurred
over the whole frame. `lrt:motionBuckets` (default 4;
`StageRenderer::setMotionBuckets`, `lrt stage --motion-buckets`) is the
slice count. Checked with a square sliding between two frames under a
shutter of half a frame about frame 0.5, path traced in eight slices:
against the same stage with the shutter closed, relMSE 1.15 with the
transform sliding and the same 1.15 with the points sliding, the two
stages being the same motion. (A first version read `shutter:open` as a
float and got nothing: the attribute is a double.)

**Velocities.** `HdsiVelocityMotionResolvingSceneIndex` is registered
ahead of the delegate's chain (phase 0, at the start, before light
linking), so a prim that authors `velocities` and `accelerations` has its
points and instance positions sampled at any shutter time from them; the
delegate's sampling above reads the same whether a stage authored samples
or velocities. Checked: a square with one sample of points and a velocity
of two units a frame, against the square with the two samples that
velocity reaches, both under the same shutter -- relMSE 0, bit for bit.
What the scene index does, measured: it extrapolates from the value the
frame reads, about the frame's time (`p(frame) + v (t - frame) / tcps`),
so the velocity stage authors its sample at the frame drawn; a sample at
frame 0 read at frame 0.5 had been held and then extrapolated about 0.5,
a frame's worth off the samples.

**Not done.** Instancers' and lights' motion is not sampled (a
PointInstancer's prototypes and a light stand at the frame's time under a
shutter); the camera's own motion neither. The raster technique draws the
frame's time, no blur. A turn between the two shutter samples is
interpolated as rows, not as a rotation. Two samples only: a shutter that
spans more than two authored samples takes the outer two. Where a prim's
transform and points both move at different sample times, the transform's
times are taken for both.

### Skinning and blend shapes, on the device

**Where the inputs come from.** In 26.08 usdSkelImaging resolves a
skinned prim through scene indices: `UsdSkelImagingPointsResolvingSceneIndex`
adds two ext computation prims under the mesh -- an aggregator holding what
does not change per frame (`restPoints`, `geomBindXform`, the joint
`influences` as (joint, weight) pairs with `numInfluencesPerComponent` and
`hasConstantInfluences`, `blendShapeOffsets` as (xyz, sub-shape) with a
`blendShapeOffsetRanges` pair a point) and the computation itself holding
the animation's (`skinningXforms` or `skinningDualQuats` with
`skinningScaleXforms`, `blendShapeWeights` a sub-shape, `skelLocalToWorld`,
`primWorldToLocal`) -- and hands the mesh its `points` as that
computation's output. Hydra never runs the computation for us: the
delegate declares the `extComputation` sprim (`HdExtComputation`, as it
comes), `HdLrtMesh::Sync` finds the computed `points` primvar, walks the
computation's scene inputs and its aggregator's outputs by name, and hands
the values whole to the engine (`SkinningArrays`), the rest points standing
in for the mesh's own.

**What runs.** `geom::Skinner` uploads those arrays as they are and
`skinning.slang` (`skinPoints`) does what usdSkelImaging's own
`skinning.glslfx` does, so a host that runs the computation itself and this
one read the same: sub-shape offsets summed into the rest point by their
weights; then linear blend skinning -- each influence's transform applied
to the point taken into bind space by `geomBindXform`, weighed -- or dual
quaternion skinning, the influences' dual quaternions blended on the
pivot's hemisphere (the heaviest influence's), normalised, any scale
applied linearly, then the point turned and moved by the blend; then
`primWorldToLocal * skelLocalToWorld`. The rest points are widened to
float4 by the builder's own decode kernel; the only host work is
transposing the matrices to the rows the kernel multiplies with. The
skinned positions go into `MeshBuilder::build` through
`MeshInput::devicePositions` -- the builder copies them instead of decoding
`points`, so normals and bounds are the skinned mesh's -- under the mesh's
topology key, so a frame of animation is a deformation in place and a
refit. The sub-shape weights, including an inbetween's share of a shape's
weight, are resolved by usdSkel on the host before they reach the
computation: a few floats a shape a frame, USD's own code.

**Checked** in `test_skinning` against closed forms a check kernel
evaluates a second time: one joint of weight 1 through non-trivial
`geomBind`, joint, `skelLocalToWorld` and `primWorldToLocal` matrices is
the matrix chain, 0 of 64 points beyond 1e-5, per-point and constant
influences alike; two joints turning about one axis by 20 and 80 degrees
at equal weight blend, under dual quaternions, to the 50 degree turn
exactly (worst 2.4e-7), where linear blending of the same pulls all 64
points off by up to 0.2; three sub-shapes over 40 of 64 points add their
offsets by their weights exactly. Through Hydra: a square bound to the
sliding joint of a two-joint skeleton is the square authored with that
slide as its transform, 0 pixels beyond 2 at rest and slid, by raster and
by rays, its generation standing across the frames; a blend shape with an
inbetween authored at 0.5 draws, at weight 1, as the square authored with
the shape's offsets and, at 0.5, as the square authored with the
inbetween's own offsets -- not half the shape's -- 0 pixels beyond 2 both.

**A bug the check caught in itself.** The first check kernel read a
point's blend shape range past the end of the ranges buffer for the points
without one; alone the stale memory read as zeros, after two other cases
it did not, and one thread looped for billions of steps -- the device
hung, `submit` never returned, and the case only "failed" by the SIGTERM
that ended it. The kernel now takes how many points have a range, as the
skinning kernel always did. A check has to guard what the kernel guards.

**Not done.** The deferred-skinning route (`HD_ENABLE_DEFERRED_SKINNING`,
`hydra:skinningXforms` and the rest as primvars named by
`HdSkinningSettings::GetSkinningInputNames`) is not read: the variable has
to be set before Hydra loads, which only a process's `main` can do through
`platform::setEnvOnce`, and it would be a second reader of the same
Skinner; the ext computation route is the one every host gets. Skinned
normals are recomputed from the skinned points (smooth) rather than
skinned from authored normals (`skinningNormalsComputation` is not read).
A skinned mesh under a shutter blurs by its transform only: the skinning
transforms are read at the frame, not at the shutter's samples.

### The timeline

`lrt view` had a time slider; it now plays. Play advances the time by the
wall clock at the stage's `timeCodesPerSecond` and wraps at the end, the
step buttons move a frame, and dragging the slider stops the play. What
frame N shows is `SetTime`'s business and when it is drawn the clock's --
the two are not mixed, which is also how `lrt live` already worked: its
`sched` clock decides when frame N is drawn, and `--start` what frame N
is. Nothing here is measured beyond the frame times the panel shows.

## Complete USD: the breadth of geometry (M8)

### hdsi's conversions ahead of the delegate

The delegate draws meshes, points and splats; what USD authors beyond
those reaches it as meshes through hdsi's scene indices, registered for
this renderer in phases ahead of light linking: `HdsiImplicitSurfaceSceneIndex`
with every implicit type (sphere, cube, cone, cylinder, capsule, plane)
set to `toMesh`, `HdsiTetMeshConversionSceneIndex` (a TetMesh's surface
faces), `HdsiNurbsApproximatingSceneIndex` (a NurbsPatch as a mesh),
`HdsiPinnedCurveExpandingSceneIndex` (for the curves M8 adds below), and
`HdsiCoordSysPrimSceneIndex`, which turns a coordinate system bound to any
xformable into a `coordSys` prim under it with that prim's transform. The
delegate's own code did not change; the registration and its arguments did.

**Checked** through Hydra: a `UsdGeomSphere` of radius 1.2 against the
analytic sphere in `sphereCheck` (beside `planeCheck`) -- hdsi tessellates
it with ten segments, so a chord sits inside the sphere by up to
`r (1 - cos(pi/10))` = 0.0587, coverage is judged outside that band about
the silhouette and depth where the ray meets the sphere squarely (within
0.8 r of the axis, where a facet's error along the ray is at most the sag
over 0.6): 0 of 26788 pixels wrong, depth within 0.075 of the sphere over
9772 pixels. A one-tetrahedron `TetMesh` against its four faces authored as
a mesh, by depth (hdsi winds the surface its own way and the headlight
shades the side it sees): relMSE 0. A degree-one `NurbsPatch` against its
quad: 0 pixels beyond 2. The first sphere check was wrong itself -- it took
`sqrt(dist^2 - r^2)` for `dist - r` and judged half the disc wrong -- and
Python recounting the same formula on the dumped depth reproduced the count
exactly, which is what told the kernel from the expectation.

### Invisible faces

A face Hydra marks invisible (`HdMeshTopology::GetInvisibleFaces`) stays
in the topology and is not drawn: unlike a hole, whose triangles the
triangulation drops, an invisible face keeps its triangles and their
numbering, so showing it again is a flag and not a rebuild. The builder
marks the faces the way it marks holes (the same kernel over another
list), `subsetTriangles` writes the flag into the top bit of each
triangle's subset word (a subset index never reaches it), the scene pools
it as it pools the subsets, and the one place every route already
evaluates a sample before keeping it -- the cutout passes' `materialCuts`
-- answers yes for a flagged triangle before it looks at the material. The
engine takes the cutout passes whenever a mesh has a hidden face, cutout
materials or not.

**Why a bit and not a buffer.** The first version gave the flag a buffer
of its own, bound wherever materials are looked up, and every path traced
test failed at once: Metal allows a kernel 31 buffers, the path tracer
was at the edge, and the one more put a binding out of range -- the
kernel did not compile, and the suite said so 18 times. The lesson stands
in the docs because it will bite again: the material frame binds a dozen
buffers and the path tracer adds its own, so a new per-triangle or
per-mesh datum rides in a word that exists.

**Checked** in `test_lights` on the bumpy grid with its odd faces
invisible, by the three routes: every pixel where the full grid showed an
even face shows exactly that (instance, triangle) -- 0 of 9945 differ, by
raster, rays and the compute walker alike, so hiding renumbered nothing
and hid nothing it should not -- and none of the 9945 lies on an odd face;
19903 pixels were drawn with every face. A first version of the check
compared every drawn pixel with the full grid's and found 131 differing:
pixels where an odd face had stood in front of an even one, which the
hidden grid rightly shows. What USD does not have is a way to author
invisible faces on a mesh (`UsdGeomSubset` carries no visibility;
`invisibleIds` is for points and curves), so the Hydra side is read and
not exercised by a stage.

### Basis curves, as tubes

`UsdGeomBasisCurves` arrive through `HdLrtBasisCurves` (an `HdRprim`, as
points are) as their topology, points and widths, and `geom::CurveBuilder`
lays a tube over every span on the device: `curve_tube.slang` evaluates
the span at `segments + 1` parameters -- the Bezier, uniform B-spline,
Catmull-Rom or linear form of its control points, with the derivative for
the tangent -- and rings `sides` vertices at half the width about each
point in a frame taken from the tangent; the width is constant, a curve's,
a control point's (blended like the point) or a span end's. The rings'
quads are host bookkeeping (a pattern of indices), the positions the
kernel's, and the whole goes through `MeshBuilder::build` as device
positions with smooth normals, so a curve is a `GpuMesh`: every visibility
route draws it, shading, AOVs, cutouts, linking, picking and the path
tracer take it as a mesh, and a curve whose points move is a deformation
and a refit (the same topology key). Periodic curves wrap their spans;
pinned ones are expanded by hdsi ahead of the delegate. A uniform primvar
becomes one value a face, each face taking its curve's.

This is not the plan's design -- a curve primitive of its own in the
visibility buffer, ribbons in raster and swept cones in the BVH -- and the
reason is cost against what it buys: the plan's purpose was that nothing
downstream should change, and a tube as a mesh changes nothing downstream
at all, at the price of a fixed tessellation (a hair far away aliases as
any thin mesh does, and a head of hair is many triangles). A curve
intersector is where the design should go if hair at scale is wanted; the
tube is exact in what it draws.

**Checked** in `test_curves` against the curve: for every basis, periodic
and not, every tube vertex sits at half the width from the point of the
curve it rings, that point evaluated a second time in the check kernel in
the Bernstein, polynomial and Hermite forms (not the builder's), 0 of up
to 504 vertices beyond 1e-5, worst 2.8e-7; the span counts and the
triangle counts are what the rule says. Through Hydra, a straight linear
curve of width 0.4 is a cylinder: `cylinderCheck` finds 0 pixels wrong
beyond the eight-sided tube's facet band and the depth within the sag
over 0.7 of the cylinder's front (0.0169 against 0.0152), by raster, rays
and the compute walker alike.

### The hair lobe (Chiang et al. 2016)

`chiang_hair_bsdf` is one more kind in the lobe library, `kLobeHair`, so
the stack, its one-sample MIS and every consumer are unchanged. The lobe
carries the R, TT and TRT tints in the three colours, the R roughness in
`alpha`, the cuticle angle in `roughness`, the fibre's ior, and the
absorption coefficient and the TT and TRT roughnesses in the struct's pads
and in Schlick's `exponent` (`setHair`, `hairAbsorption` and the two
roughness readers keep that in one place). The Lobe stays 128 bytes on
purpose: the first version grew it by two float4, and the shadowed
shading kernel then drew a point light's umbra wrong on Metal -- 2583 of
7440 pixels off a closed form that had been exact, in a kernel that never
touches the hair -- and packing the fields into the pads made it exact
again. The cause is not explained; the size is what was measured to
matter, so it is held. The MaterialX node's Slang implementation
(`lrt_chiang_hair_bsdf.slang`) fills it and pushes it with the node's
`curve_direction` as the tangent. The model is pbrt-v3's: for lobes p = R,
TT, TRT and a geometric tail, a longitudinal density M_p over theta_i
(d'Eon's, normalised against cos theta d theta), an azimuthal one N_p
over phi (a trimmed logistic about the lobe's centre), and an attenuation
A_p from the Fresnel terms and the absorption through the fibre; f cos
theta_i is their sum, the pdf the same sum with each lobe's share of the
attenuation in place of A_p, and a sample chooses a lobe by that share,
draws theta_i from M_p and phi from N_p. The offset across the fibre comes
from the hit's normal against the outgoing direction, as MaterialX's own
eval takes it: on a tube the normal is the radial one, so the offset
follows from it, and no separate curve intersector is needed. The fibre's
direction is the tube's `tangent` primvar (the curve builder writes one a
vertex, the builder takes it as a device primvar, `material_surface`
prefers it to the texture-derived tangent).

Two things differ from MaterialX's `mx_chiang_hair_bsdf.glsl` on purpose.
The cuticle tilts the *outgoing* angle per lobe (pbrt's way), not the
incoming one (MaterialX's): shifting theta_i changes the measure the
density is normalised against, and a sampler drawing from the shifted
density no longer matches the pdf -- the chi-square said z 126 before the
change and 1.1 after. And the tail lobe's azimuthal density is 1 / (2 pi):
MaterialX writes `1.0 / 2.0 * M_PI`, which is pi / 2.

**Checked** in `test_lobes` as every lobe is: a million samples binned
over the sphere against the pdf integrated over the bins, and the albedo
from the samples' weights against the albedo from eval over uniform
directions. With no absorption and white tints the attenuations sum to
one, so the lobe returns everything: albedo 1.0000 sampled, 0.9944
uniform, z -0.8; absorbing, off-axis at 60 degrees, with and without the
cuticle's tilt: z -1.0 and 1.1, the two albedos 0.307 against 0.306. The
pdf's integral over the bins reaches 0.996 under the R lobe's variance of
0.1 -- the bins' quadrature, allowed 0.01 for hair where the others get
0.002. The model is not reciprocal (the attenuation is the outgoing
side's), as its authors' is not, so no reciprocity is asserted. Two of the
lobe's own bugs the checks caught: a fourth random number taken from the
second's low bits tied theta_i to phi_i (z 4.8), fixed by rescaling what
the lobe choice left over; and dividing the pdf by cos theta_i as well as
f, which put its integral at 1.32.

Through Hydra, a B-spline curve under a chiang material draws lit
(relMSE 4.9 against blank) and unlike the same curve under Lambert
(relMSE 0.13).

### Subdivision surfaces on the device

`geom::Subdivider` refines a mesh `levels` times under Catmull-Clark, Loop
(all-triangle meshes; anything else falls back to Catmull-Clark) or
bilinear rules, and pushes the last level onto the limit surface. The
split of work follows the rule: the host lays out each level's topology
as tables of indices -- which corners make which edges (an edge map over
(min, max) pairs), what each vertex touches (a CSR of its edges and faces),
which edges and vertices are sharp, and the children's corner lists in the
[vertex points][edge points][face points] numbering -- and
`subdivision.slang` places every point: a face's centroid, an edge's point
(the four-point Catmull-Clark rule, Loop's 3/8-1/8, or the midpoint on a
boundary or a sharp edge, blended by a semi-sharp edge's sharpness), and a
vertex's new place (Catmull-Clark's (Q + 2R + (n-3)P)/n, Loop's beta rule,
the crease rule (E1 + 6P + E2)/8 through two sharp edges, blended by
sharpness; a corner, a vertex with three or more sharp edges, or a boundary
vertex that is only its two boundary edges stays -- edgeAndCorner for
points, cornersOnly for a face-varying channel). Creases and corners come
from `UsdGeomMesh` as authored (a sharpness a crease or an edge), a
boundary edge is a crease, and a child edge keeps its parent's sharpness
less one. Face-varying channels run the same kernels over their own
topology: the channel's unique values (its indices) are its vertices, an
edge one face makes is a seam and so a boundary; vertex channels follow
the points, varying ones the bilinear rule, uniform ones map each refined
face to its coarse face, and authored normals are dropped for the refined
surface's own. The limit projection is `subdivLimit`: Halstead, Kass and
DeRose's stencil for Catmull-Clark -- (n^2 P + 4 sum of edge neighbours +
sum of the quads' diagonal vertices) / (n (n + 5)) -- and (1 - n gamma) P
+ gamma sum of neighbours for Loop, (E1 + 4P + E2)/6 along a crease or
boundary, corners fixed. `ref/falcor`'s LoopSubdivide was read for the
Loop rules and never built.

**Checked** in `test_subdivision`, everything exact:

- **Euler's counts**, integer and exact at every level: a cube's 26/24/48
  points, faces and edges after one level, 98/96/192 after two, 386/384/768
  after three; an octahedron under Loop 18/32/48, 66/128/192, 258/512/768.
- **The limit converges and does not drift.** The cube corner's limit from
  the coarse cube alone, by the closed form written a second time in the
  check kernel, is (-0.5, -0.5, -0.5); the kernel's projection of that
  corner's descendant reaches it to 0.000000 at levels 1, 2 and 3 alike.
  The first stencil written took edge midpoints and face centroids for the
  two sums and put the corner at 0.75; the projections then moved with the
  level (0.38, 0.42, 0.43 off), which is what told it from the right one --
  a limit stencil applied at any level must land on one point. Loop's
  octahedron vertex reaches its closed form (0.5, 0, 0) to 0.000000 at
  every level.
- **A crease holds its plane.** A sharpness-10 crease around the cube's
  top keeps every descendant of the top on y = 1 (9 points at level 1, 25
  at level 2, none above) where the smooth cube's highest point is 0.8395.
- **A face-varying square stays a grid**: each of the cube's faces with
  its own unit square of st, refined twice, gives 96 faces whose four st
  corners are all axis-aligned rectangles (a first version of the test
  gave the six faces the same four values, and the six squares became one
  face-varying vertex a corner with no seam).

**Through Hydra.** `HdLrtMesh` hands the engine the scheme, the display
style's refine level and `UsdGeomMesh`'s creases and corners; the engine
refines a mesh whose scheme is not `none` at a level above zero (after
the skinning, when there is any; five levels at most) and builds the
refined mesh under the coarse mesh's topology key, so an animated
subdivision surface is still a refit. The level is the display style's,
as usdview's complexity sets it: `StageRenderer::setRefineLevel` and
`lrt stage --refine` set it through `HdsiLegacyDisplayStyleOverrideSceneIndex`
inserted ahead of the renderer's chain; a host's `HdDisplayStyle` reaches
the delegate the ordinary way. Checked with a catmullClark cube at refine
0, 1 and 2: the centre pixel's depth is the flat face's 5.0000 at 0, and
5.168 and 5.164 at 1 and 2 -- the limit surface inside the cube, one
surface at both levels to a facet's sag -- while the coverage falls from
8464 pixels to 2956 and 3196 as the corners pull in.

**Not done.** Boundary interpolation is `edgeAndCorner` and face-varying
interpolation `cornersOnly` whatever the mesh authors (`edgeOnly` and the
other face-varying rules are not read); holes are refined and dropped
after; invisible faces do not survive refinement; a mesh's refined
normals are the refined surface's smooth ones, not limit normals; Storm
was not used as an image oracle here -- the closed forms above are the
checks, and Storm's own OpenSubdiv would have been checked against them
the same way.

### The light BVH

With `lrt:chooseLights` a sample takes one light; it took it by power
alone, which is exact and blind to where the point is. Now the frame's
bounded lights (sphere, disk, rect, cylinder) sit in a BVH
(`light_bvh.slang`), each node a box, an orientation cone (Conty and
Kulla 2018's union of its children's) and the power under it, and a
shading point descends it choosing each child by its importance there --
power over distance squared, within what the cone allows, times the
surface's own cosine with the box's angular uncertainty allowed for --
so a light that cannot reach the point is rarely chosen and one beside it
often. Dome and distant lights are unbounded and never enter the tree:
they sit in a list after it in the same buffer, chosen against the tree
by their share of the power, so a dome never vanishes in silence. The
probability of the choice is the product of the branch fractions, and
`lightPdfChoiceAny` recomputes it for any light by walking its leaf's
parents; where neither child of a node can light the point by its bound
(the parent's looser bound let the descent in) the choice falls back to
power, so no probability is lost.

The build (`light_bvh_build.slang`, `LightTable::buildBvh`) reuses the
compute LBVH's own kernels over the lights' boxes -- Morton codes within
bounds a kernel found, `bvh_hierarchy`, `bvh_refit` settling the boxes --
then packs the hierarchy into nodes, links the parents and settles power
and cones up the tree until nothing changes. The nodes are sixteen floats
each **inside the IES values buffer**, after the profiles, and the shading
kernels read them through `lightNodeBase`: a buffer of their own put the
shading kernel at `buffer(38)` against Metal's thirty-one. And the build's
kernels live in a module of their own because a Slang module's global
parameters join every kernel that imports it -- with the build's buffers
in `light_bvh.slang`, importing the tree to choose from it cost the
shading kernels ten bindings they never used, and the same error came
back with no new buffer in sight.

**Checked** in `test_lights` with forty lights of the four bounded kinds
scattered and turned, a dome and a sun: 0 of 39 internal nodes whose box
or cone misses a child's; at a hundred random points and normals the
choice's probabilities over the 42 lights sum to one (worst 3.6e-7 off);
a million draws at one point all report the probability
`lightPdfChoiceAny` recomputes for the light drawn (0 mismatches), and
their histogram follows those probabilities -- chi-square z 0.62 on 29
degrees of freedom, which is what a per-sample check cannot see and M5's
lesson asked for; the dome's share is 0.0168, the sun's 0.0479. Then the
many-lights closed form, chosen through the tree, holds as it did by
power. A first cone union swapped the cones the wrong way round (13 of
39 cones failed to hold their children) and a first descent returned
nothing where both children's bounds gave zero, losing 2.2% of the
probability (66 of 100 points summed short); both were found by these
checks, not by an image.

**Not done.** Light instancing's copies each take a leaf (a thousand
copies are a thousand leaves); a light's cone ignores an IES profile's
shape; the tree is rebuilt whenever the table is set, which is every
frame the lights change and never when they do not. The table's tree
uniforms are set only where a kernel declares them: the first `bind`
that set them everywhere put a null cursor under two kernels that sample
by power alone (the IES check, the dome's plane) and both crashed the
host, which the suite found and a single test would not have.

### Coordinate systems, resolved by the scene index's prim

A material may name a coordinate system (`UsdShadeCoordSysAPI`: a name on
the prim, bound to an xformable). `HdsiCoordSysPrimSceneIndex` makes a
`coordSys` prim under the target with the target's transform, and the
emulation gives the prim a name of the binding's (`__coordSys_paint`); the
delegate accepts the `coordSys` sprim (`HdCoordSys`, whose Sync is hd's),
and a mesh's Sync reads `GetCoordSysBindings` and each binding's transform
into its `MeshLook` (`CoordSysBinding`: name, to-world), which the engine
keeps beside the mesh and `StageRenderer::coordSysBindings` reads back.
The name is what remains of the prim's past the scene index's prefix --
the prefix is a known constant, not a parsed path -- and the transform is
the prim's, so a target that moves moves its system with no code of ours
computing it.

**Checked** through Hydra: a square with `CoordSysAPI:paint` bound to an
`Xform` translated to (1, 2, 3) reads back one system named `paint` at
exactly that translation.

**Materials read them**, through MaterialX's `transformpoint`,
`transformvector` and `transformnormal`: a space named `world`, `object`
(or `model`) or anything else, which is a system bound to the prim. The
engine hands each binding's transform to world to the mesh as three
constant float4 primvars, `lrtCoordSys_NAME_0` to `_2` (the rows of a 3x4,
as Hydra gave them; a name with colons is not a MaterialX identifier, which
the first spelling found), and the transform node reads them through the
primvar slots every material already uses -- a per-mesh value with no new
binding in any kernel. `MaterialInputs` carries the object-to-world rows,
composed on the device from the instance's object-to-view rows and view to
world; inverses and inverse transposes are the shader's
(`lrtTransformBetween`). A system the prim does not have reads as world.

**A defect this found**: genslang's own transform nodes multiplied by
`u_worldMatrix` and its inverse, uniforms the compiler read from the blob
with nothing written there -- identity. Every object/world transform in a
MaterialX graph was a no-op on any mesh away from the origin. The four world
matrices are now built from the shading point's rows, as the float4x4
`mx_matrix_mul` applies.

**Checked** against graphs that reach the same value without a transform
node, frames compared (`compareHdr`), on a square tilted in object space
under a rotation, a non-uniform scale and a translation, with a frame
translated and scaled by 2 bound as `paint`: object point to world against
world position, 1.66e-5 largest relative difference; world to object
against object position, the same; world to `paint` against (P - t) / 2,
the same; the object normal to world against the world normal, bit for bit
-- with a control that the untransformed object normal differs from it
(0.30). That reference graph also showed **hdMtlx typing a USD `vector3f`
value as a string** (a role type MaterialX has no name for), so the value
never reached a `vector3` input and read zero; a string input the nodedef
declares numeric now takes the declared type, and its text parses as that.

**Not done.** A binding whose target moves while the mesh's arrays do not
is read again only when the mesh is rebuilt: the rows travel with the mesh's
primvars. Curves do not take the systems' primvars.

## Complete USD: render settings and outputs (M10)

### Render settings prims, products and vars

A `UsdRenderSettings` prim reaches the delegate as Hydra's `renderSettings`
bprim (`HdLrtRenderSettings`, hd's `HdRenderSettings` with a sync counter),
through `HdsiRenderSettingsFilteringSceneIndex` registered with the `lrt`
namespace prefix so `lrt:pathSamples`, `lrt:technique` and the rest arrive
as its namespaced settings, and `HdsiSceneGlobalsSceneIndex`, which is
where the active settings prim is named. `StageRenderer::renderSettings`
makes a prim active, syncs the prims (no tasks, so no frame) and reads the
bprim back: products, vars, purposes, colour space, camera, its settings.
`renderProducts` renders each product at its own resolution from its own
camera with `includedPurposes` as the render tags (default: geometry,
render: render, proxy, guide as themselves) and the `lrt:` settings applied,
and writes its vars as the layers of one OpenEXR (`io::writeExrChannels`:
named channels, half, float or uint, sorted as the format wants; `readExrChannels`
reads any file's channels back). A var names its AOV by `sourceName` (hd's
names; RenderMan's `Ci` and `z` read as colour and depth; `primvar` sources
as `primvars:NAME`; of light path expressions, `C.*<L.'NAME'>` as the light
group NAME). Layers are `NAME.R/G/B/A`, `NAME.x/y/z`, `Z` for a var named
depth, a bare `NAME` for an id plane (uint, so -1 stays 0xFFFFFFFF); 32-bit
floats unless `lrt:exrHalf` is set. `lrt stage --render-settings /Render/X`
renders the products and stops.

**Two things the plumbing needed that the reference host does not say.** A
settings prim's `active` is a dependency the filtering scene index
declares on the scene globals, and dependencies become dirty notices only
through `HdDependencyForwardingSceneIndex`, which the renderer's chain now
ends in; without it the second prim made active synced never and read
inactive. And marking the bprim dirty through the change tracker is a
no-op for a prim the emulation delivered: it dirties the legacy prim scene
index, which does not hold it.

**Checked** through Hydra: a settings prim with two squares (one a guide),
a product of four vars (beauty from `Ci`, depth from `z`, primId, Neye) at
96x64 -- the prim reads back active, synced once, with its purposes and
`lrt:` settings; the product's nine channels, read back from the file and
uploaded, are bit for bit the AOVs rendered one at a time (`countDifferent`
0 words for every layer); a second settings prim that includes guides
covers 3812 pixels against 2916, and reads active once asked, the first
one again after. The rays' render tags are part of the path tracer's frame
key, as the plan asked: a purpose that changes starts the mean again.

### Light groups

A light's `lrt:lightGroup` (or RenderMan's `ri:light:lightGroup`) names its
group; a frame asks for a group as the `lightGroup:NAME` output, and the
engine numbers the groups asked for (1 + index; 0 for a light in none, or
in one nobody asked for) into `LightRecord.group`. Both shading kernels
accumulate each light's direct contribution under its group -- the raster's
per light, the path tracer's at every bounce through the path's throughput
-- and a frame without groups compiles exactly the kernel it always did:
the groups' code is a variant of the generated source, not a branch.
Emission and the background are in no group. At most eight.

**Where the planes live** is a lesson twice over. The raster's go to a
buffer of their own. The path tracer's go into its accumulation buffer after
the colour's plane (sums, then means), because the traced kernel stands at
Metal's limit of 31 buffers: two more put it at `buffer(32)`, and the Metal
compiler in the process did not fail -- it never returned, ten minutes and
counting, while the same source through `xcrun metal` fails in a tenth of
a second with the error. `LRT_SHADER_DUMP=<dir>` now writes every generated
module, which is how that was found. And accumulating the groups in a local
array indexed by the light's group, or writing the buffer inside the bounce
loop, were both tried first; eight registers and a select each is what the
kernel has.

**Checked** through Hydra with a floor under two lights in `key`, one in
`fill` and a var for an empty `rim`: at every pixel the groups summed are
the beauty (worst 1.8e-7 relative raster, 2.4e-7 path traced, over 4292
covered pixels) and the empty group is zero exactly; the same through a
render product whose vars are the light path expressions.

**A Metal problem this found.** The raster technique with any shadow ray
through Hydra wrote rows of garbage in blocks of half a threadgroup,
nondeterministic, on an Apple M5 Pro: clean under Metal shader validation,
clean without the trace, clean with the structure bound but unused -- and
clean once the kernel stopped copying the material's lobe stack into a
local and read it in thread memory where the material left it. The path
tracer, which keeps its stack in a struct it passes on, never showed it.
Live state across the intersector call is the reading that fits; it is
measured, not understood. Every Hydra light test had shadows off, which is
why nothing had caught it; the regression that does compares shadows on
against off over an unoccluded floor, three times, 0 words apart.

### ACES 2.0, on the device, tables and all

`aces2.slang` is the Academy's ACES 2.0 output transform ported function
for function from the reference CTL (aces-core `Lib.Academy.OutputTransform`
and `Lib.Academy.Tonescale`): Hellwig 2022's appearance model to J, M, h;
the tonescale on J; the in-gamut compression of M; the compression towards
the display cube's boundary; and its inverse. What the reference computes
once at init -- the reach of AP1 at every hue, the display gamut's cusp at
every hue with the hue table it samples, the upper hull's gamma -- are
tables of 362 entries built by three kernels (`aces2_prepare.slang`: the
parameters and the hue table on one thread, since the hue table follows
the sorted corner hues in order; then a thread a hue; then the wrap
entries) into one float buffer, `technique::Aces2Tables`, rebuilt when the
peak luminance or the limiting primaries change. The renderer's linear
Rec.709 goes to ACES2065-1 by a Bradford adaptation computed on the device
from the chromaticities. The display transform's third view, limited to
Rec.709 or to P3 as the display is; `lrt view` lists it.

**Checked** in `aces2_check.slang`, at 100 nits limited to Rec.709 and at
1000 nits limited to P3: 256 scene greys over sixteen stops come out
neutral (spread 5e-6), on the tonescale written a second time in the check
from the published constants in its own form (7e-6 relative), rising every
step; 18% grey shows 0.10000 of reference white at 100 nits (the
reference's 10.013 nits less its flare term); inverse then forward over
8192 display colours returns them to 6e-6 of the peak. The gamut
compression approximates the cube's boundary (a smooth cusp, a hull gamma
fitted at five points), so 4.7% of scene colours of any saturation and
exposure land a little outside it at 100 nits (worst 0.11 over), 0.45% at
1000 nits: the reference clamps at the display encoding, and so does the
display kernel; the check bounds the count and the worst.

### Extended range, and what OCIO is not

`DisplayEncoding::LinearP3` leaves linear P3 with 1.0 at the reference
white and the headroom above it, clamped at ACES's peak; `lrt view --edr`
asks the window's Metal layer for extended range content in
`kCGColorSpaceExtendedLinearDisplayP3` (`platform::enableExtendedRange`,
through the Objective-C runtime and CoreGraphics, in Platform where the
rule puts OS calls), takes an `RGBA16Float` surface, and shows ACES 2.0 with
the screen's headroom times 100 nits as its peak
(`platform::extendedRangeHeadroom`, the screen's
`maximumExtendedDynamicRangeColorComponentValue`). Measured here on a
screen reporting a headroom of 1.00: the pipeline runs (draw 12.6 ms
medians at 640x400, snapshot written), which shows the plumbing and not the
range; a screen with headroom is what would.

**OCIO is not built.** The USD prefix carries no OpenColorIO, so the
optional route the plan described -- OCIO as a compiler whose shader text
and LUTs become a generated Slang module and textures, the LUT values the
one thing the host would compute -- is not in the tree; ACES 2.0 analytic
is the default and the only colour management, and `renderingColorSpace`
is read from the settings prim and reported, not acted on.

**Light groups and what is not a light.** A dome the camera sees is in its
dome's group: `C.*<L.'NAME'>` matches the camera's ray meeting the light
with `.*` empty. The engine paints it into the group planes as it paints
the colour's background, and the camera's exposure scales the planes as it
scales the colour. For that the path tracer's group means are copied out of
its accumulation each frame into the engine's plane buffer (the raster
writes there itself): scaled in place, a converged pixel that no pass
rewrites would take the exposure twice. Checked with a sky dome in a group
of its own under one stop of exposure: the groups sum to the beauty at all
6144 pixels, sky included, raster and path traced; before, every pixel was
off (relative 1.0, the factor of two). Emission stays in no group, as the
expression says (an emitting surface is `O`, not `L`): a frame with
emissive geometry sums its groups short of the beauty by exactly that.

**Not done.**
`materialBindingPurposes` is applied since the usd-wg end to end (below):
this line used to say the delegate bound `full`, and it bound Hydra's
default, `preview`. Products' `disableMotionBlur` and
`disableDepthOfField` (theirs or their settings prim's) are applied per
product, through the delegate's `lrt:disableMotionBlur` (one shutter slice)
and `lrt:disableDepthOfField` (the camera's fStop ignored), reset when the
products are written: a sliding square under an open shutter and a lens
focused before it, with both switched off, is bit for bit the stage with
neither authored (0 words of 18432); the lens alone drawn, 4464 off; the
blur alone, 7506; a render after the products, the effects again. A render
var of any other
light path expression is refused with a message.

## Complete USD: volumes (M9)

### From .vdb to the device, with no statistic on the host

`io::readVdbGrid` reads a float grid with OpenVDB, voxelises its active
tiles (so the tree the kernels walk is leaves alone) and lays it out with
NanoVDB's converter with statistics and checksum off: what crosses to the
device is the layout and the header's leaf count. The rule's exception is
bounded to a re-layout, as decompressing SPZ is, and nothing about the
values is read on the host. `world::VolumeSet` puts every volume of the
frame in **one buffer of words** -- a header, a 32-word record a volume
(world to index, extinction scale, albedo, phase g, and what the finish
kernel writes: index bounds, leaf 0's word, majorant), the grids, then each
leaf's largest value -- because the traced kernel has one Metal buffer
slot left. `volume_prepare.slang`'s kernels take the bounds from the
leaves' origins, each leaf's maximum over its active voxels and the
grid's majorant.

PNanoVDB compiles as Slang: `lrt/volume/nanovdb.slang` includes
`PNanoVDB.h` as HLSL (its buffer is a `StructuredBuffer<uint>`) and exports
grids and leaves by offset; `slangc` takes it to Metal, CUDA and SPIR-V. The
header travels with the shaders from the USD prefix. NanoVDB 32's
`GridBuilder` still names `std::result_of`, removed in C++20, which libc++
keeps behind `_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS`.

**Checked** on a fixture of two boxes (32^3 voxels at 0.5, 8^3 apart at
2.0): 33280 active voxels counted through the leaf masks, bounds
[0,48)x[0,32)x[0,32), all 65 leaf maxima matching a voxel-by-voxel recount
through the accessor, majorant 2.0.

### The medium: delta and ratio tracking leaf by leaf

`medium.slang` walks a volume's leaves with PNanoVDB's HDDA at the leaf's
size and, inside each, samples free flights (delta tracking) and
transmittance (ratio tracking) under **that leaf's** majorant, so an empty
leaf costs one step and a dense one is sampled at its own rate. The ray is
taken to index space with its parameter kept in world units.
Henyey-Greenstein for the phase, drawn with its own density.

**Checked** through the dense box along six axis directions and one
oblique ray: ratio tracking's transmittance and delta tracking's scatter
fraction both on Beer-Lambert within the standard error the kernel
measures (|z| <= 1.2 over 16384 rays each); no density over the leaf
majorant at 16384 random points in and about the box; the leaf walk
crossing exactly the cells a walk of a tenth of a voxel crosses. That last
one needed the walk to skip cells a rounding sliver long at a boundary.

### In the path tracer

The traced kernel's sample loop now walks vertices that are either a
surface the ray met or a collision in a medium before it. A collision
gathers light through the phase (lights chosen by power) and sends the path
on by the phase; transmittance to a light is ratio-tracked -- **only for a
light that casts shadows**, since a volume is an occluder like any other and
`shadow:enable` off means none dims it. A pixel whose ray finds no surface
still walks the medium along its camera ray. An imageless dome is drawn
uniformly over the sphere at a point in a medium: its cosine sampling about
a normal never draws the half behind, and flipping between halves weighs a
sample near their horizon by one over its cosine. A frame without volumes
compiles exactly the kernel it did (`kVolumes` stubs).

**Checked**, pixel by pixel, with an absorbing box (albedo 0) over the right
half of a sun-lit plane: every pixel sees one radiance per sample, so the
frame with the box over the frame without is a binomial mean whose
expectation is Beer-Lambert along the pixel's own ray and whose deviation
the check derives from the path count. At 1024 paths, 0 of 9801 pixels
beyond five deviations and mean z^2 0.986 with the sun's shadows off
(camera chord only); 0 and 0.992 with them on (camera and light chords);
the 9680 pixels outside the box unchanged. **The volume furnace**: an
albedo-one medium of optical depth at most 0.56 under a dome of radiance 1,
32 bounces -- every walk ends where an escape would have seen the dome, so a
pixel reads 1 whatever the phase: 0.99983 isotropic, 1.00228 at g 0.7, within
0.2 and 0.6 standard errors measured from the pixels' own spread.

The first pass at the Beer-Lambert check read the square of the expected
transmittance: the renderer was right (the sun's light crosses the box
too) and the expectation was missing a chord. And one bug was the host's:
`VolumeSet` read OpenVDB's row-vector map untransposed and composed it on
the wrong side, so a translated volume sat offset by its translation in
voxels. A pure scale -- the first fixture -- cannot show that.

### Through Hydra

`HdLrtVolume` (the `volume` rprim) takes the field named `density`, else the
first, its transform, and constant primvars `lrt:densityScale` (default 1),
`lrt:albedo` (0.8) and `lrt:anisotropy` (g, 0); `HdLrtVolumeField` (the
`openvdbAsset` bprim) takes the file and grid name. The engine reads each
grid once per file and name, lays the frame's volumes out again whenever a
volume or field changes, and a change raises the path tracer's revision.
**Checked** with the same box authored as a `Volume` and an `OpenVDBAsset`
over a mesh plane under a `DistantLight`: 0 of 9801 pixels beyond five
deviations, mean z^2 1.014, mean ratio 0.1920 against 0.1921, the outside
unchanged.

**A frame of volumes alone** is drawn too. Path tracing runs in the mesh
layer, which the engine took only when a mesh was in the frame, and whose
scene and material programs it made only when a mesh had arrived; a traced
frame with a visible volume now takes the layer with an empty scene, the
visibility finds nothing, and every sample walks its camera ray. Checked as
the furnace through Hydra with no mesh at all -- a `DomeLight` of radiance 1
and an albedo-one `Volume`: 4278 pixels read 1.00044, 0.44 standard errors
from the dome.

### Not done

- The raster technique draws no volume, and the engine says so once.
- Only float grids, one field a volume (density); no temperature, emission
  or colour fields, no `UsdVol` material networks -- the medium's
  parameters are the three primvars.
- Light groups do not collect light scattered in a medium differently from
  a surface's: a collision's light goes to its light's group, as a
  surface's does.
- In a medium lights are chosen by power, not through the light BVH.
- No MIS in media; no spectral tracking for a coloured extinction.
- Verified on Metal; the L4 run of these tests is M11's.

## End to end: Kitchen_set lit, and two defects only a real stage showed

The plan closes with real stages through `lrt stage`. Kitchen_set with the
lights `Kitchen_set_lit.usda` adds (a dome, a window rect, two normalised
spheres), from inside the kitchen, rendered black under both techniques --
while the same view without lights, under the headlight, was right to the
last texture. Two defects, each hidden from the suite by the shape of its
fixtures.

### The geometric normal was reversed in view space

`surface.slang` formed a triangle's geometric normal as the cross product
of its view-space corners. The view is a reflection -- it flips z to look
down +z -- and a cross product under a transform of negative determinant
comes out reversed. So every front face read as a back one, and the
backface test flipped the shading normal. With a computed normal (the
geometric one) the two reversals cancelled; with an **authored** normal,
carried by the normal matrix which has no such sign, the shading normal
ended facing away. Lobes are evaluated with an absolute cosine, so a
sphere, a rect, a sun and the headlight all lit such a surface correctly;
a dome samples its directions about that normal, and lit nothing. Every
lighting fixture computed its normals. The geometric normal is now signed
by the object-to-view determinant, which covers mirrored instances too.

**Checked**: a Lambert plane of albedo 0.18 under an imageless dome of
radiance 1 reads 0.18 at every pixel (worst 1.7e-5 relative) with computed
and with authored normals, raster and rt; without the fix the authored
case is 1.09 off, the computed one passes -- as the cause says. The whole
suite passes with the sign; nothing depended on it.

The dielectric's `inside`, which read the same flipped test, is checked
through a camera on a closed cube (`inside_check.slang`, "a closed mesh is
inside only where it is seen from within"): every covered pixel's
`MaterialInputs` counted, with the camera outside the cube and within it,
the cube plain, mirrored, and authored left-handed. Outside: 7078 pixels, 0
inside; within: all 7081 inside; the shading normal handed to the material
faces the eye at every pixel of the six. Without the sign the plain and
left-handed cubes invert exactly (7078 inside from outside, 0 from within)
and the mirrored one reads right, its reflection cancelling the view's --
which is why no mirrored fixture could have caught it.

### A dome's share of the lights' power ignored the scene's size

The path tracer chooses one light a sample by power. An area light's power
was its radiance over its area in the scene's units; a dome's was its
radiance alone. In a kitchen in centimetres beside a 120 x 160 window light
the dome's share was 2.4e-6 -- unbiased, and dark and blotchy at 256 paths
a pixel, where the raster, which loops over every light, was lit. A dome of
radiance L lights a scene of radius R with pi L pi R^2 and a sun of
irradiance E with E pi R^2; over the pi common to every emitter these are
L pi R^2 and E R^2, set against an area light's L A. The table takes the
scene's radius; the engine reads it from `GpuScene::worldBounds` (a
kernel's) when the mesh set or its points change. **Checked** by a kernel
writing the shares again from the records at scene radii 1 and 400: 0 of 3
off (worst 3e-8). **Not done**: an instance moving without a change of the
mesh set keeps the old radius, which costs sampling efficiency and not
correctness.

After both, Kitchen_set lit renders under both techniques from inside the
kitchen; the path traced frame at 256 paths is clean without the denoiser.

### Measured (release, Apple M5 Pro)

`lrt stage --frames 11` at 1920x1080 from inside the kitchen, median of
the ten frames after the first (Hydra sync, drawing and the readback):

| Stage | raster | rt (1 path, 1 bounce) |
|---|---|---|
| Kitchen_set, no lights (headlight) | 43.1 ms | 333.3 ms |
| Kitchen_set lit (dome, rect, two spheres) | 96.0 ms | 358.9 ms |

The first frame, which loads the stage and compiles, is 4.5 to 5.5 s.
`lrt view --frames 120` on Kitchen_set lit at 1600x900, framing the whole
set: raster 33.3 ms a frame, rt 127.7 ms a frame while it accumulates
(medians).

### A render settings prim over a real stage

A layer over Kitchen_set lit adds a camera, light groups on its lights
(`sky`, `window`, `bulbs`) and a `RenderSettings` prim with one product of
seven vars -- beauty, depth, primId, normal and the three groups as light
path expressions -- traced at 256 paths and 3 bounces. `lrt stage
--render-settings /Render/Final` writes one EXR of 21 channels in 41 s
(release). Read back and checked with the light group kernel: the groups
sum to the beauty wherever a surface was drawn; the 19% of pixels where
they do not are the dome's background seen through the openings, which is
in no group, as M10 records. The `window` group is empty in this layer
because its rect, turned 90 degrees about Y, emits away from the kitchen:
the bench layer's authoring, not the renderer's.

## End to end: usd-wg assets and OpenUSD's own test stages

Nineteen stages, each drawn by `lrt stage` under raster and under rt (64
paths a pass, 128 in all, 2 bounces) at 640x480: from usd-wg/assets the
standard shader ball, McUsd, the chess set, the carbon frame bike, the
elephant with monochord, the spinning pyramids (subdivision, creases), the
MaterialX texture and texture coordinate tests, the Utah teapot and five
USDZ glTF conversions (DamagedHelmet, CesiumMan, RiggedFigure, BrainStem,
AnimatedCube); from OpenUSD's usdImagingGL testenv the basis curves, curves
with vertex colour, LBS skinning, the skinned arm, blend shapes, a VDB smoke
volume and the simple volumes. A stage without a camera is framed by a new
`lrt stage --frame-all`, which is also what a stage with no camera gets: a
small raster frame commits the scene and the engine's bounds of what it drew
place an orbit camera, as `lrt view` opens. Nine defects, every one found by
a real asset and none by a fixture, each now with a case of its own that
fails without its fix where a control was run:

- **Textures authored relative to their layer did not load.** hdMtlx
  writes a file input as its *authored* path, so `./textures/wood.jpg`
  was read relative to the process. The resolved path, which the value
  carries, goes in first ("a texture authored relative to its layer";
  without the fix the square reads black). The teapot, the shader ball, the
  bike and every USDZ lost their textures to this.
- **A float3 primvar reader into a colour input failed the material.** The
  reader is a vector to MaterialX; UsdPreviewSurface's diffuseColor a
  colour. It takes the colour nodedef where what it feeds is one (a tint
  primvar reads 0.699 0.300 0.499; without, the displayColor fallback).
- **USD's types for UsdUVTexture's scale and bias, and a colour into the
  vector normal.** USD authors float4, MaterialX declares color4; a normal
  map's rgb is a colour into a vector. hdMtlx types inputs by what they were
  given, the nodedef stops matching and the whole material fails. Inputs of
  the same float count take the declared type after hdMtlx builds the
  document. McUsd's 21 materials, the bike's, the helmet's and the
  elephant's all failed so. hdMtlx still prints its own mismatch messages
  before the fix-up runs: noise, not failure.
- **UsdUVTexture's wrap `repeat`** is `periodic` in MaterialX's enum, and
  `useMetadata` its default: mapped (CesiumMan and AnimatedCube failed).
- **Material binding purposes.** The delegate never overrode
  `GetMaterialBindingPurpose`, so Hydra's default, `preview`, was resolved,
  and a `material:binding:full` -- the shader ball's walls -- was never
  seen; M10's note said the opposite. The delegate answers `full`, and
  `StageRenderer` resolves a settings prim's `materialBindingPurposes` in a
  filtering scene index of its own (`BindingPurposes.h`), since hdsi's
  resolver fixes its purposes when made and a chain the render index
  observes cannot swap it. Changing the list dirties every prim with
  bindings; `HdChangeTracker::MarkAllRprimsDirty` does not, under scene
  index emulation, reach prims the stage scene index owns (measured: the
  test's square stayed blue). The case: full, preview then all-purpose,
  all-purpose alone, and a settings prim's list through its products.
- **Primvars on the material** (the blend shapes stage's
  `primvars:displayColor` on its Material) reach bound geometry only
  through `HdsiMaterialPrimvarTransferSceneIndex`, which Storm registers for
  itself: now registered for this renderer too, in Storm's phase. The blend
  shapes read green, as the baseline image does.
- **Materials identical but for their name compiled apart.** The generator
  names the surface's variables after the renderable element, which hdMtlx
  names after the material prim, so the shader ball's 17 materials were 13
  modules of 5 sources. The renderable is renamed before generation (17
  materials, 4 modules; "materials that differ in name and values alone
  share one module").
- **The path traced kernel ran the Metal compiler service out.** With the
  shader ball's 13 modules and the chess set's 15, `tracePaths` failed
  ("XPC_ERROR_CONNECTION_INTERRUPTED ... after multiple retries") after 4 to
  5 minutes. The front end is not it: `metal -c` on the translated source
  takes 0.6 s. Timed with a scratch tool that makes the library and the
  pipeline from the same source (chess, 15 materials, 37k lines of MSL):
  as generated, 233.6 s and the service gives up; with only the material
  dispatch marked `noinline`, 5.8 s. The dispatch was reached from five
  call sites -- the camera's hit, a lens sample's, a surface bounce's, a
  medium bounce's -- and each inlined copy carried every material. Slang
  accepts `[noinline]` and emits nothing for Metal, so the kernel was
  restructured instead: finding a hit (`Found`) and evaluating its material
  are apart, and materials are evaluated at one site in the vertex loop,
  the camera's hit still once a pixel. Chess under rt: 31 s for the whole
  command, the shader ball 30 s. The one behaviour that moved: the aux
  planes carry the first *shaded* surface hit, which differs from the old
  first hit only when a medium scatters in front of it on the first path.
  The raster's shading kernel has one call site and never showed it.
- **A stage without lights path traced black** while the raster lit it
  with its headlight. The engine now asks the path tracer for the same
  headlight when the frame has no lights (`PathSettings::headlight`), at the
  first vertex only: an unlit stage is the same image under both (p99 0,
  max 0). It is the engine's to ask and not the kernel's to assume: the
  closed emissive shell, lit by its emission alone, read 1.67 times its
  series when the kernel lit every lightless frame.

**lrt view and lrt live over the same stages** (release, Apple M5 Pro):

- `lrt view --play` starts the timeline as its Play button does, and
  `--frames` now reports how many distinct times the frames drew. The
  skinned arm (raster, 800x600): 240 frames, 240 distinct times, wrapping
  at the end, draw 8.7 ms; CesiumMan path traced: 240 of 240, draw 21.4 ms,
  the snapshot mid-walk. Without `--play`, 1 time over 120 frames. The
  viewer's technique menu said "Ray traced (splats)" for rt, from before rt
  path traced surfaces: "Path traced".
- `lrt live` renders a stage without cameras from the framing camera
  `lrt stage` uses (`StageRenderer::framingCamera`); it refused them. At
  25 fps on this machine's clock: CesiumMan raster at 1280x720, 50 frames,
  render 19.5 ms, none skipped; Kitchen_set raster at 1920x1080, 50 frames,
  render 25 ms, none skipped; the chess set path traced at 1280x720, one
  pass of one path, render 139 ms, so 25 frames written and 75 skipped for
  time -- said, not hidden.

Not defects, and left as they are:

- **McUsd blows out.** Its DistantLight and DomeLight leave intensity
  unauthored ("no intensity often helps the viewer pick a default"), and
  UsdLux's default for a distant light is 50000: with its 1 degree angle
  that is an illuminance of about 12. The renderer follows the schema.
- **The MaterialX texture test's teapot is black**: its `.mtlx` sets
  `fileprefix="./textures/"` and also writes `./textures/` in every value,
  so the path is `./textures/./textures/brass_color.jpg`; its own flattened
  sibling authors `textures/brass_color.jpg`.
- **glslfx materials** (the VDB test's checkerboard, the simple volumes'
  ellipsoids) are Storm's own and have no MaterialX network: displayColor.
- **The chess set's glass pawn heads** are black under the raster's
  headlight, which has no transmission.
- The shader ball under raster is dim beside rt: the box is lit mostly by
  its bounces, which raster does not draw.

## Linux, on the 94 (M11's first half)

### The first table, after the port was reconciled with engine

`engine` at the M6 path tracer, built with GCC 13 on the L4 -- once
`retainedDataSource.h` was patched: GCC rejects the injected-class-name written
with its template arguments in the bool specialisation's constructor
(`HdRetainedTypedSampledDataSource<bool>(const bool&)`), which only a
translation unit including that header meets, and the light linking's retained
data sources do. `scripts/build-usd.sh` now patches it after install so the two
machines build against the same prefix. Then `ctest`: **107 of 119 passed, 12
failed, 38 skipped** (no display, no gpe on this backend, no OptiX). The twelve,
before any of them is looked at: six of the splat ray tracer (the tests that
choose the Hardware route, with no OptiX to give one), three of eight-bit
textures (the port's own section below on what a float4 store becomes), the
lobes against MaterialX's genglsl, and two Hydra cases with splats. That is the
table M11 starts from; the plan's order verifies each new piece on both
devices from here.

The engine built and ran on Linux for the first time: Ubuntu 24.04, an NVIDIA
L4, CUDA as the backend. What follows is what the port needed, what it found
and what is still wrong.

### The toolchain

- **OpenUSD 26.08** with MaterialX 1.39.5 and OpenVDB/NanoVDB, from
  `scripts/build-usd.sh`. `--ignore-homebrew` is a macOS-only option of
  `build_usd.py`, so the script passes it only there.
- **OIDN 2.5.1** with the CUDA device. Its CUDA device wants CUDA 12.8 or
  newer and Ubuntu 24.04 ships 12.0, so `cuda-toolkit-12-8` goes beside it and
  both `scripts/build-oidn.sh` and the top-level CMake pick the newest
  `/usr/local/cuda-*` (`LRT_CUDA_ROOT` overrides). One toolkit for gpe's
  kernels, slang-rhi's CUDA device and OIDN.
- **zlib** is found at the top level now: an imported target belongs to the
  directory that found it, and the tests link `ZLIB::ZLIB` too.
- **`<cstring>`** before slang-rhi's `acceleration-structure-utils.h`, which
  calls `memcpy` without including it -- libc++ carries it in anyway,
  libstdc++ does not.

### The CUDA driver's names were slang-rhi's variables

slang-rhi loads the CUDA driver with `dlopen` and keeps its entry points in
variables named after the driver's own functions (`cuInit`, `cuLaunchKernel`,
...). At global scope those variables are the definitions the rest of the
program binds to, and a strong data definition in an object file beats a
shared library's function whatever the link order -- both orders were tried.

So gpe and OIDN, which call the driver directly, called through slang-rhi's
pointers instead of through libcuda and died on a null one: `lrt info` exited
139 after printing correctly, and the gpu_host tests, every aofx host test and
`single_tbb` crashed in what the backtrace called `cuInit ()` at an address in
the executable's BSS.

`cmake/patches/slang-rhi-cuda-driver-symbols.patch` puts the block in
`namespace rhi::cuda_driver` with a using-declaration after it: the names keep
working inside slang-rhi and collide with nothing outside it. After it,
`cuInit` is undefined in our binaries (resolved from libcuda), `lrt info`
exits 0 and reports `denoiser OIDN 2.5.1 on CUDA` and `tbb libraries 1`, the
aofx tests are green and `single_tbb` passes.

### The radix sort on CUDA: a cursor re-read after it was advanced

For as little as two pairs, one chunk and one pass: the generator writes
`(1766601275, v0) (3252568193, v1)` and the sort leaves
`(1766601275, v1) (0, v0's slot never written)` -- two destinations collided.
What that rules out, each measured rather than argued:

- **The generator** is right: the probe dumps the pairs before the sort.
- **Ordering between passes** is not it: CUDA launches all go on one stream,
  and `LRT_RADIX_SUBMIT_EACH_PASS=1` -- a submit and a wait between every pass
  -- changes nothing, the same 89 assertions fail.
- **The constants** reach each dispatch: four dispatches queued into one batch
  with four different parameter blocks each read back their own `count`,
  `chunkCount` and `shift`.

- **The counting, totalling and cursor stages are right.** One pass over two
  pairs (eight key bits is one pass) leaves a histogram counting both pairs,
  totals that agree, and 197 cursors past zero -- the same 197 as Metal.
  `RadixSort::working` names those buffers so a test can say so.
- **The bindings land where they are named.** Seven buffers, each holding its
  own marker, bound by the names the kernel declares: every name reads its
  own marker on CUDA as on Metal.
- **A scatter-shaped kernel reads what it was given.** The same seven
  bindings, one invocation, two elements, no sorting arithmetic: both elements
  read their own key and value on CUDA as on Metal. Its writes were another
  matter, and they are what gave the answer away: with the cursors zeroed, the
  probe reported taking slots 1 and 2 where Metal took 0 and 1, and the key of
  a pair landed one slot past the value written beside it from the same local.
- **Two wrong turns, recorded so they are not taken again.** Changing which
  buffer the scatter's unused high-word names point at appeared to change a
  low word the sort placed, and the slots the probe reported appeared to be
  off by one. Both readings came from buffers nobody had written -- a buffer a
  test makes is not zeroed -- and both evaporated once the probe wrote its own
  cursors. The sort itself never reads an uninitialised cursor: radix_starts
  writes every entry of every chunk's row. Giving the scatter distinct
  stand-ins made CUDA worse (four of the sort's cases passing fell to one), so
  it binds dummy_ and the histogram as it always did.

What was left was the generated code, and reading it ended the hunt. Slang's
CUDA emitter does not materialise `const uint at = chunkStarts[slot]`; it
keeps the cursor's address and re-reads through it, after the store that
advanced it:

```cuda
uint * _S14 = &chunkStarts_0[slot_0];
*(&chunkStarts_0[slot_0]) = *_S14 + 1U;   // the cursor now holds at + 1
*(&dstKeysLo_0[*_S14]) = _S9;             // re-read: one slot past
uint * _S16 = &dstValues_0[at_0];         // materialised: the right slot
```

One `at` in the source, two indices in the object code -- which is why a key
landed one slot past the value written beside it, and why every stage feeding
the scatter measured correct. The Metal emitter materialises the load, so the
suite here never saw it:

```metal
*((&kernelContext_0)->dstKeysLo_0+at_0) = lo_1;
```

Reduced to the smallest thing that shows it, for whoever takes this upstream
-- a load, a store that advances it, and a second use of the load:

```slang
RWStructuredBuffer<uint> cursor;   // cursor[0] starts at 0
RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(1, 1, 1)]
void repro(uint3 tid: SV_DispatchThreadID) {
    const uint at = cursor[0];
    cursor[0] = at + 1;
    out[at] = 7;   // CUDA writes out[1]; Metal writes out[0]
}
```

**The fix is to place the pair and then advance the cursor.** With no store to
the cursor between its load and the uses, there is nothing for an emitter to
re-read, and the walk is one invocation's own, so the order costs nothing and
the sort stays stable. `slangc -target cuda` confirms it on the generated code
before any device runs it, which is the cheapest way to check this class of
bug and worth reaching for earlier next time: four stages were measured right
one at a time when one look at the emitted kernel would have said why.

### What else the port found

- **`SampleGrad` was not available in a compute entry point on the CUDA
  target** (Slang `E36107`). The material system answers that in the shader,
  with a `__target_switch` choosing gradients or an explicit level at the
  footprint's wider side, so the kernels build here now. What is left is not a
  capability error but wrong pixels, and reading the emitted CUDA says why.
  A decoded 8-bit image is an `RGBA8Unorm` texture written through an
  `RWTexture2D<float4>`, which on this target becomes
  `surf2Dwrite<float4>(texel, surf, x * 16, y)`: sixteen raw bytes of float at
  a sixteen-byte stride, into a surface holding four bytes a texel. CUDA
  surface writes do not convert formats; Metal's texture write does. So every
  component is wrong and each row runs four times past its end, which is the
  3386-of-3404 mismatch and the mip means collapsing to zero. No intrinsic is
  missing, so no `__target_switch` reaches it -- it is the store itself that
  means something different on the two targets. The float and half formats go
  through the same kernel untouched, which is why the gpu texture tests, which
  build `RGBA32Float`, pass here while the material ones do not.

  Measured, rather than read off the emitted code: a probe writes
  `float4(1, 0, 64/255, 1)` through an `RWTexture2D<float4>` into one texel of
  an `RGBA8Unorm` texture and reads that texel back, with no decoding,
  sampling or mips in the way. Metal returns the colour written. CUDA returns
  `(0.0, 0.0, 0.502, 0.247)` -- which is `00 00 80 3F`, the four bytes of
  `1.0f`, sitting in the texel as bytes. The store wrote the float4's memory,
  not its colour; the second, third and fourth components landed in the next
  three texels along, which is also why each row runs four times past its end.
- **gpu_host's `OutOfMemory` was never about memory.** A CUDA context is
  current per *thread*, and `CudaDevice::open` makes it current on the thread
  that creates the Context -- not on the worker `Context::run` spawns after
  it. gpe guards its own entry points (`ensureCurrent` before each), so gpe's
  work was fine; the work queued on the GPU thread reaches the same context
  through slang-rhi, so a buffer allocated there was the first driver call on
  a thread holding no context at all. 16 GB free, 4 KB refused, and the
  identical allocation off the thread succeeding. gpe grew a `bindThread()`
  for exactly this -- a caller about to reach the shared context another way
  -- and the GPU thread binds once before it takes work. Metal keeps no such
  per-thread state and takes the default, which does nothing.
- **A buffer nobody wrote is not a buffer of zeros, in the engine too.**
  `meshHoles` marks a hole face with a 1 and leaves every other face alone,
  and nothing else wrote `holeFlags`, so each face was judged by whatever the
  device last left there. Metal returned zeros; CUDA did not, and a stale word
  reads as "this face is a hole", dropping its triangles. The mixed-face mesh
  triangulated to 4 of its 10 triangles, and to none with the handedness
  flipped. The flags are cleared on the device before the topology dispatches.
- **Two failures that look like load, and are not.** The test presets already
  set `execution: { jobs: 1 }`, so the suite is serial as it stands. Run
  entirely alone on an idle box, the free-running clock test (#93) still fails
  in 0.4 seconds, twice running, and the lobe test (#52) still fails in 7.4
  seconds. Neither is contention and neither should be written off as the
  machine: #93 is a timing bound this box does not meet and #52 is a Monte
  Carlo tolerance, and both want measuring on their own terms.
- **Tests skip rather than fail where CUDA cannot answer**: no rasterisation,
  so the raster visibility, mesh and Storm-oracle tests skip with a reason, as
  does `lrt view` with no display.

### Open on this backend

**A float4 stored through an `RWTexture2D` does not arrive as the texture's
format on CUDA.** `lrt_gpu_tests` "a float4 written to an 8-bit texture comes
back as the colour it wrote" fails here, deliberately: it is the smallest
statement of the defect, one texel with no decoding, sampling or mips in the
way, and it names what it means when it fails. It is not skipped, because CUDA
can run it -- it answers wrongly, and a skip would hide that behind a green
suite. Skipping is for what a backend cannot do.

Everything the texture store decodes from an 8-bit image is wrong here until
it is fixed, which is most of what still fails: the PNG and UDIM tests
directly, and the ray tracer and USD tests that shade through a decoded image.
The fix is ours rather than Slang's -- the lowering is faithful, and the two
targets simply mean different things by the store. The obvious route, packing
the texel in the shader and storing it through a uint view of the same
texture, was tried and measured, and it does not hold: **the two backends fail
in opposite places.**

- The `float4` store converts on Metal and writes the float's own bytes on
  CUDA.
- A uint (`R32Uint`) view of the same `RGBA8Unorm` texture aliases correctly
  on CUDA -- written packed, it reads back as the colour through the texture's
  own view -- and does not alias on Metal, where the store lands (read back
  through the uint view, 0 of 256 texels differ) but is invisible through the
  colour view (256 of 256 differ). Metal wants the texture created able to be
  viewed as another format, which slang-rhi does not ask for.

So the packed route trades a CUDA bug for a Metal one, and was reverted after
being measured; Metal is back to its 451 assertions exactly. What is left is
to decode into a buffer and copy the buffer into the texture, which asks
neither backend to reinterpret anything. That is not attempted here.

`lrt_gpu_tests` keeps the probes that establish all of the above, and they
pass on both backends bar the one that names the defect.

### Measured (NVIDIA L4, Ubuntu 24.04, debug)

`ctest --preset linux-x86_64-debug`, with engine's materials merged in:
**91 of 104 pass, 13 fail, 27 of those passes skips** -- from 47 of 83 when the
port first ran, and from 63 of 97 before the sort was fixed. One of the
thirteen is the eight-bit texture probe above, which fails here deliberately
and says why when it does; the other twelve are the texture surface write and
the two measured on their own below. Lights (M5) are merged and build here,
and their tests run: the count grew from 97 to 104 with them.
Passing outright: the prefix sum, textures, mips, the texture table and its
sRGB views, the shader cache and link constants, every loader (PLY, .splat,
SPZ, SOG, points), the lobe library, the display transform, the codeless
schemas, the hdLrt plugin, timecode and PTP, the aofx host and its SDK
manifest hash, the whole sort, and gpe sharing the device. What is left is the
texture surface write (#46, #48, and the ray tracer and USD tests that shade
through it), the hole flags (#43, fixed after this reading), and the two
measured above.

### The second half's tools: a backend by environment, labels, the remote run

- **`LRT_BACKEND=cuda|vulkan|metal|d3d12`** (an order, comma-separated)
  chooses the backend for a whole run where a caller left it to the
  platform, so one suite runs once a backend at a time; `lrt info` shows
  which was taken.
- **ctest labels.** Every case carries `gpu`; `lrt_view_tests` carries
  `display` and the gpe-backed binaries (`lrt_gpu_host_tests`,
  `lrt_aofx_tests`) `gpe`, so a machine without a window or without gpe
  excludes them by label (`ctest -LE display`) instead of reading their
  skips as passes. A case that skips for a capability the device lacks --
  rasterisation, ray queries, a uint view of an 8-bit texture -- says so in
  its skip message; that is per case, where a label is per binary.
- **`scripts/remote-test.sh [user@host] [preset] [branch]`** checks the
  branch out on the machine, builds the preset, runs ctest and prints the
  table -- passed, failed, and every skip with the reason it gave -- from
  the remote's `LastTest.log`, which it brings back to
  `build/remote/<preset>/`. A skip is not a pass, and the table says which is
  which.
- **A test's counters start from zero on purpose.** `test::uintBuffer` now
  writes zeros: the light BVH's draws test read back 1053144244 draws of a
  million on the L4, the counter having started from what the device last
  left there -- the same lesson the engine's hole flags taught, arriving in
  the tests.

### The two devices at the end of M9 (commit c94f497)

Parity here means the same GPU-computed metric on each machine,
computed on that machine; no pixels travel between them.

| | Metal (Apple M5 Pro) | CUDA (NVIDIA L4, no OptiX) |
|---|---|---|
| Cases | 168 | 168 |
| Passed | 167 | 89 |
| Failed | 1: openFXplayer's bundles are ABI 23, this host speaks 22 | 12, the known ones below |
| Skipped, with reason | 0 | 67 |

CUDA's 67 skips, by the reason each gives: 43 "no rasterisation on this
device", 13 "needs rasterisation and ray queries" (two of them "to compare
all three routes"), 2 "needs both rasterisation and ray queries", 4 points
drawn as discs because the device does not rasterise, and one each for no
window session, no openFXplayer build, Kitchen_set not on the machine, a
Metal-only tracked buffer, chunks being the hardware route's, and one with
an empty message. Every one names a capability CUDA without OptiX lacks, or
an asset or host that is not there.

CUDA's 12 failures are the ones "Open on this backend" already explains:
the float4 store into an 8-bit texture (and what decodes through it: PNG,
UDIM, the lobes against genglsl, the two Hydra splat cases), and the
Gaussian ray tracer's six cases. None is new.

New on both devices: `lrt_volume_tests` -- the NanoVDB layout counted
voxel for voxel, and the medium's Beer-Lambert, majorant and leaf-walk
checks -- passes on CUDA as on Metal, so PNanoVDB as Slang holds on both.
The path-traced volume tests need ray queries, so they skip on CUDA.

**Vulkan is not measured.** slang-rhi's Vulkan backend is built on the 94,
but the machine has only Mesa's Vulkan drivers, and `lvp` (lavapipe) runs
on the CPU, which is no device for this table. With NVIDIA's Vulkan driver
(`libnvidia-gl-580-server`) installed, `LRT_BACKEND=vulkan
scripts/remote-test.sh` runs the same suite on the L4 with rasterisation and
ray queries -- the column most of CUDA's skips would move to.

### A note on WebGPU

Everything a frame does is Slang, and Slang emits WGSL, so the shading,
the materials, the lights and the compute BVH route would port. What would
not, without work: there are no ray queries, so the path tracer would trace
over `BvhScene` alone; buffers bind through bind groups with a small limit
per stage and no bindless textures, which the texture table and the
kernels at Metal's 31-buffer limit would both have to be reshaped for; no
fp64, no OIDN, no OpenUSD in a browser -- so the delegate stays native and
a WebGPU build would take the engine's records, not Hydra's prims.

### Not done

- The texture surface write: an 8-bit texture written as float4 needs either a
  target the backend converts into or a store that converts itself. Until then
  every test that shades through a decoded 8-bit image is wrong here.
- A release build and any timing beyond the sort's own.
- Vulkan: the backend is compiled in and untried, since CUDA is what gpe
  shares.
- OptiX: absent on this box, so slang-rhi warns and falls back to CUDA
  compute.
