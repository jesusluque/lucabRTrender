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

## Linux, on the 94 (M11's first half)

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

### Measured (NVIDIA L4, Ubuntu 24.04, debug)

`ctest --preset linux-x86_64-debug`, with engine's materials merged in:
**84 of 97 pass, 13 fail, 24 skip** -- from 47 of 83 when the port first ran,
and from 63 of 97 before the sort was fixed. That reading predates the hole
flags fix, so #43 is still counted as failing in it.
Passing outright: the prefix sum, textures, mips, the texture table and its
sRGB views, the shader cache and link constants, every loader (PLY, .splat,
SPZ, SOG, points), the lobe library, the display transform, the codeless
schemas, the hdLrt plugin, timecode and PTP, the aofx host and its SDK
manifest hash, the whole sort, and gpe sharing the device. What is left is the
texture surface write (#46, #48, and the ray tracer and USD tests that shade
through it), the hole flags (#43, fixed after this reading), and the two
measured above.

### Not done

- The texture surface write: an 8-bit texture written as float4 needs either a
  target the backend converts into or a store that converts itself. Until then
  every test that shades through a decoded 8-bit image is wrong here.
- A release build and any timing beyond the sort's own.
- Vulkan: the backend is compiled in and untried, since CUDA is what gpe
  shares.
- OptiX: absent on this box, so slang-rhi warns and falls back to CUDA
  compute.
