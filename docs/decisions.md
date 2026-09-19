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
- **MIS is the path tracer's** (M6, below): the raster's shading samples the
  lights alone.
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
- **A relit splat casts a shadow ray now** (`lrt:splatShadows`, off by
  default): one ray a splat against the cloud's own proxies, through
  `rt_shadow.slang`'s product of `1 - alpha`. The rasteriser's projection
  kernel traces it, and the proxies come from a tracer of the engine's own on
  the hardware route (`GaussianRayTracer::prepare`), since the frame's tracer
  may be on the compute route, which has no structure an inline ray can walk.
  - **The ray starts past the splat's own neighbourhood**, `shadowOffset`
    sigmas of the splat itself (3 by default). A captured surface is a crowd
    of overlapping Gaussians and a ray that starts at one peaks inside its
    neighbours within a fraction of their size: without the bias a relit
    capture renders black, every splat shadowed by the splats it is made of.
    Measured, and black is what it looked like.
  - **A frame of splats alone had no lights at all.** The light table was
    built only where there was a mesh layer, so a stage of a relit capture
    under a light showed what it was baked with -- the feature had only ever
    been exercised through the render API, never through USD. A frame with a
    relit cloud and no mesh now builds the table (and takes the cloud's own
    bounds as the scene's reach), and nothing else of what a mesh layer needs.
  - **What it does not fix** is what relighting approximates: the normal is
    still the splat's shortest axis and the albedo the harmonics' constant
    term, so a photogrammetric capture relights streaky whatever the shadow
    does. The train at 960x540 shows the key light's shadow across the whole
    body, and the same streaks as before under it.
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
- **MIS, since the usd-wg end to end** (the section after M6's "not done").
  At first next event estimation covered the analytic lights and sampling
  the material covered emissive geometry, disjoint sets with no weight; a
  one-sided power heuristic had been wrong -- see below.
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
- On Vulkan OIDN has no device of its own, so it runs on the same GPU through
  CUDA and imports the staging buffers' memory: they are created
  `BufferUsage::Shared`, slang-rhi exports an opaque file descriptor
  (`getSharedHandle`), and `oidnNewSharedBufferFromFD` takes a duplicate of
  it -- OIDN owns the descriptor it is given, slang-rhi keeps its own. The
  copies in and out are Metal's kernels. `create` refuses where the CUDA
  device reports no `OPAQUE_FD` in `externalMemoryTypes` rather than
  producing a wrong image.

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
- **A ray that only needs transmittance** has its own kernel now,
  `shaders/lrt/rt/rt_shadow.slang`: no k-buffer, no segments, no order.
  Transmittance is a product of `1 - alpha` and a product does not care in
  what order its terms arrive, so every proxy the traversal offers is taken
  as it comes and the ray stops once the product falls under its cut. Three
  closed forms hold it: a ray through a particle's centre peaks at power 0,
  so it lets exactly `1 - opacity` through (0.50000 measured against 0.5);
  four such particles give `(1 - opacity)^4` (0.06250); and with the cut
  raised to 0.3 the ray stops after three of the four, which the counter
  says (1 ray cut, 3 particles taken).
  - **Back faces are not culled here.** A shadow ray is born on a surface,
    and a surface inside a cloud is surrounded by proxies: with culling on,
    a ray starting inside a proxy sees only the exit face and misses the
    particle it stands in -- exactly the particles whose shadow touches the
    geometry. Measured both ways in the same binary: born inside with the
    peak ahead, 0.50000 against the closed form; with culling on (the
    control), 1.00000 and 0 particles taken.
  - **Both faces then arrive**, so a ring of the last 16 particles taken
    collapses them: 5 particles taken and 5 duplicates caught in the stack
    test, where `rt_integrate` does it by comparing with the last particle
    blended after sorting.
- **The integrator answers a query, not a pixel.** `rtTraceSegment(ray,
  windowMin, windowMax)` returns a `SegmentResult` -- radiance premultiplied
  by what it covered, the transmittance left, the depth -- and `writeSegment`
  turns one into a pixel, so the camera's kernel and a secondary ray ask the
  same thing. `segmentOver(near, far)` composes two. The windows are half
  open, `[a, b)`, so a peak exactly on a cut is taken once.
  - **What the check found.** Drawing a frame as `[near, s)` over `[s, far)`
    and comparing it with the one query: the far query first traversed from
    `s`, and a proxy entered before the cut that peaks after it was lost
    entirely -- max 199 of 255 on 445 pixels of a sparse cloud. The traversal
    and the window are two different things: a query walks from where the ray
    starts and takes the peaks in its window. After that, a sparse cloud is
    **bit for bit** (max 0) and a dense one differs by 1 code at worst with 0
    pixels over 2 -- the carry (`kCarry`), which the header already names as
    the one place order can be approximate.
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
- **Emissive geometry is a light** since the motion work (below, "Emitting
  triangles as a light").

### Multiple importance sampling

**The missing piece was the other direction.** `lightHit(l, p, wi)` (and
`lightHitImaged`, with a dome's image) says where a ray from p along wi
meets a light and the radiance it carries: a sphere's near root, a disk's
or a rect's plane within the shape, a cylinder's lateral surface, and for a
dome or a distant light's cone an infinite distance only an escaping ray
reaches -- the shaping cone and the IES profile applied as `sampleLight`
applies them. Checked in the chi-square tests beside `lightPdf`: every
sampled direction of the sphere, disk, rect, sun, dome, cylinder and a dome
with an image is found again by `lightHit` at the sampled distance and with
the sampled radiance, 0 mismatches each.

**The weights.** At a vertex the path leaves by sampling its material, next
event estimation's sample of light k is weighed by the power heuristic
against `stackPdf` in its direction, and the material's sampled ray -- after
it is traced -- gathers every light it meets before the surface it found
(or, escaping, the domes and distant lights), each weighed against the
density next event estimation would have drawn that direction with: the
light's choice probability at p (by power, or `lightPdfChoiceAny` under the
light BVH) times `lightPdfImaged`. A delta light and a delta lobe keep a
weight of one, as does the last vertex, which samples no material.
`lrt:pathMis` (default on; `StageRenderer::setPathMis`) switches it off.

**What is left out, and why.** A light that casts no shadow, or whose shadow
links leave occluders out, keeps its weight of one: the material's ray is
stopped by any surface and would see another visibility. So is every light
in a frame with volumes (a medium would have to dim the material's ray as it
dims the shadow ray), and in a frame of more than 64 lights, since each
bounce tests its direction against every light.

**A defect before it passed:** an escaping ray's distance and a dome's
were the same 1e30, and `t >= reached` dropped every dome. Deep MIS then
converged (8192 against 32768 paths, 2e-5 apart) to an image off by 100% at
the 99th percentile -- visible only by comparing it with a deep frame of
light sampling alone.

**Checked** on a floor of metal (roughness 0.2) under each light, one
bounce, 64x48: deep frames of 8192 paths with and without MIS agree within
their noise, and at 32 paths against the deep frame without, MIS is 4.5
times less error under a 4x2 rect, 57.7 times under an imageless dome and
2.4 times under a sphere of radius 0.8. The glossy dome's deep frame without
MIS is heavy tailed (8192 against 32768 paths, 3.4e-2 apart), so its
sameness is shown on a rough diffuse floor under the same dome, where light
sampling converges: deep frames 2.6e-7 apart, their summed noise about
4e-7, and MIS 1.1 times less error there, as expected where the material's
density and the dome's are the same function.
### Emitting triangles as a light

**The table** (`technique::EmissiveTable`). Each material row's emission is
probed once on the device -- the material evaluated at a neutral point, its
emission's luminance -- and every triangle of the frame's records weighed by
its world area times its row's luminance, accumulated in order into a
distribution. It lives in one float buffer with its counts, the rows'
luminances and each record's first triangle, since the path tracer's kernel
had one binding left. The probe shapes where samples go, not what they
carry: a textured emitter whose probe point is dark is simply left to the
material's rays, and nothing is biased by it. Rebuilt when the scene, its
positions, the materials or the engine's revision change; not in a frame
with volumes, whose kernel does not sample it.

**Sampling it without a second material call.** The emission a sample
brings back must be the material's at the point it lands on, and a second
place in the kernel that evaluates materials is what ran the Metal compiler
out (the usd-wg section). So the vertex loop became steps: a step shades
either a vertex of the path or the point next event estimation chose on an
emitting triangle, through the one call; a vertex that chose such a point
waits a step, takes its emission, and goes on. The trip count is a
uniform's, so the loop cannot be unrolled into copies. The choice between
the lights and the emitting triangles is by power (an area light's L A
against a triangle's area times luminance), and each side's density carries
the other's share.

**The weights.** Next event estimation's sample of a triangle is weighed by
the power heuristic against the material's density in its direction; emission
the material's ray meets is weighed against the density next event estimation
gives that point -- the triangle's tabled power over the total, times its
distance squared over the cosine and the area, where the area cancels. Both
sides take that tabled density, so the weights sum to one; the estimate
divides by the density the sample was actually drawn with. Emission is
two-sided in both. With `lrt:pathMis` off, next event estimation leaves the
emitting triangles to the material's rays, as before -- otherwise both would
count them.

**A defect on the way:** the shadow ray to the chosen point was cut a
relative 1e-4 short, but `pathOccluded` moves its origin up to 2e-3 of the
scale along the ray, so it reached the emitting triangle and every sample
was its own shadow: the frame came out black. It now stops 3e-3 of the scale
short.

**Checked** through Hydra: a 1 x 1 quad whose MaterialX `surface_unlit`
emits 3, above the frame, lighting a floor, against a UsdLux rect light of
the same size and radiance in its place (one bounce, 64x48): deep frames of
8192 paths agree to relMSE 4.1e-7, and at 32 paths the error is 8.7e-5 sampled
as a light against 11.75 by the material's rays alone.

**Not done.** Emitting triangles are not sampled from inside media, nor
under motion at their shutter slice (the table is the frame's). A table of
millions of triangles accumulates in float: a triangle whose power is below
the running total's precision is sampled with a rounded probability.

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

**Instancers move.** Under a shutter `HdLrtInstancer` samples each of its
per-instance arrays (`SamplePrimvar` of translations, rotations, scales and
transforms) and its own transform (`SampleInstancerTransform`) about the
shutter, keeping the samples' times, as a mesh does; the engine composes the
chain three times -- at the frame, and at the two samples, each level from
its own samples where it has them and the frame's arrays where it does not
-- and hands the set both motion chains. `GpuScene` puts moving sets first
among the sets, copies their records per shutter slice after the single
instances' copies (`instanceRecordsMotion`: the chain rows interpolated to
the slice's centre, times the prototype, answering to the slice's bit), and
the acceleration structures take those copies instead of the moving sets'
frame records -- the range stays contiguous because the moving sets lead.
Checked: two squares under a PointInstancer whose positions are time
sampled, path traced in eight slices, are bit for bit the same squares
authored as two meshes sliding by their transforms (relMSE 0, max 0), and
differ from the shutter closed (0.92); with the motion chains withheld from
the scene the instancer drew sharp, and the comparison failed.

**The camera moves.** The delegate's camera sprim is `HdLrtCamera`, hd's
`HdCamera` that also samples its transform about the shutter; the pass
hands the projection view to world at both samples (the camera's transform,
then the flip to +z -- no inverse) with their times. A moving camera makes
the scene cut the frame into shutter slices even when nothing else moves,
and turns on the path tracer's own primary rays, each sample's camera
interpolated to the centre of the slice its rays answer to -- the time the
moving geometry it meets is drawn at. The raster draws the frame's camera.
Checked: a camera sliding +x over two still squares, path traced in eight
slices, is bit for bit a still camera over the squares sliding -x (relMSE 0,
max 0), and differs from the shutter closed (0.92); with the samples
withheld from the kernel the camera drew sharp and the comparison failed.

**Lights move.** A light samples its transform about the shutter as a mesh
does; the table carries, only when some light moves, 26 floats a record
after the light BVH's nodes in the IES values buffer -- the rows at both
samples and their times, a still record's times equal -- since the path
tracer's kernel binds its 31 buffers already. A moving light makes the
frame's slices as moving geometry does, and each sample places the light it
chose (next event estimation, media, and the material's rays under MIS)
between its samples at the centre of the slice its rays answer to
(`lightFor`); its choice by power, and the light BVH's, stay the frame's.
Checked by the relative scene: a sphere light sliding over a floor with an
occluder, under a still camera, against the light still and the camera,
floor and occluder sliding the other way -- relMSE 4.9e-9 (p99 relative
1.7e-5) -- while against the light standing still the frames part by
2.8e-4, a soft shadow sweeping a part of the floor. With the samples
withheld from the kernel the moving light drew the still frame (3.7e-13).
(The test's first arrangement was wrong, not the renderer: the reversed
stage put the occluder two units from the light at mid frame instead of
one.)

**A shutter that changes after prims synced.** The pass marked every rprim
dirty through the change tracker, and under scene index emulation those
marks do not reach prims the stage's scene index owns (as with binding
purposes): a camera whose shutter was authored open after a first frame drew
the next frame half resampled, relMSE 1.18 from the stage authored so.
`StageRenderer` now dirties every prim's transform and primvars through its
own filtering scene index when the shutter it reads differs from the one
the prims were sampled about -- meshes, instancers, lights and the camera
alike -- and the frame after the edit is bit for bit the authored stage's.

**Instanced lights move.** A light under an instancer composes its chain
at the shutter's samples as a mesh does (`composeChains`, one helper for
both now), and the table writes each copy's 26 floats as the prototype's
rows at the two samples; `lightInstancesMotion` places them by the chain's
rows at the same samples on the device, as `lightInstances` places the
frame's. The times are the chain's where it moves, the light's where only
the prototype does. Checked in the moving light's test: the bulb as a
`PointInstancer` prototype whose one position slides 0 -> 2, against the
bulb's own transform sliding so -- relMSE 0, max 0; with the chain's
samples withheld the instanced frame parted from it by 3.1e-4, the still
light's distance.

**A host driving the plugin gets the same.** The pass's change tracker
marks could never do it -- under scene index emulation they do not reach
prims a scene index owns, and without emulation (UsdImagingGLEngine's
chain) `MarkRprimDirty` is refused outright as "requires emulation". So the
delegate registers a pass-through scene index of its own for this renderer
(`HdLrtResampleSceneIndex`, phase 4 at the end), which every chain built
for `lucabRTrender` holds, and finds it by walking the inputs of the
terminal scene index the render index hands it. `StageRenderer` uses the
same call. A shutter changed after the prims synced leaves two frames
unconverged, whatever they hold: the one that drew the old samples and the
one that resamples -- otherwise a host that draws until `IsConverged` stops
at the stale frame. Checked in `lrt_host_tests`, an executable that runs
`UsdImagingGLEngine` -- usdview's engine -- on the plugin loaded by name:
the frame after the edit is the fresh engine's on the edited stage (relMSE
0, max 0) and differs from the sharp one by 1.31. That executable links no
`lrt::usd`: with the delegate's classes in the executable as well as in the
plugin, a template instantiated in both (`make_shared` of the render pass)
binds to the executable's copy, and the pass then fails to recognise the
plugin's own render buffers -- measured, as an image of zeros with five AOV
bindings and no outputs.

**Not done.** The raster technique draws the
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

**OCIO as a compiler.** OpenColorIO 2.5.2 is built by
`scripts/build-ocio.sh` into `~/tools/ocio-2.5.2` on both machines (Ubuntu
ships 2.1, which has no ACES 2.0), its own dependencies linked into it
statically so none meets OpenUSD's. `DisplayTransform::setOcio` takes a
config (OCIO's built-in studio config by default), a display and a view,
from the engine's linear Rec.709 (`lin_rec709_scene`); the processor's GPU
shader is generated as HLSL, which Slang reads as it is, and wrapped in a
generated module named by its hash that imports the display kernel's
pieces -- the pixel under the output pixel, the premultiplied colour over
the background, exposure, and the non-colour modes -- so `ViewTransform::Ocio`
is the same kernel with OCIO's function where the view transform was. Two
rewrites of the text, both about the target and not the maths: 1D LUTs are
asked for as 2D textures (`setAllowTexture1D(false)`), and `Sample` becomes
`SampleLevel(..., 0)`, since a compute kernel has no derivatives and the
tables have one level. Uniforms (dynamic properties) are bound by the names
OCIO gives them; array uniforms are refused when the view compiles.

**The exception, as the plan wrote it.** The LUT values are OCIO's,
computed on the host when the processor is built -- the one piece of
arithmetic on colour this route does not do on the device. They are
uploaded as OCIO lays them out and placed into `RGBA32Float` textures by a
kernel (`lrt_ocio_fill`), so the host does not even re-lay them. Every pixel
is a kernel's. OCIO reports failure by throwing; `setOcio` is where those
exceptions stop and become a `Result`.

**Checked against aces2.slang.** The studio config's "ACES 2.0 - SDR 100
nits (Rec.709)" view on its sRGB display is OCIO's own implementation of the
output transform -- its fixed functions, its hue tables, its Rec.709 to
ACES2065-1 matrix -- and aces2.slang is a port of the reference CTL: two
implementations, each run on the device over the same sixteen stops of hues
and coverages at three exposures. 0 of 16448 pixels differ by more than
1/255 at any exposure; the worst difference is 3.2e-4 at 0 stops, 5.5e-4 at
+2.5 and 2.2e-5 at -3 (the test bounds it at 1e-3). The control, the
config's un-tone-mapped view, differs at every pixel (worst 4.9). The same
on the L4, under Vulkan and under CUDA alike: 0 pixels past 1/255, worst
3.2e-4, 5.5e-4 and 2.2e-5. `lrt view
--ocio-display D --ocio-view V [--ocio-config C]` starts on the OCIO view,
and the panel lists it beside the others (Kitchen_set, 800x450: draw 17.5
ms median, as AgX's 17.3).

**What OCIO does not settle.** `renderingColorSpace` is still read and
reported, not acted on: the engine renders in linear Rec.709 and OCIO is
told so. For a studio config that is not ACES there is no second
implementation to check against, only that the kernel runs what OCIO wrote.

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
being measured; Metal is back to its 451 assertions exactly.

**What every kernel that writes a texture does about it.**
`Caps::convertingStores` says whether a float4 store arrives as the texture's
format, false on CUDA, and the probe checks the capability tells the truth
either way -- on CUDA that the texel is *not* the colour. Where it is false
the kernels pack each texel into a buffer instead (`lrtPackTexel` in
packing.slang: four 8-bit unorms in a word, four halves in two, four floats
in four) and the buffer is copied into the texture with
`copyBufferToTexture`, which asks neither backend to reinterpret anything.
The decode does it for level 0 and the mip generator for every level; the
textures keep the formats they have on Metal. That is the fix the section
above said was left: 8-bit images, their mips and their sRGB views are right
on CUDA now, and the PNG, UDIM and Hydra tests that shade through them pass.

**`RayDesc` is not a type on every target.** The compute BVH route shares its
ray with the hardware route, and `RayDesc` exists only where the target has
ray tracing: on CUDA without OptiX nvrtc answered "identifier RayDesc is
undefined" and the whole route failed to compile, which is why the splat ray
tracer's six cases and two Hydra cases failed here. `rt_integrate.slang`
carries its own `LrtRay` (the same four fields) and the hardware route fills
a `RayDesc` from it at the trace. The compute route now runs on CUDA: its
images match the GPU reference (p99 0, max 1 of 47500), and the comparisons
against the rasteriser skip as the device has none.

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

### The three devices, at the end of this pass (commit 2e6e183)

Parity here means the same GPU-computed metric on each machine and each
backend, computed there; no pixels travel between them.

| | Metal (Apple M5 Pro) | Vulkan (NVIDIA L4) | CUDA (NVIDIA L4, OptiX 9, no inline rays) |
|---|---|---|---|
| Cases | 188 | 188 | 188 |
| Passed | 188 | 180 | 96 |
| Failed | 0 | 0 | 1 |
| Skipped, with reason | 0 | 8 | 91 |

**Vulkan's eight skips** are the host's and the hardware's, not the
renderer's: the Metal-only tracked buffer, `lrt view`'s window on a headless
box, the five gpe-backed cases and the aofx host (gpe has no Vulkan backend;
CUDA and Metal cover them), and openFXplayer's own bundles, which are not
built there. Everything else the suite checks -- the path tracer, materials,
lights, motion, skinning, subdivision, curves, volumes, render settings,
Kitchen_set against Storm through EGL, OCIO and the denoiser -- passes on the
L4 exactly as on Metal.

**CUDA's 91 skips** are what a device without rasterisation and without
OptiX cannot answer: 47 "no rasterisation on this device", the rest needing
rasterisation and ray queries together (motion, volumes, the lens, light
linking, the Storm oracle, the host's engine), plus points as discs, the
window, the Metal buffer and openFXplayer's bundles. Each names its
capability in its skip message.

**CUDA's one failure** is `open_pbr_surface` against MaterialX's genglsl
closures, which "Open on this backend" describes. The run before this one
had a second: the layers case composites rasterised passes and said "the
render pass drew nothing" instead of skipping; it names the capability now.
Measured with `LRT_BACKEND=vulkan|cuda scripts/remote-test.sh` at commit
2bf4589, and `ctest --preset macos-arm64-debug` on the Mac.

### Vulkan on the L4

With NVIDIA's Vulkan driver (`libnvidia-gl-580-server`) installed,
`LRT_BACKEND=vulkan` runs the suite on the L4 with rasterisation and ray
queries, the column most of CUDA's skips move to. The first run left four
failures and a handful of skips; each was a real difference between the
backends, fixed where it lives rather than excused.

- **A dispatch that runs too long loses the device.** The lights' chi-square
  consistency check walked every sample in one invocation, which Metal's
  watchdog tolerates and NVIDIA's Vulkan driver answers with
  `VK_ERROR_DEVICE_LOST`. It is a parallel kernel now (`lightConsistent`
  writes a flag and a gap per sample, `lightConsistentReduce` counts them), so
  no invocation is long on any backend.
- **A sphere light's `lightHit` missed at the tangent.** Deciding a hit by the
  sign of the ray-sphere discriminant is at the mercy of FMA: a direction
  `sampleLight` drew inside the cone came back a miss on Vulkan, where the
  compiler contracts differently. The hit is decided by the cone itself --
  `dot(wi, toCentre) >= cosMax`, the same test the pdf uses -- and the
  distance takes a clamped discriminant, so sampling and `lightHit` agree by
  construction.
- **An 8-bit mip level drifted by one.** Writing a float into a `UNORM8`
  texture rounds as the implementation pleases, and Vulkan's differs from
  Metal's at the half. The mip kernel quantises itself for 8-bit formats,
  `(round(saturate(v) * 255) + 0.25) / 255`, so the store has nothing to round.
- **A fixture drew its random numbers in argument order.** GCC evaluates
  function arguments right to left and Clang left to right, so the EWA
  cloud fixture built a different cloud on Linux. Draws go into named
  locals first (`SplatFixtures.h`, `test_lod.cpp`).
- **The free-running clock missed its deadline by the sleep's overrun.**
  `FrameClock` spins the last part of a wait; the margin is learnt now
  (`spinNs_`, from 0.2 ms up to 10 ms as overruns are seen) instead of fixed.
- **Storm needs OpenGL, and a headless box has no window to get it from.**
  `platform::makeHeadlessGlContextCurrent()` opens an EGL display on the GPU
  itself (`EGL_EXT_device_enumeration` through glvnd's dispatch, since the
  prototypes resolve only by `eglGetProcAddress`), a 1x1 pbuffer and a
  GL 4.5 *compatibility* context -- HgiGL issues calls a core profile rejects
  as an invalid enum. The Storm oracle runs before `CreatePlatformDefaultHgi`
  with it, and Kitchen_set against Storm passes on the L4.
- **The denoiser runs on Vulkan through OIDN's CUDA device.** OIDN has no
  Vulkan device, but its CUDA device imports external memory: the staging
  buffers are created `BufferUsage::Shared`, slang-rhi exports their memory
  as an opaque file descriptor, and `oidnNewSharedBufferFromFD` imports a
  duplicate of it (OIDN takes ownership of the one it is given). The copies
  in and out are the Metal staging path's kernels. Measured on the L4, 16
  paths against 4096: relMSE 3.65e-4 noisy, 7.16e-5 denoised with albedo
  and normal, 6.29e-5 without -- the same shape as on Metal -- and the
  `lrt:denoise` render setting changes 8748 of 27648 words at the total and
  none before it.

### The engine as an MCP server

`lrt-mcp` speaks JSON-RPC 2.0 over stdin and stdout, and `modules/mcp` holds
the protocol and the tools. The shape follows openFXplayer's server, which
this project's aofx host already speaks to: the protocol is the only thing
`Server` knows about I/O -- one message in, one reply out -- so the transport
is the app's, and every tool lives in a table whose entries carry their own
schema, which is what keeps the listing and the dispatch from disagreeing
about what exists.

**Why a server rather than a shell.** The device and the stage stay open
between calls. A CLI shelled out to once a frame opens the GPU, compiles the
kernels the frame needs and throws the lot away; here the second render of a
stage costs what a second render should cost. That is the whole reason to
drive the engine from outside.

**A render answers with the image.** The display transform runs on the device
as it does for `lrt view`, the eight-bit result is read back and packed into a
PNG (`io::writePng`, zlib's deflate and a CRC -- a container, not a codec: the
pixels were decided by a kernel), and the PNG goes back as an MCP image block
beside the timing. A model that cannot see what it rendered is guessing.

The tools: `open_stage`, `stage_tree`, `device_info`, `render` (camera of its
own or the stage's, any AOV, either technique, every path tracing and
lighting setting, an EXR beside the image), `pick`, `bounds`,
`render_products`, `convert` (a capture into a stage, on the device the
session already has), `timings` (the median of several frames, as the CLI
measures), `settings`. `.mcp.json` registers it as `lrt`.

**Driven over real stages**, not only the splat captures this engine started
from: the OpenChessSet path traced at 640x360 (64 paths a pixel, one bounce,
ACES 2.0, 25.7 s on the M5 Pro) with its MaterialX materials, its sun and its
dome; Pixar's Kitchen_set path traced from a camera of the caller's own (32
paths, 6.5 s) with its instancing and its textures, where `pick` answers
`/Kitchen_set/Arch_grp/Kitchen_1/Geom/TileFloor/pPlane357` for a pixel of the
floor and `bounds` gives the room's extent. The tools are the engine's, so
what a stage holds -- meshes, materials, lights, curves, volumes, splats --
is what they can show.

**Checked** at the protocol's level (`lrt_mcp_tests`): the handshake answers
in the client's version and in ours where the client asks for one nobody
implements; a notification is answered with silence, since the first message
a client sends is one and answering it is a violation; the listing carries a
schema for every tool; an unknown tool is a result with `isError` while an
unknown *method* is a JSON-RPC error; a message that is not JSON is -32700.
With a GPU: a stage written by the test opens, renders at 96x64, and the
answer carries a PNG (checked by its signature through the base64) and a pick
that names `/Square`.

## A note on WebGPU

Everything a frame does is Slang, and Slang emits WGSL, so the shading,
the materials, the lights and the compute BVH route would port. What would
not, without work: there are no ray queries, so the path tracer would trace
over `BvhScene` alone; buffers bind through bind groups with a small limit
per stage and no bindless textures, which the texture table and the
kernels at Metal's 31-buffer limit would both have to be reshaped for; no
fp64, no OIDN, no OpenUSD in a browser -- so the delegate stays native and
a WebGPU build would take the engine's records, not Hydra's prims.

### Not done

- **open_pbr_surface on CUDA: found, and it was the read-only cache.** Of the
  twelve materials the lobe library is checked against MaterialX's genglsl
  closures with, eleven agreed here to 2e-5 and `open_pbr_surface` did not:
  every component differed, ours 1.339 where genglsl's was 0.054, with the
  same source agreeing on Metal and on Vulkan. It was the same defect as the
  furnace's, written up below: the material read its own parameters through
  `__ldg`. The tell was in the line the test prints -- **six lobes where
  there are four** -- so a stale read had changed the lobe stack itself, not
  just a value. With the prelude fixed it builds four lobes and agrees to
  9.39e-06; with `LRT_CUDA_LDG=1` it builds six again and fails exactly as it
  always did.
- A release build and any timing beyond the sort's own.
- gpe on Vulkan: gpe has no Vulkan backend, so its tests and aofx's host
  run on CUDA and Metal only.
- **Ray tracing on CUDA works now, as a pipeline.** Three things were in the
  way, and none of them was the SDK.
  - **Slang has no `RayQuery` for the CUDA target at all**, not even inside a
    ray generation program ("uses features that are not available in
    '_raygen' stage for 'cuda'"). A ray there is the classic kind:
    `TraceRay` with a payload, a miss and a closest hit. `visibility_trace`
    keeps the inline traversal and the compute entry, `visibility_trace_rays`
    holds the pipeline's three programs, and `visibility_trace_common` what
    both use -- three modules because **a module compiles every entry point
    it holds**: with the closest hit beside the compute kernel, Metal could
    no longer load the module at all and the whole visibility pass went with
    it.
  - **nvrtc wants `<optix.h>` at run time.** Slang's CUDA path compiles
    kernels as the frame asks for them, and a kernel that traces includes the
    OptiX header; Slang looks for an installed SDK (`NVIDIA-OptiX-SDK-*`) and
    the box has none -- only the headers slang-rhi fetches for itself. The
    build now tells the device where those are (`LRT_OPTIX_INCLUDE_DIR`, and
    `LRT_OPTIX_INCLUDE` overrides it) and the device hands nvrtc the include
    path through Slang's downstream arguments.
  - **`RayTracingScene` asked for inline rays** before it would build
    anything. The structure is the same either way.
  - **Checked** by a test that needs no rasteriser, so it runs on every
    device: the ray route's ids against the compute BVH's, over three turned
    squares at three depths. 0 of 43621 pixels differ on Metal, on Vulkan and
    on CUDA, where it reports "ray route: a ray tracing pipeline".
  - **The shadow query is ported too** (`rt_shadow_rays.slang`): the
    candidate loop becomes an **any hit** program that evaluates the
    particle, multiplies `1 - alpha` into the payload and ignores the hit so
    the traversal carries on, and ends the ray with
    `AcceptHitAndEndSearch` once what is left is under the cut -- which is
    what `query.Abort()` is inline. The ring that collapses a proxy offered
    twice lives in the payload, eight entries rather than sixteen, since a
    payload is registers. The closed forms come out identical to the inline
    route on the L4: 0.06250 through four particles of opacity 0.5, 0.50000
    through one, 5 particles taken and 5 duplicates caught, the cut stopping
    the ray after three, and the control (back faces culled) missing the
    particle a ray is born inside. Two things OptiX taught while porting: a
    payload is declared and **checked** (it refused a trace using 32 values
    where 24 were configured -- `payloadBytes` is not advisory), and
    `TraceRay` needs its culling flags passed explicitly, since a template
    parameter is not where they live.
    - The tracer builds its proxies and structures there without being able
      to draw with them: `GaussianRayTracer::prepare` and `shadowScene` work
      where `render` refuses, which is the split a shadow query needs.
  - **A mesh frame renders on CUDA now**, which it never had: the engine
    takes the pipeline route for `MeshVisibility::Rays` (it asked for inline
    rays before), so Kitchen_set lit draws at 480x270 on the L4 through
    OptiX with the lights and materials it has, on a device with no
    rasteriser at all. Before this the only thing CUDA could draw was
    splats, through the compute BVH.
  - **What is still inline only**: the path tracer, the cutout pass (a
    generated kernel) and the splat integrator. The last one needs its
    k-buffer in global memory, since a payload cannot hold 256 entries.
- **What the engine's inline rays mean for CUDA.**
  The box has OptiX after all -- `lrt info` on the L4 reports `optix 90000`,
  the driver's `libnvoptix.so.1` and the headers slang-rhi fetches itself
  (`_deps/optix_8_0-src`, `8_1`, `9_0`) -- and its caps read `ray tracing
  yes (pipeline), no (ray query), yes (AS)`. What is missing is inline
  `RayQuery` in a compute kernel, which OptiX does not offer: its traversal
  lives in ray generation and hit programs, reached through a shader binding
  table. Every ray this engine traces is inline (`visibility_trace.slang`,
  the path tracer, `rt_render.slang`, `rt_shadow.slang`), so on CUDA those
  kernels have no route and their tests skip. Earlier notes here said OptiX
  was absent; that was wrong, and the skips were right for the wrong reason.
  Giving CUDA the rays back means a second route through ray tracing
  pipelines, which is a piece of work nobody has started.

## A ray tracing launch on CUDA reads memory that is no longer current

The path tracer's OptiX route was ported and its shadow and visibility routes
measured exact, and then the closed form of a furnace stopped holding: a
sphere seen from inside, every face emitting `E` and reflecting `rho`, must
read `E (1 + rho + ... + rho^N)` to float precision, because cosine sampling
of a Lambert lobe has no variance at all. On Metal it does. On CUDA the same
frame read the series of a **different** `N` -- and not always the same one,
and not always all of it.

**What it turned out to be, measured.** The test that says it is
`tests/gpu/test_uniforms.cpp`, "a launch reads the uniform it was given, not
the one before it". A kernel of one line writes a number back; the number is
changed before every launch and read three ways -- out of a `ConstantBuffer`
(how every per-frame parameter is bound), out of a `StructuredBuffer` (how
every scene resource is bound), and out of an `RWStructuredBuffer` over the
**same memory** as the read-only one. An atomic tallies the launches that ran.
Sixty-four launches, the value alternating so that reading the one before is
never off by one:

| route | compute, CUDA | ray generation, CUDA |
|---|---|---|
| `ConstantBuffer` | 0 of 64 wrong | **31 of 64 wrong**, all of them the launch before |
| `StructuredBuffer` | 0 of 64 wrong | **32 of 64 wrong** |
| `RWStructuredBuffer`, same memory | 0 of 64 wrong | 0 of 64 wrong |
| launches that ran | 64 | 64 |

The pattern of the failures is `..X.X.X.X...`, every other launch, and the
value a failing launch reads is frozen rather than lagging. Every launch runs.
So: **inside one OptiX launch, a read-only load and a writable load of the
same address return different values**, and the read-only one is serving a
line no launch invalidated. It does not matter who wrote the memory -- the
host through `Buffer::write`, or a compute kernel dispatched immediately
before, both measured, both stale. On Metal every route is exact, and the ray
generation section skips there for want of ray tracing pipelines.

**What was ruled out, each by measurement, not by reading:**

- *Our shader.* The same body is exact on Metal inline and reaches the same
  closed forms. A counter inside the kernel (a bitmask of every `path.bounces`
  a thread saw across its sixteen samples) came back with two bits set: one
  thread, one launch, two different values of one uniform.
- *Ordering of the upload.* `CUDA_LAUNCH_BLOCKING=1`; a `cuStreamSynchronize`
  after the constant pool's `cuMemcpyHtoDAsync`; another immediately before
  `optixLaunch`. A probe printing the writes, the upload and the launch in
  order shows them in the right order, to the right device address, every
  time. None of the three changed the count.
- *The address.* Forcing the parameter block to land somewhere new on every
  launch changed nothing.
- *The pipeline's stack.* `OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW` and
  `OPTIX_EXCEPTION_FLAG_TRACE_DEPTH` with validation on report nothing, and an
  explicit `optixPipelineSetStackSize` made it worse (OptiX's own default is
  documented correct for a call tree of depth one, which is ours).
- *The payload and the binding table.* `maxRayPayloadSize` 64 and 128,
  `maxRecursion` 1 and 2, and a geometry contribution multiplier of 0 instead
  of 2 all behave the same.
- *The optimiser.* `OPTIX_COMPILE_OPTIMIZATION_LEVEL_0` cannot be measured:
  compiling the path tracer's ray generation entry that way is killed for
  memory on a 16 GB box.
- *Memory errors.* `compute-sanitizer --tool memcheck` reports 0 errors -- and
  passes the furnace, which is itself a datum: it serialises the launches.

Driver 580.173.02, OptiX 9.0 (`optix 90000`), NVIDIA L4, slang-rhi pinned at
`e17f6d7`.

**What this means for the CUDA route.** A single launch is sound -- which is
why the mesh visibility comparison (0 of 43621 pixels differing against the
compute BVH) and the shadow query's closed forms were exact, and why
`Kitchen_set` path traced to a plausible image. A *sequence* of launches is
not: whatever a ray generation entry reads through a constant or read-only
buffer may be what the launch before it read. That is every per-frame
parameter and every scene resource, so an animation, an accumulating frame or
a sweep of settings can silently render the frame before. Nothing in an image
says so; only a closed form does, which is how this was found.

**The fix: the CUDA prelude does not read through that cache.** Slang's CUDA
target emits every load of a constant buffer or a read-only buffer as
`__ldg`, the load that goes through the read-only data cache -- visible by
compiling anything with `slangc -target cuda`:

```
uint _S1 = __ldg(&globalParams_0->echo_0->count_0);
uint _S9 = __ldg((&(globalParams_0->echoWords_0)[int(0)]));
```

A writable buffer is not read that way, which is exactly why it was the one
route that stayed current. So `gpu::Device` now creates the Slang global
session itself for CUDA and appends to that target's prelude

```
#undef __ldg
#define __ldg(p) (*(p))
```

after Slang's own declaration of `__ldg`, so the declaration still parses and
every call site that follows it is an ordinary load. It gives up that cache
and nothing else. nvrtc could not be told this instead: Slang keeps **one**
`DownstreamArgs` entry per downstream compiler -- a second is silently dropped,
measured by handing it a define that would have changed every answer and
seeing none change -- and that one entry is already the OptiX include path.

**After it**, on the same box: the echo test reads 0 of 64 wrong on all three
routes in a ray generation entry, where it read 31 and 32; the furnace reads
its series to `0.00e+00` relative at every bounce count, run after run; and
`open_pbr_surface`, which had been the one material of twelve that disagreed
with genglsl on CUDA and nowhere else, agrees to 9.39e-06. **The whole CUDA
suite is 197 of 197**, where it had been 97 passed, 2 failed, 94 skipped
before the ray tracing pipelines and 2 failed after them. Metal is 197 of 197
and Vulkan on the same box is 197 of 197, so for the first time the three
backends are green together.

**What it costs**, medians of `lrt stage --frames` on the L4, Kitchen_set,
twice each way so the pairs can be read against their own spread:

| | without the cache | with it (`LRT_CUDA_LDG=1`) |
|---|---|---|
| raster, 1280x720 | 20.39 ms, 21.44 ms | 22.81 ms, 21.35 ms |
| rt, 640x360, 4 paths, 2 bounces | 55.79 ms, 56.12 ms | 56.51 ms, 56.14 ms |

Nothing outside the spread of a repeat. The loads this gives up are of
parameters and pools that every thread reads alike, which the ordinary caches
hold as well.

**One trap this leaves, and it is closed.** The shader cache is keyed by
slang-rhi, which knows nothing of a prelude this process hands Slang -- so a
cache written before the fix would have been served back into a fixed build,
putting the defect quietly under it. The cache path now carries a generation
(`shaders/gen1`, and `gen1-ldg` under the escape hatch), so a changed prelude
looks elsewhere instead.

## lrt view: the path tracer's own settings in the window

Switching the viewer's Technique from Raster to Path traced was reported not
to work. What was measured, before touching anything:

- **The switch itself works.** A new test ("a renderer told another technique
  between frames draws that technique") draws raster frames and then rt
  frames on one `StageRenderer`, and the other way round, against renderers
  that drew one technique from the start: 0.00e+00 relative both ways, with
  the two techniques 7.07e-01 apart at their worst pixel as the control.
- **A viewport accumulates.** `draw` adds its paths every frame of a still
  camera -- 2 after two frames, then 3, 4, ... 12 -- whatever `pathTotal`
  says; the total only decides when the frame counts as converged (and so
  when a denoise runs).
- **Nothing failed in the window**: a session on Kitchen_set_lit logged 608
  frames and no error.
- **Kitchen_set.usd has no lights.** Path traced, it is lit by the headlight,
  which lights the first hit from the eye as the raster does: the two look
  the same by construction. `Kitchen_set_lit.usda` is the one to look at.

What the window lacked is the path tracer's settings: it left the delegate's
defaults, one path a pixel a frame and one bounce, which lights a room little
more than the raster does. The View panel now shows, under Path traced,
paths per frame, bounces (4 by default in the window), denoise, and how many
paths a pixel the frame holds; `lrt view` takes `--path-samples`,
`--path-bounces`, `--path-total` and `--denoise` as `lrt stage` does.

The report's screenshots then said the rest, on the chess set:

- **The window froze on the switch.** The first frame of a technique compiles
  its kernels for every material on the stage -- the chess set's fifteen --
  on the thread that draws the window. The first time that is long; with the
  shader cache warm, opening the chess set path traced takes 6 s in all. The
  window now puts up a notice over the last frame first ("Preparing Path
  traced: its kernels compile for this stage's materials"), and draws the
  compiling frame after it. The wait is not removed: compiling off the
  drawing thread would mean a second caller on the device.
- **Path traced still looked like the raster.** The chess set authors no
  lights, like Kitchen_set, so both were lit from the eye.
  `StageRenderer::setDefaultLights` puts a sky dome (0.6) and a sun (2.5,
  2 degrees, tilted from overhead) in the stage's **session layer**, at
  `/lrtDefaultLights`: the file is not touched, and the lights reach the
  engine through Hydra as authored ones do. `hasLights` does not count them.
  The viewer turns them on for a stage with no lights, with a checkbox to
  turn them off, and `--no-default-lights`. A test on a floor and a wall with
  no lights: lit against unlit relMse 0.366 raster and 0.504 rt, and removed
  again the frame is the unlit one to 0.00e+00 on both.
- **A frame took 1190 ms** at 1920x1018 path traced with 4 bounces; Render
  scale is what trades that for interactivity.

## The raster sees lights at infinity along its lobes: glass and polished metal

The chess set's glass pawn heads and polished rims drew black under the
raster. Its surfaces were lit by light sampling alone, and three kinds of
lobe get nothing from that: light **through** the surface (a dome is sampled
over the hemisphere above the normal), a **delta** lobe (smooth glass, a
mirror, which answers no light sample), and a **glossy** lobe too narrow for a
dome's samples to find.

`MaterialShading` now also samples the lobe stack (up to 32 samples a pixel)
and meets the lights at infinity -- domes and distant lights -- along those
samples: through the surface, only the dome, which light sampling never
reaches there; a delta lobe, weight one; a narrow lobe, weighed against light
sampling by the power heuristic. Through the surface the lobe's sample is
followed as it leaves this surface, without the second refraction a solid adds
on its far side, and the raster traces nothing behind it: a glass object in
the raster shows the sky, not the room behind it.

**What the Metal compiler allowed, measured each time.** This kernel already
carries a note: with a copy of the lobe stack live across the shadow ray's
intersector, it writes garbage. Three arrangements hit it again, each caught
by a test that must be bit exact:

| arrangement | caught by | result |
|---|---|---|
| lobe samples drawn after the light loops | a raster frame after a path traced one, against a fresh one | differed in 3 runs of 4, worst 69 times the value |
| drawn before, kept in arrays for shadow rays after | shadows on against off over an unoccluded floor | 681-1625 of 24576 words, different each run |
| the true lobe density (`stackPdf`) asked for inside a loop that traces, before or after its shadow ray | the same | 969-1424 words |

What holds: every lobe sample is drawn, looked up and summed **before the
first shadow ray**, and nothing of the stack is asked for after. That has two
consequences, both deliberate:

- **A lobe's sample traces no shadow ray.** A reflection sees the sky whether
  or not something stands in the way, as an environment map's does.
- **Both strategies are weighed by a proxy density**, not the stack's: a Phong
  lobe about the mirror direction with the stack's own peak density, taken
  once before any shadow ray. MIS weights only have to sum to one wherever
  both strategies can sample, which a density shared by both does. The proxy
  is **zero for a broad stack** (peak density 4 or less, a Phong exponent
  near 24): there light sampling keeps the whole weight and its shadow ray, so
  a diffuse floor's dome shadows are untouched. The narrow lobes the proxy
  hands to the lobe side are where the missing shadow ray shows, as a
  reflection of sky under something that should hide it.

After it: the floor 0 words in 6 runs of 6, the technique switch 0.00e+00 in
4 of 4, and the light groups and shadow link tests, which the corrupted
arrangements had also failed, pass.

Measured, in "the raster sees a dome through glass as the path tracer does":

| material, under a uniform dome | check | result |
|---|---|---|
| `dielectric_bsdf` RT, roughness 0 | every pixel reads the dome (lossless) | worst 1.66e-05, at 1 and 4 samples |
| `standard_surface`, transmission 1 | raster against rt at 4096 paths, 2 and 32 samples | relMse 1.98e-03 then 1.24e-04: **16.0** |
| `conductor_bsdf`, roughness 0.1 | the same | relMse 7.51e-02 then 4.78e-03: **15.7** |

Sixteen times the samples dividing the squared error by sixteen is what an
unbiased estimator does. A bias would have stayed where it was. The first attempt
measured 1 against `standard_surface`, which was wrong: MaterialX layers its
specular reflection over the transmission and attenuates that by one minus
the reflection's albedo, so a pane reads F + (1 - F)^2, and the error sat at
3.8% whatever the samples. That is why the lossless case is a bare dielectric.

Not done: reflections in the raster see only the sky -- not the neighbouring
pieces, and not what should hide the sky -- so the chess set's rims read
darker than under rt.

`LRT_VIEW_SWITCH_AT=N` makes `lrt view` flip its Technique selector at frame
N, as a click would, so a sequence someone reports ("opened in Raster, switched
to Path traced") runs under `--frames` and `--snapshot` instead of being
described. Run that way on the chess set, the switched frame showed the glass
heads as glass: the black heads in the report were the raster's.

## aofx at ABI 25, from aopenfx

The AOFX ABI now lives in its own repository, **github.com/jesusluque/aopenfx**,
and that is where the SDK is copied from -- `sdk/include/aofx`, verbatim, at
`73d8071` -- not openFXplayer. The ask came from `bundles openFXplayer built
load in this host` failing: openFXplayer's bundles were rebuilt against ABI 25,
and this host spoke 23.

What the two bumps are, with comments set aside (every header differed, but
eleven only in their copyright line and in dropping the names of particular
hosts):

- **24.** `EffectDesc::flowsWhen` (a node is a live source while a parameter
  says so), `RenderRequest::projectWidth`, `projectHeight` and `complaint`,
  and the verbs `Gpu::borrow` and `Gpu::importFd`, which default to an invalid
  buffer -- the contract's "use `keep`".
- **25.** `buildTag()` says which ABI of its standard library a translation
  unit was built with and what `std::string`, `std::vector`, `std::function`
  and a pointer measure. No struct moves; the tag's contents do, and the host
  compares tags, so 24 and 25 do not match.

What the host does with 24:

- **`borrow`**, where the device reads host memory in place: the pages wrapped
  as a shared-storage Metal buffer without a copy
  (`platform::newMetalBufferOverPages`: page-aligned, whole pages, refused on
  a discrete GPU), adopted by gpe's pool like any foreign buffer, given back
  by `drop`. The same key over the same pages hands back the same buffer;
  over other pages the old binding goes first. Measured: a kernel's read of a
  borrowed page gave 3, then `0xb0770` after the test wrote that into the
  page, while a `keep` of the same pages still read 3.
  `platform::pageSize`, `mapPages` and `unmapPages` join Platform for it.
- **`importFd`** answers invalid, as the reference host does: importing
  another process's exported device memory is CUDA or Vulkan external memory,
  which gpe does not reach.
- **`projectWidth`/`projectHeight`** from `EffectJob`, which gains the two
  fields; zero there means the output's bounds, since this host renders one
  image and that image is the frame.
- **`complaint`**: when `process` returns false, the effect's own sentence is
  what the error says ("'tv.mediapro.aofx.test.reporter' did not render: the
  reporter was told to fail"), and the host's own complaint from a refused
  load or run only when the effect gave none.
- **`flowsWhen`** changes nothing here: it tells a node graph that a node is a
  live source, and this host renders single jobs.

After it, `bundles openFXplayer built load in this host` loads openFXplayer's
ABI 25 bundles, `aofx_sdk_manifest` is re-recorded against the copied
headers, and the manifest's own message now names aopenfx as the reference.

## Splats shadow meshes in the path tracer

A relit cloud lit a floor it never darkened: the path tracer traces triangles,
and a splat is not one. Its shadow rays now carry the transmittance of every
cloud in the frame -- the same query the splat tracer has always had
(`rt_shadow.slang`: a particle evaluated at its peak, `1 - alpha` multiplied
in, a ring that collapses a proxy offered twice), asked at three places: a
light's next event estimate, the same inside a medium, and an emitting
triangle's.

**The tables are packed into one buffer** (`rt_shadow_packed.slang`,
`technique::SplatShadows`). The query wants four -- frames, colours, each
instance's world-to-cloud rows, its bases -- and a kernel of its own can bind
them; the path tracer cannot, because it sits at Metal's limit of 31 buffers.
So `rtShadowPack` gathers them on the device into one `float4` buffer whose
offsets travel in `PathParams`, and the two uint tables ride in float4 lanes
(`asfloat` in, `asuint` out). Two slots were still one too many, so **the
denoiser's guides became a link-constant variant too**: a frame that asks for
albedo and shading normal compiles a kernel with them, one that does not gets
the buffer back. With both asked for at once the guides win, and the engine
says so once -- a denoised frame needs them, while a cloud that shadows
nothing is a frame too bright in one place.

The engine asks for the guides only when something wants them (`denoise`, or
an albedo/shadingNormal AOV: `AovRequest::aux`), so the ordinary path traced
frame has room for the cloud.

**Measured** (`tests/technique/test_splat_shadows.cpp`):

- **The packed tables answer what the separate ones answer**: 0 of 4 rays
  differ, over a cloud of 20004 particles, with the first ray stopped almost
  entirely (0.0000) so that agreement is not two ones agreeing.
- **A cloud between a point light and a floor darkens it by what it lets
  through**: under an isotropic particle's centre a shadow ray peaks with
  power 0 and takes its opacity whole, so the frame with the cloud over the
  frame without it reads **0.5000** where the particle's opacity is 0.5, and
  **1.0000** at a pixel whose ray passes ten sigmas away. No pixel of 3185
  came out brighter with the cloud than without.

**What the dispatch cost to find.** `ComputeKernel::dispatch` takes threads
and divides by the kernel's group size; the packing passed groups, which were
divided again. A cloud of 741872 particles wants 3.7 million entries packed
and got 14592 -- the frames, and not one colour -- so every shadow ray read
opacity 0 and said the cloud was transparent, while the tests passed: a cloud
of four particles needs nine entries, and one group is 256 threads. The test
now packs twenty thousand particles, where the difference is a black floor
rather than nothing at all.

**Not done.** A cloud still does not shadow itself through this path (that is
the splat tracer's own `--splat-shadows`, unchanged), a bounce ray meets no
splats (only shadow rays do), and on a device that traces in a pipeline
(CUDA) there are no splat shadows at all: the traversal would have to be an
any hit program of its own, and the path tracer already carries two.

## mesh2splat: a mesh becomes a cloud, through the plugin interface

Electronic Arts' [mesh2splat](https://github.com/electronicarts/mesh2splat)
turns a textured mesh into 3D gaussians. It is BSD-3, the algorithm is short
and the result is exactly the primitive this engine already draws, so it was
worth having. What was decided is **where it runs**: it is an AOFX effect
(`plugins/mesh2splat`), not a module of the engine.

**Why a plugin and not a module.** The conversion is a kernel over triangles
with no state, no device to own and no scene to walk -- which is the shape an
AOFX effect has, exactly. Making it one costs a boundary and buys three
things: it runs unchanged in openFXplayer, it is the proof that this SDK
serves a *process* and not only a filter over pictures, and the engine keeps
a conversion out of its own modules, where it would have been the first thing
in `modules/` that is neither a renderer nor a loader.

**What the boundary costs.** An effect is handed pictures. A mesh is not one,
so the triangles travel as a picture of numbers -- six `float4` entries a
triangle, `(position, u)` and `(normal, v)` for each corner, in world space
(`shaders/lrt/usd/mesh_pack.slang`) -- and every map a material names travels
as rows of linear `float4` sampled out of the texture table
(`shaders/lrt/usd/texture_rows.slang`). Neither is a copy: the pictures are
allocated by the AOFX host's own image storage, which on a device with
unified memory is memory a kernel reads, and `Context::renderView` gives a
slang-rhi view of the same bytes, so the packing kernel writes where the
effect will read. What does cross back is the records, once, because
`usd::writeParticleFieldStage` takes a `io::RawSplats` -- and the numbers in
a file are the processor's business by definition.

**The port.** Their pipeline is a geometry shader and a fragment shader; this
is one compute kernel with the fragments walked (`m2sEmit`, one thread a
triangle, over the cells of its projected bounding box). Three things in it
were read out of their source rather than guessed, and getting the first two
wrong is what the first render showed:

- the Jacobian is over their **`orthogonalUvs`** -- the triplanar projection,
  normalised by the model's box -- and **not** over the texture coordinates;
- the scale is `|Ju| * sigma / resolution`, the `/ resolution` being applied
  by their *exporter* (`SceneManager.cpp`: `scaleMultiplier = gaussianStd /
  resolutionTarget`) and not by any shader. Without it every gaussian is the
  size of the whole model, which renders as a heap of slabs;
- the frame is the triangle's **longest edge**, its normal and the cross of
  the two -- not the Jacobian's columns, which is what the sizes are measured
  along. That inconsistency is theirs and it is kept: the gaussian is a disc
  in the triangle's plane however that plane was parametrised.

**What is ours, and not theirs: transmission.** Their fragment shader samples
albedo, normal and metallic-roughness. A `standard_surface` with
`transmission = 1` -- the chess set's glass -- has no channel there at all, so
it would come out as an opaque white gaussian. Here the material's
`transmission` and `transmission_color` are read from the stage and the
gaussian keeps `lerp(1, minOpacity, transmission)` of its opacity and takes
that colour. `minOpacity` is 0.25 rather than zero because `1 - transmission`
is nothing at all for glass, and a gaussian with no opacity is not a
translucent gaussian but an absent one -- `splatExport` drops it. So glass
converts to **a tint, not a lens**: what stands behind it is dimmed rather
than refracted, and that is the honest limit of the primitive.

**Reading a stage without Hydra.** `usd::MeshStage` is new, and it is the
first thing in this repository to walk `UsdShade` itself: every other route
into the engine goes through Hydra, which hands over a network. It reads the
meshes, their world transforms and, from the bound material's surface, the
`standard_surface` or `UsdPreviewSurface` inputs a gaussian can carry --
following connections through node graphs, interface inputs and a normal-map
node to whichever image node finally produces them. A value that is *computed*
rather than authored (a mix, a noise) has no answer without shading a point,
so the default stands.

**Measured** (`tests/aofx/test_mesh2splat.cpp`, the inputs written by a kernel
and the answers counted by one):

- **A unit quad at resolution 16** gives between 256 and 272 gaussians -- one
  a cell whose centre it covers, the slack being the cells the diagonal runs
  through, which both triangles claim. None off the quad's plane, none off
  its extent.
- **Every one is `sigma / resolution` wide** on both axes, `1e-7` on the
  third, and its frame's short axis is the quad's normal: 0 violations of
  each over every gaussian.
- **Colour**: against a checker, a gaussian well inside a cell takes that
  cell's colour, through the texture coordinate its record carries. 0 wrong.
- **Glass**: `transmission = 1` with `minOpacity = 0.25` leaves every gaussian
  at a quarter of its opacity and the transmission colour, exactly.
- **The chess pawn**, end to end: `lrt mesh2splat Pawn.usd` reads the two
  meshes with their bound materials (`M_Pawn_Body_B` with its black marble
  maps, `M_Pawn_Top_B` with `transmission = 1`), writes 729073 gaussians at
  resolution 512 in about two and a half seconds of a debug build, and the
  stage path traces with the viewer's default lights into a pawn whose head is
  green glass.

**Not done.** The PBR channels the conversion can write (shading normal,
metallic, roughness, the texture coordinate) have nowhere to live in a
`ParticleField3DGaussianSplat` yet, so `lrt mesh2splat` asks for the four-entry
record and leaves them out; they arrive when the per-gaussian PBR carrier
does. A mesh whose `GeomSubset`s bind different materials is converted as one
material, the one bound to the mesh. Nothing here is built or checked on
Linux yet, so CUDA is unverified.

## A texture's colour space, which a material network drops

Two renders of the same chess pawn disagreed: the mesh showed pale grey marble
where the cloud converted from it showed the black marble the asset is made of.
The cloud was right.

**What was happening.** `HdMaterialNode2::parameters` is a map of names to
values and nothing else. USD carries a texture's colour space beside the value
-- as `colorSpace` metadata on `inputs:file`, or as a UsdUVTexture's
`sourceColorSpace`, which usdImaging folds into the same place -- and all of it
is gone by the time a delegate reads `GetMaterialResource()`. So hdMtlx wrote a
document whose file inputs said nothing, `MaterialCompiler` read every texture
**raw**, and an 8-bit sRGB base colour was taken for linear: 0.2 became 0.2
where it should have been 0.033. Dark textures came out pale and washed; light
ones came out flat. Every asset with an sRGB base colour was affected, which is
every asset.

**Where it survives**, and where a Hydra 2.0 delegate should have been reading
it: the scene index. `HdMaterialNodeParameterSchema` carries `colorSpace`
beside the value, and `UsdImagingDataSourceAttributeColorSpace` is what fills
it -- including the `sourceColorSpace` consolidation. So `HdLrtMaterial::Sync`
reads the network for its values as before and the terminal scene index for
this one thing, and writes it onto the MaterialX input the document ended up
with (`applyColourSpaces`, matching nodes by `HdMtlxCreateNameFromPath`).

**And a second half, in the compiler.** A MaterialX shader port does not carry
the colour space its document input had: MaterialX puts one there for a colour
management system to act on, and this generator registers none -- the decode is
`TextureStore`'s, on the device, where the texture already is. So
`compileDocument` now reads the document itself before generating (every
`filename` input's `getActiveColorSpace()`, which falls back to the document's
own), keyed by the path, and a file input whose port says nothing takes its
answer from there. `auto` is passed through as a colour space of its own, which
is not MaterialX's: it means the file decides, which is what USD's
`sourceColorSpace = auto` says and what `ColourSpace::Auto` already did.

**Measured** (`tests/usd/test_usd.cpp`, "a file's colour space reaches the
decode"): one 8-bit picture read twice through a MaterialX `image` node. With
`colorSpace = "srgb_texture"` every pixel of the square shades to
(0.6039, 0.3184, 0.0332), the texture's (0.8, 0.6, 0.2) through the sRGB curve
(0.6038, 0.3185, 0.0331) -- and every pixel equals its centre exactly, so it is
one decode and not a gradient. With `lin_rec709` it shades to the values as
held, 0 mismatches at the check's own 1e-5.

**A converted cloud is relit.** The other half of the pawns' difference was
not a defect: a cloud out of `lrt mesh2splat` carries an albedo, not radiance
somebody captured, so the export now writes `primvars:lrt:splat:relight = 1`
(`ExportOptions::relight`, `lrt mesh2splat --baked` to turn it off) and the
scene's lights light it. What is left after that is the representation itself:
splat relighting is one diffuse sample per light with a normal a splat never
had, so a dark glossy marble reads darker as gaussians than as a surface with
a specular lobe. That is what the per-gaussian PBR carrier is for.

## A relit splat: both routes, and what it reflects with

Two renders of the converted pawn disagreed again, and this time the cloud was
the wrong one: black where the mesh was polished marble. Two separate defects
and one missing piece.

**The flag meant nothing when a frame was traced.** `LrtSplatLightingAPI` was
read in `splat_project.slang`, the tile rasteriser's, and nowhere else. A
cloud drawn by the ray tracer -- which is what a stage of nothing but splats
gets under `lrt:technique = rt` -- showed exactly what it was baked with. For a
capture that is right by luck; for a cloud converted from a mesh, whose colours
are an albedo, it is the albedo itself with no light on it: measured, a white
material came back at **1.0** and the chess set's black marble at **0.003**.
The relight now lives in `shaders/lrt/splat/splat_relight.slang` and both
routes call it, so the two cannot drift again. The traced route needed the
light table as well, which only a mesh layer used to build
(`Engine::prepareSplatLights`).

**It was relit in the wrong space.** The rasteriser sampled world-space lights
with a splat's position in its *cloud's* space. Every stage this had been
tried on had the cloud at the origin, so nothing showed. Both routes now take
the cloud to the world first, by rows the host passes, which is what
`splat_shadow.slang` already did.

**And it was Lambert.** A splat had a diffuse lobe and nothing else, which is
why the marble was black: at `0.003` of albedo, everything you see of that
material in a render of the mesh is **reflection**. So a splat now has a
specular lobe -- GGX, Smith's height-correlated visibility, Schlick's Fresnel,
the shape the surface lobes use -- and carries what it reflects with:
`GpuSplats::pbr`, one word a splat, metallic and roughness a byte each. The
conversion writes them (`primvars:lrt:splat:metallic` and `:roughness` on the
ParticleField prim), `CloudLoader` packs them, and a cloud without them -- a
capture, every file a trainer writes -- relights as it did, with metallic 0
and roughness 1.

**A dome is answered whole rather than sampled.** Every other light is
somewhere: one sample at its centre is a direction worth taking. A dome is
everywhere, and one sample of it gives a splat a spike where a surface shows a
broad sheen. So the dome's contribution is integrated in closed form: uniform
radiance over the hemisphere is `albedo * L` of diffuse and
`environmentBrdf * L` along the mirror direction of specular, with Lazarov's
analytic fit of the split-sum term.

**Measured.**

- `tests/render/test_splat_render.cpp`, "both routes relight a splat, and
  alike": the traced frame changes with the flag (max 178, 9216 pixels past
  2), and the relit pair agrees between the routes exactly as closely as the
  baked pair does (p99 1, max 1 for both) -- so relighting does not widen the
  difference the two renderers always have.
- The pawn, path traced with the viewer's default lights, over a 60 by 60
  patch of its body: the mesh averages **0.085, 0.099, 0.097** and the cloud
  **0.044, 0.046, 0.046**. Before this it was **0.0036** against the mesh's
  floor of 0.043, and black on the screen.

**What is left, and why.** The cloud is about half as bright as the mesh over
that patch. A splat gets one sample a light, no bounce, and a normal it never
had; the mesh gets a path tracer. That is the representation, not a defect,
and it is the reason the conversion carries metallic and roughness at all --
without them the difference was a factor of ten and a black pawn.

**One more thing the conversion got wrong**, found on the way: a material that
names a roughness map and no metallic map had its roughness written into all
four channels of the packed map, so metallic came out equal to roughness --
the chess set's glass, whose only map is a roughness. The defaults are laid
down first now, and each file writes only its own channel.

## What a converted surface is worth, measured against the mesh it came from

The relit pawn was still half the mesh's brightness, so it was measured
properly rather than argued about: a unit quad, one material (base colour 0.5,
roughness 0.3, metallic 0), one distant light of intensity 1 head on, rendered
as a mesh and as the cloud converted from it. The mesh reads **0.5032**. The
cloud read **0.0358**. Four things were wrong, each found by taking the next
number apart.

**The colour was put through the sRGB curve twice.** A cloud is blended in the
space it was trained in and the finished pixel is linearised
(`frame.slang`); relighting produces light, which was then linearised again.
0.5 came out 0.214, and where the curve is steepest a dark material came out
ten times too dark -- that is what made the pawn black. A relit colour now
goes into the blend encoded (`relitForBlend`), and the blend's own linearise
gives back exactly the light that was computed. The same applies the other
way: the **albedo** a relit splat uses is the stored colour decoded, which no
relight had been doing -- a capture's trained colour is sRGB too.

**A conversion's colours are light, and a cloud's are not.** So
`io::SplatEncoding::Colour::LinearLight` says which, and the export encodes
them into the cloud's space on the way in. A baked converted quad now reads
0.3508 against a coverage of 0.7016: 0.5 exactly.

**The traced route never set `linearise`.** The shade pass writes the colours
the rays read and did not know which space it was writing into.

**And the conversion never passed the material's own metallic and
roughness.** mesh2splat's shader defaults to (0.1, 0.5) and reads the rest
from a map; a material that names no map got that plastic, whatever it
authored. They are parameters of the effect now, multiplied into the map where
there is one, as glTF multiplies its factors.

**With all four, the quad reads 0.5009 against the mesh's 0.5032** -- 0.5%,
which is what is left of MIS weighting the surface's own specular sample.

## Translucency: what a gaussian can do about glass

A gaussian cannot refract. What it can do is three things, and with them glass
reads as glass:

- **Let what is behind it through**, which is its opacity: a transmitting
  material keeps `lerp(1, minOpacity, transmission)` of it (0.15 by default),
  so the collar is visible through the pawn's head.
- **Reflect**, which is the specular lobe, and which is most of what a real
  glass surface shows.
- **Send on the light that arrives from behind it.** This is the translucent
  half and it is new: the body of the material is split, `1 - transmission` of
  it facing the light and `transmission` of it facing away, so a light behind
  a glass splat lights it instead of leaving it black. Under a dome, which is
  on both sides at once, the two halves sum to the whole body -- a glass ball
  under a sky is its own colour, not a dark shell. `transmission` travels per
  splat, in the third byte of the packed word beside metallic and roughness.

Killing the diffuse outright was tried first and is wrong: the pawn's head
went grey, because what makes it mint green is exactly the light that goes in
and comes back out.

## The width of a converted gaussian, and what it covers

mesh2splat's `sigma` is 0.65 of a cell. Traced here, that leaves a converted
surface **30% transparent**: the ray meets a gaussian 0.7 of a cell from its
centre and takes what the falloff leaves. Measured on the quad, the coverage
(and the pixel, against the mesh's 0.5032):

| sigma | coverage | pixel |
|---|---|---|
| 0.65 | 0.702 | 0.351 |
| 0.85 | 0.848 | 0.425 |
| 1.0  | 0.911 | 0.456 |
| 1.2  | 0.957 | 0.479 |
| 1.5  | 0.986 | 0.493 |

So `lrt mesh2splat` defaults to **1.0** and `--sigma 0.65` asks for theirs.
It is a departure from their number and it is deliberate: their renderer
composites its own way, and a surface you can see 30% through is not what a
conversion of a solid mesh means. The rasteriser is less affected than the ray
tracer (0.94 against 0.70 at 0.65), which is the two integrations differing,
not the cloud.

## Path traced into gaussians: the bake

A relit cloud is an approximation and says so -- one sample a light, no
bounce, a normal a splat never had. The conversion can do better, because the
scene it converts is the scene the path tracer already knows: `lrt mesh2splat`
now **bakes**, and what it bakes is the path tracer's own answer at every
gaussian.

**How little it took, and why that matters.** A bake is a frame whose camera
is a list of rays. So it is not a second integrator -- the thing this
repository would least want -- but a *variant of the path tracer's kernel*
where the first hit comes from a buffer instead of from a camera
(`kBake`/`foundBaked`), and everything after that first vertex is the frame's
own path: the same lights, the same shadows, the same bounces, the same
materials. The engine reaches it through `Engine::bakePoints`, which is a
render with a bake request in it, so the scene preparation is shared by
construction rather than by copy.

**Which direction is baked.** One colour cannot be view-dependent, so the
bake stores the **cosine-weighted average over the hemisphere the surface
faces** -- the radiance a diffuse surface of the same radiosity would have.
The direction varies per sample, so the mean over the paths is the mean over
the hemisphere.

Two things were measured on the way:

- **Along the normal was wrong.** It was tried first, and under a dome it
  makes every gaussian show the same reflection: the chess set's marble came
  out as polished plastic, 0.21 against the mesh's 0.085, with its texture
  gone.
- **The directions must be stratified, not drawn.** The radiance leaving a
  glossy surface swings by orders of magnitude across the hemisphere, so
  random directions leave one gaussian in the mirror of the sun and its
  neighbour nowhere near it. That was salt and pepper over the whole model,
  and **four times the paths barely touched it** -- 64 against 256 looked the
  same, which is how it was diagnosed as not being noise. A grid with a jitter
  in each cell took it out at 64.
- **A ray starts along the direction it is seen from**, not along the normal:
  one that starts above the point and travels sideways misses its own surface
  at grazing angles, and the gaussians it misses come back black.

**And two defects it turned up in the tracer:**

- `setPrograms` did not know about the new variant, so it returned early and
  the frame ran the kernel compiled without it: the bake's first hits came
  from a camera of one pixel, and **exactly one gaussian of 729073** came back
  with anything in it.
- A bake has no camera, so it can have no headlight. Without that, a stage
  with no lights of its own bakes in a lamp standing whereever the frame's
  one-pixel camera happened to be.

**Where it stands.** The pawn bakes in 25 seconds at 64 paths a gaussian
(debug, 729073 gaussians) and the body's mean lands at 0.099/0.097/0.085
against the mesh's 0.085/0.099/0.098 -- the same light. What it does not
reproduce is the *look* from a camera: a glossy surface reads as uniformly
shiny, because the average over the hemisphere has the specular everywhere
while a view has it in one place. That is the limit of a colour with no
direction in it, and what answers it is spherical harmonics -- which the
clouds already carry to degree 3 and the bake does not write yet.

## What one colour a gaussian cannot do, measured

The baked pawn had the right light in it and still did not look like the mesh:
its marble came out smooth. Three arrangements were tried and measured over
the same 60 by 60 patch of the body, against the mesh's **min 0.024, mean
0.085, max 5.55**:

| what the gaussian carries | min | mean |
|---|---|---|
| the whole material, baked | 0.066 | 0.137 |
| its body baked, the polish added by the frame | 0.071 | 0.110 |
| the material, relit every frame | 0.004 | 0.044 |

The means can be made to match. The **minimum cannot**: every bake floors at
0.066 where the mesh reaches 0.024. That is not a bug to find -- it is what a
colour with no direction in it means. A surface looks dark from the directions
where it reflects nothing bright, and an average over the hemisphere has none
of that: it puts the sky's reflection on every gaussian from every side, which
lifts the darks and, with them, buries a texture whose contrast is smaller
than the lift.

So the split that was built -- the body baked (`bakeBody`: the diffuse, what
the material transmits, and a conductor's reflection, which is all a metal
has) and the polish added at render time from the metallic and roughness the
gaussian carries (`primvars:lrt:splat:litBody`) -- is worth having and is not
enough. It gets the texture's own light right, with its shadows and its
bounces, and leaves the reflection to a lobe that knows where the eye is. What
it does not have is the dome's *visibility* per gaussian, so the added
reflection lifts exactly the places the mesh leaves dark.

**What ends it is spherical harmonics**, which is what a trained cloud carries
and what this engine already reads to degree 3: the bake would run once per
coefficient, projecting the radiance it already samples over the hemisphere
onto the basis, and the renderer would need nothing added at all. Apple's
LiTo calls the same thing a surface light field and learns a latent for it;
the classical version is four to sixteen numbers a gaussian, and the format
has room for them.

## The bake fits harmonics: what a colour could not hold

A gaussian carries one colour and a reflection is a function of direction, so
the bake now fits **spherical harmonics** -- the same ones a trained cloud
carries and this engine already reads to degree 3. Apple's LiTo calls the
thing they stand for a surface light field; the classical version is four to
sixteen numbers a gaussian, and the format has room for them.

**How it is fitted.** The path tracer's bake already looks at every gaussian
from many directions; each sample is now weighed by a basis function where it
looked from (`shBasisValue`, beside `evaluateRest` so the two cannot drift)
and the coefficients come out of one pass, a plane each. Three things had to
be got right, and each was got wrong first:

- **Over the sphere, not the hemisphere.** Harmonics are orthonormal over the
  sphere and over nothing else. Fitted a coefficient at a time over the
  hemisphere a surface faces, each one explains the same light again and their
  sum overshoots: **ten times too bright**, measured. The integral is over the
  sphere with the far half taken as nothing -- which is what a one-sided
  surface sends there -- and drawn from the near half with a measure of 2 pi,
  so no sample is thrown away.
- **One pass, not one a coefficient.** The paths are the same for every
  coefficient and only the weight differs. Tracing them once and weighing them
  sixteen ways took the pawn from **5m43 to 32 seconds** at degree 2.
- **Which surface and from where are two questions.** The ray that finds the
  surface goes straight down the normal, which always meets the point it was
  built from; the direction the sample looks from is set afterwards, on the
  hit. Aiming the ray along that direction instead leaves it travelling beside
  the surface at grazing angles: **55000 gaussians of 729073 found nothing**,
  came back black, and speckled the model in a way no number of paths touched.

**What is baked and what is not.** The body of the material -- its diffuse, what
it transmits, a conductor's reflection -- and not its polish. Baking the polish
was tried: a reflection off a surface of roughness 0.1 is far too sharp for
sixteen coefficients, and the pawn came back **silver, 0.276 where the mesh
reads 0.085**. Adding the polish back at render time was tried too, and the
frame's own reflection has no occlusion in it: **0.135**, and the marble washed
out again. So a cloud with harmonics carries the body and nothing is added; a
cloud baked to a single colour keeps `litBody` and the frame puts the polish
back, which is the best a colour can do.

**Where it lands.** The pawn, 729073 gaussians, 1024 paths each at degree 3:
**6m45 in release**, and over the same patch of its body the cloud reads
**0.095 / 0.093 / 0.081** against the mesh's **0.085 / 0.099 / 0.098**. The
marble is marble again: its texture, its shading and its tone, from a cloud.

**What is still wrong.** The glass head. Nearly all of what it shows is a
mirror reflection of the sky and a refraction of what stands behind, and
neither is a low-frequency function of direction: degree 3 cannot hold the
first, and the second arrives through a lobe that bends every sample somewhere
else. It comes back mottled. What would answer it is either far more
coefficients than a cloud carries or the thing LiTo went after -- a learned
latent instead of a basis -- and neither is this conversion's business today.

## A traced frame of splats alone did not finish

A cloud drawn by the ray tracer -- what a stage of nothing but splats gets
under `rt` -- returned from `Engine::render` as soon as it had drawn, before
the two things every other route ends with: the sky behind the frame
(`paintDomes`) and the camera's exposure (`applyExposure`). So the same cloud
came back over the dome's grey when rasterised and over nothing when traced,
and an exposure the camera asked for was applied to one and not the other.

It was found by looking at two renders side by side and noticing the
backgrounds did not match -- which is the only way a difference like this is
ever found, and the reason the comparison was being made at all. Both routes
now end the same way; the pawn's sky reads 0.600098 in each.

## The bake's ray started a millimetre off the model

The pawn's head has a mint glass ball standing on a ring of gold. Converted
and baked, the ball came back dark and mottled and the ring came back **grey**
-- and the same conversion **without** the bake, the albedo carried and relit
every frame, came back with both right. So the conversion's colours were not
the problem; something the bake did lost them.

What it was, measured over the ring (50 by 5 pixels, both clouds and the mesh
path traced with the same camera and the same default lights):

| | R | G | B | ratio |
|---|---|---|---|---|
| mesh | 0.225 | 0.168 | 0.086 | 1 : 0.75 : 0.38 |
| carried and relit | 0.452 | 0.334 | 0.177 | 1 : 0.74 : 0.39 |
| baked, before | 0.298 | 0.289 | 0.264 | 1 : 0.97 : 0.89 |
| baked, after | 0.406 | 0.325 | 0.191 | 1 : 0.80 : 0.47 |

The baked ring was not dark. It was **the right level with no colour in it**,
which is a surface lit and then multiplied by the wrong albedo -- and the
albedo it had was the marble of the body, a few millimetres below.

**How far off the surface a bake ray starts is a fraction of the model, and of
nothing else.** It was a fraction of the scene's unit instead -- a thousandth,
floored at one -- and the chess pawn is 66 mm tall in a stage whose unit is a
metre. So every ray began **a millimetre** above its gaussian: thicker than
the gold ring, and high enough to start *inside* the glass ball above it. The
ray then came down onto whatever that offset had put it in front of, which for
the ring was the body behind it and for the ball was its own far side. The
floor was there to keep the step from being zero for a model at the origin; it
made the step enormous for every model smaller than a metre. It is now the
bounding box's diagonal times 1e-4 -- 8.8 µm on this pawn.

**And a bake is not a frame, so it has no camera hit to cache.** The tracer
shades the first vertex once a pixel and reuses it for every sample, which is
right when the sample only changes what happens *after* that vertex. A bake's
samples each look at the point from a different direction, so the cached shade
answered all 256 of them with sample zero's eye, and every harmonic above the
constant was noise about zero. `cameraHit` is now false in a bake.

**Where it lands.** The ball reads 0.198/0.304/0.266 against the mesh's
0.222/0.333/0.306, and the ring has its gold back. What is still over is a
factor of 1.8 on the ring, and it is not the bake's: the ring is five pixels
tall and the gaussians that stand on it are wider than it is, so a crop that
catches dark edges on the mesh catches gold on the cloud. The cloud that was
never baked is over by the same factor.

**And the MCP server can ask for the default lights** (`defaultLights`), which
it could not: it drew a stage exactly as the stage stood, so the chess set --
which carries no light of its own -- came back black on black, and no
comparison with what `lrt view` and `lrt stage --default-lights` show was
possible through it.

## The bake fits the harmonics, and no longer projects them

The harmonics were fitted by **projecting** the radiance over the whole sphere
with the half the surface does not face taken as nothing -- which keeps the
basis orthogonal, and is a different question from the one a bake is asking.
Three things came of it, and they are one thing:

- **Degree 0 at a fifth.** The projection gives `Y0 * 2 pi * mean` and reading
  it back gives `kSH0` times that, which is **half** the light the surface
  sends. The pawn's glass ball read 0.048/0.059/0.054 against the mesh's
  0.224/0.335/0.307.
- **A dark rim around every silhouette.** A silhouette is the surface seen
  from the equator of that half -- exactly where the function being projected
  steps from the radiance to nothing, and where a series through a step is
  worth the middle of it. Across the ball, the mesh reads a flat 0.309 and the
  cloud read **0.152 at the edge**, climbing over a hundred pixels to 0.325 at
  the centre.
- **Degree 3 further from the mesh than degree 2**, ringing about it
  (0.224, 0.271, 0.193 across the same ball). Gibbs, not noise: more paths did
  not touch it.

**What settles it** is that a Lambertian surface's radiance does not depend on
direction, so neither can the answer depend on the degree. Over the pawn's
marble body, which is diffuse once the polish is taken out of it:

| | degree 0 | degree 2 | degree 3 |
|---|---|---|---|
| projecting | 0.045 | 0.094 | 0.069 |
| fitting | 0.0749 | 0.0725 | 0.0745 |

Three answers to a question with one, against three that agree to two percent.
`tests/usd/test_usd.cpp` holds that invariant on a plane under a dome.

**The fit, and why it is small enough to solve a gaussian at a time.** It is
the normal equations `G c = b` over the half of the sphere the surface faces,
with `b` what the samples already sum to and `G_kj = integral(Y_k Y_j)` over
that half. Two bands of the **same parity are orthogonal over any half of the
sphere**: `Y(-w) = (-1)^l Y(w)`, so the two halves' integrals are equal and
each is half the sphere's, which is `delta/2` whatever the normal is. So only
the block between the even and the odd bands depends on the direction the
surface faces, and only it is worked out -- six by ten at degree 3. What is
left is `[[I/2, E], [E^T, I/2]]`, whose Schur complement is **six by six**,
symmetric and positive definite, and Cholesky ends it. Degree 0 falls out of
the same arithmetic: no odd bands, so `c = 2 b`, which is the half the
projection was missing.

**The matrix costs no rays.** It depends on the normal and on nothing else --
not the light, not the material, not a path. Estimated from the paths instead,
at 256 of them, the fit came apart: the sampling error on the matrix is the
size of the entries themselves, and the pawn came back with pixels in the
thousands. It is integrated deterministically over 32 elevations by 16
azimuths, which for a product of two basis functions is worth about four
figures, and costs 512 directions of arithmetic against 256 of ray tracing.

**Where the noise is stopped, and why not with a ridge.** Half of a sphere
does not determine sixteen harmonics equally: the combinations that are nearly
nothing on the half the surface faces are what the data cannot see, and solved
exactly the fit puts a few hundred paths` noise into exactly those. On the
pawn a few gaussians reached **65344** -- fp16's ceiling, which is what a
cloud stores them in -- and burned out as white blobs.

A ridge on every diagonal fixes it and charges every gaussian for the few that
need it. At 0.02 the whole fit shrinks by `0.5 / 0.52`, and the test's
Lambertian plane, whose light is 0.4614, came back at **0.4436** -- 3.9% low,
exactly that ratio -- with degree 2 at 9%. A **floor under Cholesky's pivot**
charges nobody: the pivot only goes small where the matrix is near singular,
which is the direction the data could not see, so flooring it bounds what that
direction contributes and leaves every well determined gaussian solved
exactly. At 0.005, against a diagonal of a half, the plane comes back at
**0.4614 exactly at degree 0** and within 1.4% and 2.0% at degrees 2 and 3 --
which is the paths' own noise -- and the pawn's brightest pixel is 8.4 where
the mesh's is 41.

**Where it lands** (1600 by 1600, 512 paths, degree 3, 256 paths a gaussian):

| patch | mesh | carried and relit | projected | fitted |
|---|---|---|---|---|
| glass ball | 0.224/0.335/0.307 | 0.208/0.328/0.299 | 0.198/0.304/0.266 | 0.173/0.265/0.237 |
| gold ring | 0.224/0.167/0.086 | 0.456/0.337/0.179 | 0.408/0.327/0.192 | 0.306/0.239/0.137 |
| marble body | 0.084/0.099/0.097 | 0.064/0.080/0.078 | 0.073/0.094/0.091 | 0.061/0.075/0.074 |

Across the ball's silhouette, where the projection swung from 0.152 to 0.325
against a mesh that is flat at 0.300 to 0.320, the fit reads 0.238 to 0.262 --
**flat**, which is what the dark rim was.

The fit is nearer the mesh than the projection on the ring and further on the
ball and the body, and what is left is **not the harmonics**: the cloud that
was never baked at all sits in the same place as the fitted one on the body
(0.080 against 0.075 against the mesh's 0.099), so the last quarter is the
conversion's own -- how wide a gaussian is against its cell, and what it
therefore covers -- and not how its colour was arrived at.

## The sky was painted where nothing was drawn, not behind what was

Every silhouette in a converted cloud wore a dark fringe. Zoomed to where a
viewer puts it, that fringe is a row of black splinters, which is how it was
noticed.

It is not the cloud and it is not the bake -- the cloud that was never baked
has it too, and so does the rasteriser. Read out of the render buffer, the
pixel just outside the pawn's collar is **0.087/0.071/0.049 with an alpha of
0.199**: a fifth of the body's colour, premultiplied, and none of the four
fifths of sky that belongs behind it.

`dome_background.slang` painted the sky where the depth buffer said nothing
had been drawn, and wrote it **over** the pixel. For a mesh that is right: a
mesh covers a pixel or it does not. A cloud's silhouette is a ramp of partial
coverage five pixels wide, and every one of those pixels has a depth, so none
of them got any sky at all.

The sky is opaque and it is behind everything, so it is composited under, by
the pixel's own coverage, and the pixel is opaque afterwards. The profile
across the collar now runs 0.600, 0.567, 0.554, 0.536, 0.488, 0.446 -- the
sky falling to the body -- where it ran 0.600, 0.087, 0.123, 0.202, 0.332,
0.446. **A fully covered pixel is bit for bit what it was**, which is what a
mesh always saw.

The light groups' planes take the sky under the same coverage, so that they
still sum to the beauty, and they read that coverage out of the beauty's own
alpha -- which the beauty pass sets to one, so the groups are dispatched
first.

**And a gaussian the bake found nothing under is not a black gaussian.** Its
coefficients come back as zeros, and zero is not "no colour": the constant
term is kept shifted to where 3DGS trains it, so a zero there decodes as
`0.5 - 0.5`. A cloud out of mesh2splat is nearly all discs, so one of those
seen edge on at a silhouette is a black splinter of its own -- thirty-nine of
them in 729073 on the pawn. It stands for nothing, so it now draws nothing.

**What was tried and was not it.** The third axis: mesh2splat writes 1e-7
there, a length in the model's own units while the other two sizes are in
cells, so on a pawn 66 mm tall it is a thousandth of a cell -- a disc a ray
meeting it side on is barely stopped by. Widened three hundred times, the dark
minimum at the edge was **just as deep** (0.027 against 0.033), so the razor
was not what made the fringe, and EA's number stands.

## The conversion's order is the mesh's, and a slot with nothing in it is kept

Two changes to the static conversion that nothing animated can do without, and
that are worth having on their own.

**The same mesh now writes the same array.** The slot a gaussian went into came
from `InterlockedAdd(counters[0], 1u, slot)`, and the shader said so in its
header: *"it keeps the atomic append, which is what makes the order of the
output nobody's business"*. Measured rather than assumed: two conversions of
the chess pawn, same options, same stage, wrote `.usdc` files that **differ
from byte 1001**. The same 729073 gaussians, in a different order.

That is fine for one still frame and impossible for a sequence, because a
gaussian is followed from one pose to the next by being the same element of the
array. So the emit is now three passes: a thread counts each triangle's covered
cells, one workgroup settles where each triangle's gaussians start, and a
thread writes each triangle's at that offset. Two conversions now give files
that are **identical byte for byte**.

The scan is one workgroup of 256 in three phases -- each thread adds up a
slice, thread zero runs the 256 slice totals into a running sum, each thread
lays its own running sum down. A serial scan in one thread was the other
option; the pawn has 42892 triangles in one of its two meshes and a character
will have more.

Two things fall out of it. The per-triangle geometry is worked out by one
function, `m2sTriangleOf`, that both the counting pass and the writing pass
call, so they cannot disagree about a single bit. And **a budget too small now
keeps the first splats in the mesh's own order** rather than whichever ones won
a race.

**A slot with nothing in it is still a slot.** The writer skipped a record that
decoded to nothing -- a scale that overflowed, an opacity under a 255th, a
position that was not a number (`splat_export.slang`) -- so the file came out
shorter than the conversion that made it. In a sequence that is worse than
untidy: a triangle that goes degenerate in one pose alone would take its
gaussians out of that frame's array and put every later gaussian out of step,
in that frame and in no other. The slot is kept now and written empty: no
opacity, no size, a position that is at least a number, and zero coefficients.
Nothing draws. The cloud's extent is folded over the splats that are there, not
over where an empty slot happens to stand, and the writer says how many it
wrote empty.

Measured on the pawn: a bake at 16 paths leaves 10 gaussians of 729073 with no
surface under them, and the file now carries 729073 where it carried 729063.

**What checks them**: `tests/aofx/test_mesh2splat.cpp` runs the conversion twice
and counts, in a kernel, the entries that differ bitwise (0 of 1024); and
`tests/usd/test_usd.cpp` writes a cloud with three records that decode to
nothing and reads back the array lengths, which are the conversion's.

## The conversion reads a pose, not a rest

`lrt mesh2splat` had a `--time`, and it reached the bake's ray tracing and
nothing else. The gaussians came from `MeshStage`, which read every attribute
at the stage's **default** time (`UsdGeomXformCache` default-constructed, every
`Get` with no time argument), so at any other instant the rays stood where the
mesh used to be while the scene they traced was somewhere else. On a static
stage nobody could see it; on an animated one it is the whole of the answer.

`MeshStageOptions` carries a `time` now, and the geometry reads and the xform
cache take it. For a stage with no animation in it **nothing changes**: an
attribute with no time samples answers with its default whatever time is asked
for, and the chess pawn converts to a file identical byte for byte to the one
it converted to before.

**A skinned mesh's `points` attribute does not animate**, though, because the
deformation belongs to the skeleton and USD resolves it through UsdSkel. A
conversion that reads `UsdGeomMesh` directly would see the rest pose at every
time. The renderer does not have this problem -- usdSkelImaging hands Hydra an
ext computation and `geom::Skinner` runs it on the device (M7) -- but the
conversion goes round Hydra on purpose: it wants none of a render index, and
the whole material side is built on `MeshStage`'s own narrowing.

So the stage is **posed** before it is read, with `UsdSkelBakeSkinning`, which
writes the posed points as time samples onto the meshes themselves. Two things
make it cheap and safe: it is baked into the **session layer**, so the file on
disk is untouched, and over `GfInterval(time, time)`, so it costs one pose and
not a range. A stage with no `SkelRoot` comes back unchanged. The reads below
then need to know nothing about skinning, and there is no second
implementation of UsdSkel in this repository.

What it leaves out, and what going round Hydra costs: instancing, velocities,
visibility and purposes. A point instancer's copies are not converted, and a
stage whose motion is authored as velocities rather than samples is read at
the sample.

**Measured** (`tests/usd/test_usd.cpp`): a square carried entirely by one joint
that slides (1.2, 0.4, 0) between time 0 and time 1 converts to a mesh whose
bounds -- folded on the device, not on the host -- are the square as authored
at time 0 and exactly that slide away at time 1.

## A gaussian carries the joints its triangle carries

The second half of reading a rigged asset: not the pose, but what moves it.
`MeshStage` now resolves each mesh's skel binding -- `UsdSkelCache` over every
`SkelRoot`, one `UsdSkelSkinningQuery` a skinnable prim -- and hands back, per
mesh, the joints each of its points is held by and how much, its
`geomBindTransform`, its skeleton's path and its skeleton's joint order.

**In the skeleton's order, not the mesh's.** A mesh may name its own subset of
the skeleton's joints with `skel:joints`, and the indices UsdSkel hands back
are then into that subset. USD keeps a mapper for it, and the mapper runs the
other way -- skeleton order to the mesh's -- so it is run over the identity to
learn each mesh index's skeleton index, and the influences are written in the
skeleton's order. One cloud then has one joint order whatever mixture of
meshes it was converted from.

**Posed, or carried, and never both.** `MeshStageOptions::skinned` reads the
**bind** pose and skips `UsdSkelBakeSkinning` entirely, because that is where
the skeleton's transforms expect to find the geometry; posing first would skin
it twice.

**The blend is the effect's.** `mesh_pack.slang` puts each triangle corner's
four heaviest influences in a second picture of exactly the same shape as the
mesh's, so the effect addresses the two alike and needs no second set of
dimensions. The effect then blends the three corners by the barycentric
coordinates of the cell the gaussian stands in, which is what interpolating
the skin means: twelve `(joint, weight)` pairs go in and four come out -- a
joint already there gains the weight, an empty slot takes it, otherwise it
displaces the lightest -- and the four are renormalised, so a gaussian is
carried entirely however its triangle was authored. It is the effect's because
it is the effect that knows where inside the triangle the gaussian stands.

The record grows to **eight entries**: four, six with the PBR channels, eight
with the joints, and nothing in between. `writeInfluences` and the
`Influences` clip are additive, so the bundle's ABI is untouched and
`aofx_sdk_manifest` never moves.

**`--skinned` forces `--no-bake`**, and says so. What the harmonics hold is
this scene's environment and its bounce -- the ground under a paw is in them --
and carrying that up with the leg when it lifts is the mistake of rotating a
lightmap. A cloud a skeleton moves carries its material and is relit every
frame, which is right by construction.

**Measured**: a square bound entirely to one joint converts to 1056 gaussians
that all name that joint with all of the weight; and through the effect, 272
gaussians of a quad whose corners all name joint 3 come out on joint 3, with
nothing in the other three slots and the weights summing to one
(`tests/aofx/test_mesh2splat.cpp`, `tests/usd/test_usd.cpp`).

**Not done here**: nothing yet deforms those gaussians. The cloud carries its
joints and the file does not write them.

## A cloud carried by a skeleton, on the device

`scene::SplatSkinner` is `geom::Skinner`'s counterpart for gaussians, and it
takes the same inputs in the same layout because they come from the same
place: the joints each gaussian is held by, the skeleton's transforms at this
instant, and the three matrices that take a point from the cloud's own space
to the skeleton's and back. One thread a gaussian
(`shaders/lrt/scene/splat_skin.slang`).

**The position is exactly the mesh's arithmetic.** The same linear blend, the
same `geomBindTransform` then `skelLocalToWorld` then `primWorldToLocal`, the
same transposition on the host -- `GfMatrix4f` is row-major with vectors on
the left and the kernel multiplies rows by a column, so the matrix goes over
turned, as a mesh's does.

**The frame is the same chain's linear part.** The two axes a gaussian spreads
along are carried by `worldToPrim . skelToWorld . (sum of w_j X_j) . geomBind`
-- which is the Jacobian of the position map, so a gaussian stretches with its
triangle instead of sliding along beside it. They are squared up again
afterwards, because a joint may shear where a rotation would not, and the
third axis is a disc's and is left alone. The two sizes in the plane come out
as the lengths of the carried axes.

**What is not touched** is everything else a gaussian carries: its opacity,
its colour, its harmonics, its PBR channels, and the word that holds its third
size. A frame therefore costs one kernel over the cloud and no re-decode of
anything -- and, because the deformed cloud is written into buffers that
outlive it, the cloud's identity never changes from one frame to the next.
That is what will let a ray tracer refit rather than rebuild.

`quaternionOfAxes` moved into `common/packing.slang`, beside the quaternion's
own packing, since the LOD merge and the skinner both want it; the LOD build
now calls the one in common rather than its own copy.

**Measured** (`tests/scene/test_loading.cpp`, 4096 gaussians on a helix, each
turned differently and each a different size): a skeleton at rest leaves every
one of them where it was, and a skeleton given a rigid turn of 1.2 radians
about a slanted axis and a slide carries every one of them rigidly -- the
position by the transform, the frame turned with it, the sizes not at all.
Nothing is read back but four counters.

The tolerance is the format's and not the arithmetic's: a frame is a
smallest-three quaternion at ten bits a component, so an axis cannot be pinned
closer than about `sqrt(2)/1023`. At rest the frame written is the frame read,
and re-encoding a value that was already a word's decode lands on that word --
except at the boundary of the rounding, where two gaussians of 4096 did.

## The file carries the rig, not the frames

`LrtSplatSkinningAPI` (`modules/usd/schemas/generatedSchema.usda`) is what a
cloud a skeleton moves writes beside its gaussians: `jointIndices` and
`jointWeights`, four a gaussian with `elementSize = 4`; a
`geomBindTransform`; the `skeleton` it was converted against, for provenance;
and `skinningXforms`, one transform a joint in the skeleton's order, **time
sampled**. That last is the only thing about an animated cloud that changes
from one frame to the next, and for sixty joints it is four kilobytes a frame.

The arithmetic, for a plausible character at 150k gaussians over 120 frames:

| what the file carries | size |
|---|---|
| positions, orientations, scales as time samples | **960 MB** (half), 1.2 GB (float) |
| the rig: four joints a gaussian, plus 60 joints a frame | **2.9 MB** |

And the rig is exact at *every* instant of the range, not only at the frames
somebody sampled.

**The transforms are carried on the cloud, not bound to the Skeleton.**
`UsdSkelBindingAPI` on a `ParticleField3DGaussianSplat` is not something
UsdSkel sanctions, and usdSkelImaging makes no ext computation for a prim that
is not `UsdGeomPointBased` -- so nothing would reach a renderer. Carrying the
matrices makes the file answer for itself. **Not done**, and the cost of that
choice: retargeting and clip blending, which a cloud holding baked transforms
cannot do.

`lrt mesh2splat --skinned [--range START:END[:STEP]]` gathers them with
`UsdSkelSkeletonQuery::ComputeSkinningTransforms` at each instant, the stage's
own range by default and a time code a step. Splitting the conversion's
`(joint, weight)` pairs into USD's two arrays is a rearrangement of values the
device computed, as the record unpack already is, and is said so where it
happens.

**Measured**: the rigged square converts to 272 gaussians whose file holds its
two joints at two instants, the second sliding (1.2, 0.4, 0) exactly as the
`SkelAnimation` authored it; and `tests/usd/test_usd.cpp` writes a cloud with
three joints over three time codes and reads every primvar, the element size,
the time samples and the stage's range back off the file.

**Not done here**: nothing reads them yet. `ParticleField::Sync` does not look
for the new primvars and no frame is deformed by them.

## And the engine puts the cloud where the skeleton is

The reading side, which closes it. `ParticleField::Sync` looks for the rig's
primvars beside the ones it already reads, and a time code change dirties the
one of them that is time sampled -- so the frame that follows has the
skeleton's transforms at that instant and nothing else has moved.

`Engine::carryCloud` then does three things. It pairs the file's two arrays
into the `(joint, weight)` layout every skinner here reads, and transposes the
joints' matrices the way `geom::Skinner` does, because USD puts vectors on the
left and the kernel multiplies rows by a column. It keeps the uploaded cloud
as the **bind pose** and writes the posed one into buffers of its own -- a
`GpuSplats` that shares the harmonics and the PBR channels, since the skinner
writes neither -- so what a renderer holds never changes identity between
frames. And it runs `scene::SplatSkinner` over the cloud.

**Two things the measurement turned up**, both of which would have been wrong
in any route that posed a cloud:

- The posed cloud's **extent** is not the bind pose's. A skeleton moves a
  cloud out from under its own box, and everything that culls, frames or sorts
  by it would have been looking in the wrong place. It is folded again on the
  device after the skinning.
- `Engine::bounds()` folded `entry.gpu->bounds` -- the cloud as uploaded --
  rather than the cloud as drawn. It now folds what is drawn.

**Where the spaces meet.** The conversion packs its triangles in world space,
so the gaussians stand there and not in the mesh's own space, while UsdSkel's
bind transform starts from the mesh's. The two are composed once, at
conversion, so what the file carries is the one matrix a renderer needs: the
cloud's own space into the space the joints are measured from. **Not done**: a
stage with several skinned meshes at different transforms keeps only the
first's, and a skeleton with an animated transform of its own is not folded in.

**Measured** (`tests/usd/test_usd.cpp`): a cloud of 512 gaussians whose file
says one joint carries all of them, sliding a unit in x a frame, is drawn two
time codes on with its bounds slid exactly two units in x and not a thousandth
in y or z. And end to end, the rigged square converts and draws from one fixed
camera at two instants with the gaussians where the `SkelAnimation` put them.

**Still not done**: the whole cloud is re-uploaded at every time step before
it is posed, because `ParticleField::Sync` re-reads every array on any
`DirtyPrimvar` and `CloudLoader::upload` has no in-place counterpart. The
skinning adds one cheap pass to that; what it does not do is make the frame
cheap. That is the next change, and it is on the other side of the boundary
from this one.

## What a rigged cloud actually costs

Measured, not estimated. The asset is a tube of 200 rings by 64 segments --
25472 triangles -- with forty joints down its length, two influences a vertex
blended smoothly between neighbours, and a travelling wave of rotations over
25 time codes. It stands in for the Fox until `Fox.glb` has been through a USD
exporter; what it exercises is the same path, and its weights are smooth,
which one influence a vertex would not have been (bound one ring to one joint,
the rings tear apart and the picture shows it).

`lrt mesh2splat --skinned --resolution 512`, on the Mac in release:

| | |
|---|---|
| gaussians | **184 320** |
| joints | 40 |
| the file, at 25 instants | **9.27 MB** |
| the file, at 121 instants | **9.50 MB** |
| the same range as per-frame arrays, half precision | **~713 MB** |

**Five times the frames costs 230 kilobytes.** That is the whole argument for
carrying the rig rather than the frames, and it is why the file is one number
and the sampled alternative is two orders of magnitude bigger. It is also
exact between the instants, where a sampled cloud is whatever the reader
interpolates.

(With two influences a vertex the file is 12.55 MB rather than 9.5: the
blend gives most gaussians four joints where one influence gave them one, and
the joints and weights are 32 bytes a gaussian.)

**A frame costs about 0.9 s** at 640 by 640 path traced, 184320 gaussians --
and almost none of that is the skinning. It is the cloud being re-uploaded and
its BVH rebuilt at every time step, which is the same cost a cloud with
per-frame arrays would pay and is written down above as the next change. The
skinning itself is one kernel over the gaussians.

**Not done**: the Fox. `Fox.glb` is CC0 for the model and CC-BY for the rig
and the glTF conversion (PixelMannen; tomkranis; @AsoboStudio and @scurest),
and getting it into USD needs an exporter this machine does not have -- `guc`
says plainly that "all glTF features with the exception of animation and
skinning are implemented", and Blender is not installed. The run above is what
that run would report.

## The fox

`scripts/fetch-fox.sh` fetches the Khronos glTF sample and turns it into a USD
stage with its rig: model CC0 by PixelMannen, rig and animation CC-BY 4.0 by
tomkranis, glTF conversion CC-BY 4.0 by @AsoboStudio and @scurest. It needs
Blender, which is a dependency of that asset and of nothing else -- `guc`, the
converter this project would otherwise use, says plainly that animation and
skinning are the two glTF features it does not implement.

`lrt mesh2splat --skinned --resolution 512`:

| | |
|---|---|
| triangles | 576 |
| gaussians | **204 517** |
| joints | 24, over 28 instants |
| the cloud | **13.35 MB** |

Rendered pose for pose against the mesh, path traced under the same lights,
the cloud is the fox: rounder at the silhouettes, because a gaussian rounds
off a 576-triangle model's facets, and the same animal in the same pose.

**Three things it turned up**, all of them defects nothing in this repository
had been able to see before, because every asset here until now was hand-built
or Pixar's:

- **One undeclared input lost the whole material.** Blender's USD exporter
  writes `inputs:specular` on a `UsdPreviewSurface`, which the specification
  does not have -- it has `specularColor` -- and MaterialX refuses a node
  whose interface does not match its declaration, so the fox arrived grey:
  *"Could not find a nodedef for node 'Surface'"*. Every asset out of Blender
  did. An input nobody declared is now dropped with a line saying so, beside
  the pass that already repairs mismatched input *types* for the same reason.
- **The written cloud was always Y-up.** The gaussians are in the source
  stage's world space, and `writeParticleFieldStage` said `upAxis = "Y"`
  whatever that stage said -- so a fox exported Z-up, as Blender exports, lay
  on its side. The export carries the source's up axis now.
- **The flat axis is a fraction, not a length.** mesh2splat writes `1e-7`
  there, in the model's own units, while the two sizes across the surface are
  in cells: how thin a gaussian is then depends on how big the model happens
  to be. The chess pawn is 66 mm across and traced correctly; the fox is a
  hundred units long, which makes the same `1e-7` fifteen hundred times more
  extreme, and **the ray tracer saw a ghost where the rasteriser saw a fox** --
  it integrates density along the ray rather than projecting an ellipse.

  It is `min(alongU, alongV) * flatness` now, default 0.1. Measured: the fox
  becomes solid, and the pawn does not move -- its marble body reads
  0.0645/0.0799/0.0780 where it read 0.0644/0.0798/0.0778, and its gold ring
  0.459/0.339/0.180 where it read 0.456/0.337/0.179.

  This was tried once before, as a guess at the dark fringe around every
  silhouette, and **reverted because the measurement did not support it**: the
  fringe was the sky not being composited under partial coverage, and widening
  the axis three hundred times left the dark minimum exactly as deep. The same
  change is right here for a different reason, and this time the measurement
  says so.

## Two assets out of Blender, and the two bugs they found

The BMW 1M of `bmw27` (Mike Pan, CC-BY, `download.blender.org/demo/test/`)
and the Eurasian tree sparrow's flight cycle are the first assets converted
that were not authored for this renderer. Each found one thing wrong.

### A gaussian cannot be bigger than the triangle it stands on

`mesh2splat.slang` takes a gaussian's two sizes across the surface from the
columns of the Jacobian of the triplanar map, `J = V O⁻¹`, and those columns
carry `1 / determinant`. The determinant is guarded against zero at `1e-24`,
which is the guard the projection needs -- but a **sliver seen almost edge on
by its own projection** passes that guard with a determinant of, say, `1e-20`
and the columns come back twenty orders of magnitude too long.

On the BMW, 1859 triangles did: the frame filled with white spikes metres
long radiating from the headlights and the wheel arches. They were there in
the relit cloud and in the baked one, and they were not noise -- a handful of
enormous gaussians.

The bound that is always true is local and needs nothing measured: a gaussian
belongs to a triangle, so it cannot be longer than that triangle's longest
edge.

```slang
const float longestEdge =
    sqrt(max(dot(e1, e1), max(dot(e2, e2), dot(t.c - t.b, t.c - t.b))));
const float alongU = min(length(ju) * params.sigmaX * perCell, longestEdge);
const float alongV = min(length(jv) * params.sigmaY * perCell, longestEdge);
```

It binds only where the projection had already failed: the pawn, the fox and
the car's well-shaped triangles are unchanged, since `length(ju) * sigma /
resolution` is a fraction of a cell there and a cell is far smaller than an
edge.

### A primvar's indices are read at the frame's time

`MeshStage` read texture coordinates at the frame's time and their **indices
at the default time**:

```cpp
primvar.Get(&uvs, at);
primvar.GetIndices(&uvIndices);   // no time
```

Blender writes `primvars:st:indices` as *time samples* when the export carries
animation. An attribute with samples and no default answers nothing, so the
indices came back empty, the face-varying values were then consumed
positionally, and every mesh whose st primvar was indexed sampled one texel:
the sparrow's wings came out flat mauve while its body, whose primvar is not
indexed, was correct. `GetIndices(&uvIndices, at)` is the whole fix.

### What the sparrow needed on Blender's side

Blender's USD exporter has `export_animation = False` by default, and with it
off a `SkelAnimation` is still written -- carrying `blendShapeWeights` and no
joint samples at all. That is what made the first sparrow look static.

With it on the exporter still wrote no joint samples for this rig, whose
action comes from an FBX import and keeps its curves in a slot's channelbag
rather than in `action.fcurves`. The animation is therefore written here, by
sampling the **evaluated pose** -- which is true however the pose is driven --
and solving the per-bone change of basis from the `restTransforms` the
exporter itself wrote, so whatever convention it used is the one reproduced.
The rest pose round-trips to `4.13e-6`; 363 of the 609 joints move across the
cycle.

USD joint names are the Blender bone names with every character USD will not
have in an identifier turned into an underscore, so `Spine.001_Pelvis` is
written `Spine_001_Pelvis`. The map back is by sanitising the *bone* names and
looking the joint up in that, not the other way round.

### The numbers

| | gaussians | joints | instants | file |
|---|---|---|---|---|
| BMW 1M, baked degree 2 | 6 774 631 | -- | -- | resolution 1200 |
| sparrow, skinned | 4 991 908 | 609 | 33 | resolution 1400 |

The BMW's body alone wants 2 917 105 gaussians at resolution 1200 and is
capped at the 2 097 152 a single run may write; the sparrow's feathers want
15 378 695 and take the same cap.

**Relighting a car under a softbox does not work**: an 11.7 × 7.0 area light
over a cloud whose paint is metallic 0.85 at roughness 0.19 blows out, because
a splat has no occlusion against the ones behind it unless `--splat-shadows`
pays for it. The bake is both cheaper and right for a turnaround, whose lights
do not move.
