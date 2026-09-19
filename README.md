# lucabRTrender

A Slang renderer for Gaussian splats and point clouds, and the engine
openFXplayer is to stand on. macOS (Metal) and Linux (CUDA/OptiX, Vulkan)
first; Windows later.

Everything numeric runs on the GPU, reference renders and test oracles
included. The CPU reads files, parses headers and decompresses (SPZ, SOG's
WebP); decoding, sorting, levels of detail and image comparison are compute
kernels.

## What it does

- **Tile rasteriser** for splats: a global radix sort on the GPU, no per-tile
  limit, Mip-Splatting filters, and SH in fp16.
- **Ray tracer** for splats, with two routes and one integrator:
  - hardware acceleration structures, with icosahedron proxies and ray
    queries;
  - a Karras LBVH built and traversed in compute, where there is no RT
    hardware.
- **Points**: raster or compute discs, eye-dome lighting and surface
  splatting, composited with splats.
- **GPU ground truth**: a brute-force per-pixel reference for the rasteriser
  and one for the ray tracer. Tests compare on the device and read back only
  p99 and max.
- **Loaders**: PLY, `.splat`, SPZ and SOG, decoded on the device.
- **Levels of detail and streaming**:
  - Octree levels are built on the GPU by moment matching.
  - A per-frame cut runs on the GPU.
  - `.lrtc` files carry the levels and are streamed chunk by chunk into a
    fixed budget, most-wanted chunks first.
- **USD/Hydra 2.0**: a render delegate (`hdLrt`) for
  `ParticleField3DGaussianSplat`, `Points` and cameras, with codeless schemas:
  - `LrtSplatEditAPI`
  - `LrtPointStyleAPI`
  - `LrtStreamedAssetAPI`, which references a `.lrtc`.
- **SplatEdit**: openFXplayer's selection-and-grade rule, applied the same way
  by every renderer.
- **Time**:
  - Frames at their SMPTE ST 2059-1 instants, on this machine's clock or a PTP
    master's (genlock).
  - EXRs carry timecode and TAI.
- **aofx**: openFXplayer's plugin SDK and host at ABI 22. Bundles built by
  openFXplayer load unchanged.

## Building

Needs CMake ≥ 3.24, Ninja and a C++20 compiler, plus:

- **Slang 2026.14.1** in `~/tools/slang`: one Slang for gpe's blobs and for
  slang-rhi.
- **OpenUSD 26.08 with MaterialX 1.39.5 and OpenVDB** in
  `~/tools/usd-26.08-mx`: `scripts/build-usd.sh` builds it once per machine.
- **Open Image Denoise 2.5.1**, GPU devices only, in `~/tools/oidn-2.5.1`:
  `scripts/build-oidn.sh`. Without it the engine has no denoiser.
- **libwebp** (Homebrew `webp`) for SOG; without it, `.sog` is refused.
- **The submodules**: `git submodule update --init --recursive` (gpe,
  genlock).

```sh
cmake --preset macos-arm64-release          # or macos-arm64-debug, linux-x86_64-*
cmake --build --preset macos-arm64-release
ctest --preset macos-arm64-debug            # GPU tests skip on a machine without a device
```

slang-rhi, tinyexr, CLI11, nlohmann_json, Catch2 and SPZ come through
FetchContent or `third_party/`. slang-rhi is patched at fetch time
(`cmake/patches`).

## Using it

```sh
lrt info                                              # device, capabilities, Slang, shader path

lrt render --splats scene.ply --rotate-x 180 --size 1920x1080 -o out.exr
lrt render --splats scene.ply --technique rt          # ray traced (rt-hw, rt-bvh to force a route)
lrt bench  --splats scene.ply --repeat 20 --stages    # per-stage timings

lrt convert scene.ply scene.usdc                      # a USD ParticleField stage with a camera
lrt convert scene.ply scene.lrtc                      # levels of detail, chunked for streaming
lrt render --splats scene.lrtc --lod 2                # cut: merged cells up to 2 px
lrt render --splats scene.lrtc --lod 2 --stream-budget 262144   # stream into 262k splats

lrt stage shot.usda --camera /World/Camera --technique raster -o out.exr

genlock-cli master --port 3190                        # a PTP master (any port >1024 both ends agree on)
lrt live shot.usda --ptp 127.0.0.1 --port 3190 --rate 25 --frames 250 \
         --at 16:07:14:00 -o live.####.exr            # every node given the same --at draws the same frame

lrt aofx list
lrt aofx run tv.mediapro.aofx.invert in.exr out.exr
```

- **Hydra.** To use the delegate in any USD application, set
  `PXR_PLUGINPATH_NAME=<build>/plugin/usd`.
- **Shaders** are compiled at run time for the device that opened. They are
  found through `LRT_SHADER_DIR` or `<exe>/../shaders`.

## What it owes to Falcor

[Falcor](https://github.com/NVIDIAGameWorks/Falcor), NVIDIA's real-time
rendering research framework, is one of the two renderers this one was read
out of before it was written. It sits under `ref/falcor` as reading material
and is **never built, never linked and never shipped**: no header of its is
included anywhere here, and nothing of it ends up in a binary. It is a
reference the way a paper is.

What it was read for, concretely:

- **Loop subdivision.** `ref/falcor`'s `LoopSubdivide` was read for the Loop
  stencils the GPU subdivider implements -- and only read; the implementation
  here is a set of compute kernels checked against closed forms, not a port
  (`docs/decisions.md`, M8, "the limit projection is `subdivLimit`").
- **How a Slang renderer is laid out**: one shader source compiled for
  whichever device the machine has, a render graph of techniques over a
  visibility buffer that every route fills alike, and a reference renderer to
  test the fast paths against. That shape is Falcor's, and it is why
  `modules/technique` reads the way it does.

Where this engine goes its own way is written down too: it refuses CPU
arithmetic on scene data (no CPU reference renderer, no fallback, no test
oracle -- the ground truth is a GPU kernel), its entities come from OpenUSD
through a Hydra 2.0 render delegate rather than from a scene format of its
own, its materials are MaterialX graphs compiled into Slang, and it carries
Gaussian splats as a primitive beside triangles rather than as a demo.

`ref/spire-engine` is kept for the same reason and on the same terms.

## The assets it is shown with

`~/tools/assets`, fetched rather than checked in:

- **OpenChessSet** -- the marble pawn with the glass head, which is what the
  conversion's textures, transmission and path-traced bake were measured on.
- **Kitchen_set** -- Pixar's, for the breadth of a real stage.
- **Fox** -- the Khronos glTF sample, fetched and turned into USD by
  `scripts/fetch-fox.sh`, which is the rigged asset `lrt mesh2splat --skinned`
  is shown on. Model **CC0** by PixelMannen; rig and animation **CC-BY 4.0**
  by tomkranis; glTF conversion **CC-BY 4.0** by @AsoboStudio and @scurest.
  <https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox>
- **bmw27** -- Blender's own benchmark scene, a BMW 1M on a seamless
  backdrop, **CC-BY** by Mike Pan. Its materials predate the Principled BSDF,
  so the Cycles node trees are reduced to Principled equivalents before the
  USD export. <https://download.blender.org/demo/test/BMW27_2.blend.zip>
- **Eurasian tree sparrow** -- a rigged and flying bird, 609 joints, whose
  clips arrive as FBX. It is the asset `--skinned` is shown moving on, and
  the one that made the case for writing the `SkelAnimation` here rather than
  relying on Blender's exporter.

The script needs Blender, which is a dependency of that asset and of nothing
else: no build, no test and no part of the engine uses it. `guc`, the glTF to
USD converter this project would otherwise reach for, says plainly that
animation and skinning are the two glTF features it does not implement.

## What it owes to mesh2splat

[mesh2splat](https://github.com/electronicarts/mesh2splat), Electronic Arts'
mesh-to-gaussian converter, is the algorithm behind `lrt mesh2splat` and the
`Mesh2Splat` AOFX plugin under `plugins/mesh2splat`. It is **BSD-3-Clause**
(Copyright (c) 2024-2025 Electronic Arts Inc.), and what was taken is the
conversion itself, ported from its OpenGL pipeline -- a vertex, geometry and
fragment shader -- into one Slang compute kernel:

- a triangle is projected onto the plane its normal points along least, with
  the position taken relative to the model's box and over the wider of that
  plane's two ranges (their `orthogonalUvs`);
- the Jacobian of that map to space, `J = V (O)^-1`, gives the two sizes:
  `|Ju| * sigma / resolution` and `|Jv| * sigma / resolution`, so a gaussian
  is as wide as one cell of the grid the triangle is drawn on;
- the frame is the triangle's longest edge, its normal, and the third axis
  square to both;
- a gaussian is written for every cell the triangle covers, sampling albedo,
  normal and metallic-roughness there, within a budget.

Two of those differ here and the reasons are in `docs/decisions.md`: the flat
axis is a **fraction** of the other two rather than their `1e-7`, which is a
length and so makes how thin a gaussian is depend on how big the model is;
and the cells are counted and then written at an offset rather than appended
with an atomic, so that the same mesh gives the same array and a gaussian can
be followed from one pose of an animation to the next.

The copyright notice and the three conditions are at the head of
`plugins/mesh2splat/mesh2splat.slang`, which is the file the algorithm lives
in. No code was copied: their shaders are GLSL and this is Slang, their
density comes from a rasteriser and here the cells are walked. What is not
theirs is marked in that file -- transmission, which their conversion has no
channel for, and which a gaussian answers with a tint rather than a lens.

EA's name and marks are not used to endorse anything here.

## Status

Verified on an Apple M5 Pro (Metal). Not verified:

- CUDA, OptiX and Vulkan: the Linux paths are written but have not run.
- The OptiX pipeline route of the ray tracer: not written.
- Windows.
- PTP on Linux.

Why things are the way they are, with the measurements, is in
[docs/decisions.md](docs/decisions.md).
