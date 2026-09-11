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
- **OpenUSD 26.08** in `~/tools/usd-26.08`: `scripts/build-usd.sh` builds it
  once per machine.
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

## Status

Verified on an Apple M5 Pro (Metal). Not verified:

- CUDA, OptiX and Vulkan: the Linux paths are written but have not run.
- The OptiX pipeline route of the ray tracer: not written.
- Windows.
- PTP on Linux.

Why things are the way they are, with the measurements, is in
[docs/decisions.md](docs/decisions.md).
