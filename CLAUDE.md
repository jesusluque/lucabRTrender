# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug
ctest --preset macos-arm64-debug                          # all tests, one at a time (they share the GPU)
ctest --test-dir build/macos-arm64-debug -R lod           # tests matching a name
build/macos-arm64-debug/bin/lrt_lod_tests "chunks*"       # one Catch2 case by name (or a [tag])
cmake --build build/macos-arm64-debug --target lrt_render_tests   # one test binary
```

- **Test binaries** are `lrt_<area>_tests` (gpu, scene, render, geom,
  technique, lod, usd, gpu_host, aofx, sched), from `tests/<area>/`.
  `lrt_storm_oracle_tests` compares Hydra outputs with Storm's; it needs
  `HDX_MSAA_SAMPLE_COUNT=1` in the environment, which ctest sets.
- **Timings** come from the release preset's `lrt bench`.
- **Checking a shader compiles** without a build:
  `~/tools/slang/bin/slangc shaders/lrt/<dir>/<file>.slang -I shaders -target metal -entry <entry> -stage compute -o /dev/null`.
  Shaders are copied to `build/<preset>/shaders` by the build and compiled at
  run time, so a shader-only change needs the copy step (any build) but no
  C++ rebuild.

## Rules this codebase holds to

- **No CPU arithmetic on data.**
  - No CPU reference renderer, no CPU fallback, no CPU test oracle.
  - The CPU reads files, parses headers, decompresses, and does bookkeeping
    (counts, slots, queues).
  - Decoding, sorting, merging, culling and image comparison are compute
    kernels.
  - Tests generate inputs and check results with kernels. They read back only
    counters or `compareImages` metrics (p99, max, over2) against a GPU
    reference (`ReferenceRenderer`).
- **Shader parameters are bound by name** through reflection
  (`cursor["name"].setBinding(...)`, `ComputeKernel::dispatch`), never by
  slot.
- **Errors** are `Result<T>` with `LRT_TRY`, no exceptions. OS calls live only
  in `modules/core/Platform`, which is the Windows port's starting point.
- **`docs/decisions.md`** records each subsystem's design, what was measured
  and what is not done. Update it with the change.
- **Submodules.** `third_party/gpe` is on branch `lrt-fixes` and genlock on
  `main`. gpe changes are committed in the submodule.
- **aofx compatibility is mandatory.** Its SDK and host change only
  additively and only following openFXplayer's ABI. `aofx_sdk_manifest` fails
  on any header change, and `lrt_aofx_tests` must stay green.
- **One Slang, one slang-rhi, one TBB** in the process. `single_tbb` checks
  the TBB count.
- **Toolchain.** OpenUSD with MaterialX/OpenVDB is built by
  `scripts/build-usd.sh` into `~/tools/usd-26.08-mx`, and OIDN (GPU devices
  only) by `scripts/build-oidn.sh`.
- **Roadmap.** The plan for complete USD (milestones M0–M11) is summarised in
  `docs/decisions.md`, one section per milestone.

## Architecture

Modules under `modules/<name>` are static libraries `lrt::<name>`. They are
listed in dependency order in `modules/CMakeLists.txt`, and each links only
the ones above it.

| Module | What it holds |
|---|---|
| core | `Result`, logging, Platform |
| sched | `FrameClock`: genlock PTP and ST 2059-1 alignment, timecode |
| io | CPU file readers (PLY header, SPZ, SOG zip/WebP), EXR with attributes |
| gpu | slang-rhi device, `ShaderLibrary`, `ComputeKernel`, `CommandBatch`, `Buffer`; `gpu/algo` for PrefixSum and RadixSort |
| gpu_host | gpe adopting slang-rhi's device: one `MTLDevice` or CUDA context, buffers shared without copies |
| scene | `CloudLoader`: raw records uploaded, decoded on the GPU into `GpuSplats` / `GpuPoints` |
| render | `TileRasterizer`, `GaussianRayTracer`, `PointRasterizer`, `ReferenceRenderer`, `Camera`/`Projection`, `SplatEdit` |
| geom | `MeshBuilder`: Hydra meshes triangulated, smooth-normalled and their primvars expanded on the GPU, in `HdMeshUtil`'s order |
| world | `GpuScene` (vertex/index/primvar pools, instance records), `Instancing` (Hydra instancer chains), `RayTracingScene` (BLAS/TLAS), `BvhScene` (two-level compute LBVH) |
| technique | how a frame is drawn: `VisibilityRaster` / `VisibilityTrace` / `VisibilityBvh` (same ids), `HeadlightShading`, `AovShading`, `Denoiser` (OIDN on the engine's own Metal queue) |
| lod | `LodBuilder`, `CutSelector`; `Lrtc.h` for the `.lrtc` reader/writer and `StreamingPool` |
| usd | `Engine`, `StageRenderer`, `Export`; the `hdLrt` plugin; codeless schemas in `modules/usd/schemas` |
| aofx | openFXplayer's plugin SDK (ABI 22) and host |

How the pieces fit:

- **The GPU cloud layout** is shared by every renderer, the loaders and the
  LOD. `GpuSplats` holds `positions` (float4), `shape` (4 uint per splat,
  packed opacity, scale, quaternion and DC) and `sh` (`shWords` uint per
  splat, f16). The packing is in `shaders/lrt/common/packing.slang`.
- **Instances** (`SplatInstance`: cloud, objectToWorld, edit) are what every
  renderer takes. `CutSelector::select` turns `LodInstance`s into per-frame
  `SplatInstance`s whose clouds it owns.
- **Shaders** mirror the modules under `shaders/lrt/`: common, algo, scene
  (decode), splat, rt, points, reference, lod, geom, world, technique, usd. The ray tracer's two routes
  share `rt/rt_integrate.slang`.
- **Hydra.** `Sync` (any thread) only hands CPU records to `Engine` under a
  lock. The render pass thread commits (uploads, opens `.lrtc`) and renders,
  so the device has one caller.
- **Meshes.** A visibility buffer holds (instance + 1, triangle) per pixel.
  Shading and AOVs rebuild the hit from it (`technique/surface.slang`), so all
  three visibility routes shade alike. Meshes and points are composited by
  view z and handed to the splat rasteriser as its opaque `under` layer.
  Hydra render buffers are bottom row first, as Storm's.
- **Levels of detail.**
  - Splats are sorted by Morton code, and an octree level's cells are runs of
    that order.
  - The finest merged level decides for its splats. Splats live in chunks
    that may be absent from the device, and a group whose chunks are missing
    draws its merged Gaussian.
- **The CLI** is `apps/lrt` (CLI11): `CmdRender` (render/bench),
  `CmdStage` (convert/stage), `CmdAofx`, `CmdLive`, `CmdInfo`.
