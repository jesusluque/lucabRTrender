# Splat fixtures

| File | What it is |
|---|---|
| `tiny.ply` | 64 splats, degree 0. From openFXplayer's `tests/data`. |
| `tiny.sog` | `tiny.ply` through PlayCanvas's splat-transform v3.2.0: SOG version 2, no higher harmonics. From openFXplayer's `tests/data`. |
| `sh3.ply` | 64 splats with degree-3 harmonics, random values (Python, seed 1234). Written for the SOG harmonics palette. |
| `sh3.sog` | `sh3.ply` through `npx @playcanvas/splat-transform@3.2.0 sh3.ply sh3.sog`: SOG version 2 with a 64-entry harmonics palette: one entry a splat, so the only loss is the 8-bit codebook. |

The `.sog` files are the reference converter's output, so reading them is a
compatibility check rather than a self-consistency one. The converter
reorders splats spatially and clusters harmonics into a palette, so the
tests compare renders rather than splat-by-splat values.
