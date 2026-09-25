# Measured trials — base-3 ternary compression (Sept 2026, Apple Silicon)

Raw traces: `benchmark_bonsai_vs_mlx.json`, `bonsai_*_trace.json`, `RELEASE_v1.3.1.md`.
Two separate mechanisms — do not conflate them.

## A. Scheme S2 base-3 repack (this repo's compression core)

Source: `RELEASE_v1.3.1.md` + `benchmark_ternary_transmission_suite.py`.

| Metric | Before (2-bit uint32) | After (base-3 uint8) |
|---|---|---|
| Bits per weight | 2.00 | **1.60** |
| Bonsai-27B footprint | 6.26 GB | **5.00 GB** (−1.25 GB unified memory) |
| Per-layer GEMV latency | 0.4566 ms | **0.3614 ms (1.26×)** |
| Reconstruction error | — | **zero, bit-exact** (3⁵=243≤256, arithmetic identity) |
| Converter peak RAM | — | <500 MB (tensor-by-tensor streaming) |

Packing rule: every 5 uint32 words (16 trits each) → 16 bytes holding 80 trits.
Lossless by construction — no quality evaluation needed or claimed.

## B. Fused decode engine vs MLX SDPA (serving speed)

Source: `logs/benchmark_bonsai_vs_mlx.json` (hardware GPU timestamps).

Bonsai-27B:

| Ctx | MLX tok/s | Fused tok/s | Speedup | Traffic ratio |
|---|---|---|---|---|
| 4k | — | 2341.3 | 2.73× | 3.16× |
| 8k | — | 1299.9 | 2.72× | 3.22× |
| 16k | — | 672.6 | 2.55× | 3.25× |
| 32k | — | 290.5 | 2.49× | 3.27× |
| 64k | — | 175.4 | 3.31× | 3.27× |

Bonsai-1.7B: speedups 2.99 / 2.89 / 2.16 / 2.09 / 2.46 across 4k→64k, same ~3.2× traffic cut.

## Honesty notes

- The repack is lossless math; model *quality* (perplexity, bench scores) lives in
  the upstream ternary checkpoint, which is **not** evaluated here and **not** shipped.
- Decode speedups are Apple Silicon + MLX-baseline specific; other GPUs/backends will differ.
- No videos are included — previously rendered comparison clips were found inaccurate
  and have been withdrawn, not replaced.
