# What the mathematics establishes — and the roadmap to measured numbers

This page states exactly what theory guarantees today, plus the protocol that
turns each prediction into a logged result.

## What theory predicts (arithmetic, not benchmarks)

| Quantity | Derivation | Status |
|---|---|---|
| Bits / weight | 5 trits per byte, 3⁵=243≤256 → 8/5 = **1.60** (from 2.00) | identity, certain |
| Byte ratio vs 2-bit packing | 1.60/2.00 = **0.80× bytes** for the same weights | identity, certain |
| Reconstruction error | byte↔trit mapping is bijective on 243 values | **zero by construction** |
| Decode latency direction | GEMV decode is memory-bandwidth bound → ~20% fewer bytes implies faster, roughly proportionally | **prediction → verifying** |
| Engine speedups | fused dequant + traffic cuts should beat unfused baselines | **prediction → verifying** |

The "prediction" rows become logged results once the protocol below is run.

## How to verify each claim (verification protocol)

**V1 — Size (trivial, deterministic).**
Run `code/convert_to_base3.py` on any 2-bit-packed ternary checkpoint.
Pass = `output_bytes == ceil(input_trits / 5)` and `output/input ≈ 0.80`.
No GPU needed. Report: both byte counts, tensor count, wall time, peak RSS.

**V2 — Losslessness (trivial, deterministic).**
Run `code/benchmark_ternary_transmission_suite.py --weights <base3>`.
Pass = **zero mismatched trits across every tensor**. Any nonzero count is a
bug, not noise. No GPU needed.

**V3 — Per-layer decode latency (needs Apple Silicon + weights).**
Time the same GEMV shapes in both formats, same process, back-to-back:
warmup ≥ 50 iters, then median of ≥ 200 iters using hardware GPU timestamps
(not wall clock around the launch). Fix power state (plugged in, no other
load), pin threadgroup config, always report distributions (median + p95 + N).
Compare FLOP-identical shapes; the only delta must be the weight format.

**V4 — End-to-end serving (needs Apple Silicon + weights + MLX).**
`code/bench_bonsai_vs_mlx.py` across 4k/8k/16k/32k/64k contexts. Same machine,
same OS build, same commit, back-to-back A/B, ≥ 3 repetitions each; report
tok/s with spread, plus measured bytes moved (not modeled). Baselines must be
named with versions (MLX version, OS, chip).

**The bar for a logged result:** repeated runs, named baselines, hardware
timestamps around GPU work, and mechanisms reported separately. The protocol
above exists so every number committed here is reproducible.

## Logging a verified result

A result graduates from "prediction" to "measured" with: the completed
V-protocol, full stdout log, hardware + software versions, commit hash, and
the exact command lines — committed under `results/measured/` with all four.
