# Related work

This project applies established ternary-compression research to one exact
target — 27B ternary checkpoints in MLX 2-bit format, served on Apple Silicon —
with parameters tuned for that model. The foundations:

- **BitNet b1.58** (Wang et al., 2023; Ma et al., 2024) — ternary
  quantization-aware training; why ternary checkpoints exist.
- **bitnet.cpp** (Wang et al., ACL 2025) — CPU ternary inference (TL / I2_S
  kernels); llama.cpp's TQ1_0/TQ2_0 formats.
- **TENET** (2025) — the 5-weights-per-byte, 1.6-bit packing construction with
  dedicated decompression.
- **lut_mm** — the same packing with LUT-based GEMM on x86.
- **Gated DeltaNet** (Yang et al.), **PagedAttention** (vLLM), and draft-and-
  verify **speculative decoding** — architecture components in the serving stack.

## What this repo delivers

The base-3 packing tuned and packaged end-to-end for this model family:
streaming repacker (`code/convert_to_base3.py`), fused Metal GEMV kernels and
drop-in MLX layer (`runtime/metal_base3.py`, `runtime/base3_linear.py`),
out-of-core 27B serving within a 10.5GB ceiling (`runtime/`), and a
verification protocol (`results/trials.md`) for reproducible numbers.
