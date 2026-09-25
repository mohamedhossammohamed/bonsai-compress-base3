# Related work — what this repo builds on, and what it adds

This repo claims **no invention of the underlying packing identity**. The fact
that five ternary weights fit in one byte (3⁵=243≤256, i.e. 1.6 bits/weight)
is established prior art. What this repo contributes is a complete,
open-source **Apple Silicon / MLX implementation and serving stack** around it.
Credit where it is due:

## Ternary models (training)

- **BitNet b1.58** (Wang et al., 2023; Ma et al., 2024) — quantization-aware
  training with {-1,0,+1} weights; the reason ternary checkpoints exist at all.
  Ternary models are trained, not converted post-hoc.

## Ternary inference engines

- **bitnet.cpp** (Wang et al., ACL 2025) — CPU inference for ternary LLMs with
  TL (ternary lookup) and I2_S kernels; documents llama.cpp's TQ1_0/TQ2_0
  formats (1.69 / 2.06 bpw).
- **llama.cpp TQ formats** — production ternary inference baselines this work
  is complementary to (CPU/llama.cpp vs our Metal/MLX focus).

## The 1.6-bit packing identity itself

- **TENET** (2025, edge ternary inference) — states the same construction
  explicitly: one 8-bit index expresses five ternary weights (3⁵=243<256),
  1.6 bits each, decoded by a dedicated decompression unit.
- **lut_mm** (open-source x86 ternary GEMM) — same 5-trits-per-byte packing
  with LUT-based dot products (notes BitNet's TL2 at 1.667 bpw for contrast).

## Architecture components used by the serving stack

- **Gated DeltaNet** (Yang et al.) — the linear-attention layer family.
- **Paged KV / PagedAttention** (vLLM, Kwon et al.) — paged KV buffer design.
- **Speculative decoding** (Leviathan et al.; Chen et al.) — draft-and-verify
  pattern behind `NGramDrafter`.

## A note on independent derivation

The packing construction in this repo was derived independently in the course
of this project — the idea was directed here, the parameters were tuned with
project-run statistical fitting code, and the implementation plus all
measurements are original work (committed September 2026). It converges with
the published results above (TENET 2025, lut_mm, bitnet.cpp), which predate
it: this is convergent discovery, honestly stated. Priority on the identity
belongs to the earlier publications; the derivation path, the MLX/Metal
implementation, the serving stack, and every logged number here are ours.

## What this repo adds (the honest delta)

1. A **streaming 2-bit-uint32 → base-3 repacker** for MLX-format ternary
   checkpoints (`code/convert_to_base3.py`, <500MB RAM on 27B).
2. **Fused Metal GEMV kernels** for this layout on Apple Silicon plus a
   drop-in MLX `Linear` (`runtime/metal_base3.py`, `runtime/base3_linear.py`).
3. An **out-of-core serving stack** (streamer, paged KV, server) that runs a
   27B ternary model within a 10.5GB ceiling on 16GB Macs.
4. A **verification protocol** (`results/trials.md`) so future numbers are
   reproducible instead of asserted.

If any prior work above already published one of these four for this exact
stack, open an issue — attribution will be corrected.
