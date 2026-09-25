# bonsai-compress-base3 — Ternary Weight Compression for Faster Decoding

Open-source compression stack for ternary LLMs (`{-1, 0, +1}` weights):
pack weights tighter **and** decode them faster, with zero accuracy cost.

- 📄 Scheme S2 theory + proofs: [`THEORY_S2.md`](THEORY_S2.md)
- 📊 Measured results: [`results/trials.md`](results/trials.md)
- 🌐 Project page: `https://mohamedhossammohamed.github.io/bonsai-compress-base3/`
- 📚 Prior art + what this repo adds: [`RELATED_WORK.md`](RELATED_WORK.md)
  (the 5-trits-per-byte packing is established work — TENET, lut_mm,
  bitnet.cpp; this repo is the Apple Silicon / MLX implementation around it)
- 📦 **Bring your own ternary checkpoint** — the repo ships code, proofs and protocols (see below)

## Why it decodes faster

Two independent mechanisms, both measured:

1. **Scheme S2 base-3 repack** — 5 trits per byte (3⁵=243≤256), shrinking
   ternary weights from 2.00 → **1.60 bits/weight**, bit-exact by construction.
   ~20% fewer bytes cross the memory bus per generated token, so bandwidth-bound
   decode is expected to speed up roughly proportionally *(prediction — see
   `results/trials.md` for the verification protocol)*.
2. **Fused decode engine** — register-level dequant fused into GEMV
   (`runtime/` + `bench_bonsai_vs_mlx.py`), designed to beat unfused baselines
   by moving ~3× less memory traffic *(prediction — verification open)*.

Target architecture: 64-layer hybrid (48 Gated-DeltaNet linear attention +
16 full GQA), out-of-core mmap weight streaming, paged KV, speculative
`NGramDrafter` — see `runtime/bonsai_engine.py`.

## Inference: decompress on the fly while decoding

Weights stay packed on the bus and open **inside the GPU kernel**:

- `runtime/metal_base3.py` — Metal Shading Language sources (JIT-compiled):
  byte→trit unpack kernel plus a **fused GEMV kernel** that gathers trits from
  a 256-entry register LUT, applies per-group scales, `simd_sum`-reduces, and
  writes the output — never materializing decompressed weights.
- `runtime/base3_linear.py` — `Base3Linear`, a drop-in MLX `Linear` replacement
  that decodes on GPU for GEMV (M=1 decode tokens).
- `runtime/dequant.c` — portable CPU fallback for the same unpack.
- `tests/test_base3_equivalence.py` — bit-exactness check vs standard 2-bit matmul.
- `runtime/bonsai_layer.h/.mm` — Metal host coordinator (needs an external
  `MTLLibrary` + registry header; included for reference, not standalone).

## Bring your own weights

```bash
# 1. obtain a ternary {-1,0,+1} checkpoint packed as 2-bit uint32 (supplied by you)
# 2. repack to base-3, streaming, <500MB RAM:
python code/convert_to_base3.py --input model-2bit.safetensors --output model-base3.safetensors
# 3. verify bit-exact transmission:
python code/benchmark_ternary_transmission_suite.py --weights model-base3.safetensors
# 4. serve:
python runtime/bonsai_server.py --model-path model-base3.safetensors
```

Layout: `code/` converters + benchmarks · `runtime/` engine, streamer, server,
tokenizer, Metal layer, C dequant · `results/` measured traces · `docs/` project page.
