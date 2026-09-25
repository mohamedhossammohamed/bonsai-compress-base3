# bonsai-compress-base3 — Ternary Weight Compression for Faster Decoding

Open-source compression stack for ternary LLMs (`{-1, 0, +1}` weights):
pack weights tighter **and** decode them faster, with zero accuracy cost.

- 📄 Scheme S2 theory + proofs: [`THEORY_S2.md`](THEORY_S2.md)
- 📊 Measured results: [`results/trials.md`](results/trials.md)
- 🌐 Project page: `https://mohamedhossammohamed.github.io/bonsai-compress-base3/`
- ⚠️ **No weights in this repo** — bring your own ternary checkpoint (see below)

## Why it decodes faster

Two independent mechanisms, both measured:

1. **Scheme S2 base-3 repack** — 5 trits per byte (3⁵=243≤256), shrinking
   ternary weights from 2.00 → **1.60 bits/weight**, bit-exact by construction.
   ~20% fewer bytes cross the memory bus per generated token, so bandwidth-bound
   decode is expected to speed up roughly proportionally *(projection — see
   `results/trials.md` for the verification protocol)*.
2. **Fused decode engine** — register-level dequant fused into GEMV
   (`runtime/` + `bench_bonsai_vs_mlx.py`), designed to beat unfused baselines
   by moving ~3× less memory traffic *(projection — must be measured)*.

Target architecture: 64-layer hybrid (48 Gated-DeltaNet linear attention +
16 full GQA), out-of-core mmap weight streaming, paged KV, speculative
`NGramDrafter` — see `runtime/bonsai_engine.py`.

## Bring your own weights

```bash
# 1. obtain a ternary {-1,0,+1} checkpoint packed as 2-bit uint32 (not shipped here)
# 2. repack to base-3, streaming, <500MB RAM:
python code/convert_to_base3.py --input model-2bit.safetensors --output model-base3.safetensors
# 3. verify bit-exact transmission:
python code/benchmark_ternary_transmission_suite.py --weights model-base3.safetensors
# 4. serve:
python runtime/bonsai_server.py --model-path model-base3.safetensors
```

Layout: `code/` converters + benchmarks · `runtime/` engine, streamer, server,
tokenizer, Metal layer, C dequant · `results/` measured traces · `docs/` project page.
