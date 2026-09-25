# Release v1.3.1 — BonsAI v2 27B Dedicated Engine & Scheme S2 Base-3 Lossless Compression

We are pleased to announce **MZSAE v1.3.1**, introducing dedicated serving and acceleration infrastructure for **BonsAI v2 27B Multimodal Vision Model** alongside **Scheme S2 (Packed Base-3 Arithmetic Encoding)** on Apple Silicon (M-Series) GPUs.

---

## 1. Scheme S2: Lossless Packed Base-3 Weight Compression (1.60 b/w)

Scheme S2 is an arithmetic compression and register-decode engine designed specifically for ternary-quantized weights ($\{-1, 0, +1\}$).

- **High-Density Packing**: Packs 5 ternary weights into exactly 1 byte ($3^5 = 243 \le 256 = 2^8$), reducing DRAM weight bandwidth from **2.00 b/w to 1.60 b/w**.
- **Bit-Exact Fidelity**: Mathematically verified bit-exact equivalence with standard 2-bit affine quantized tensors ($\cos = 1.000000$, $\max |Y_{\text{ref}} - Y_{\text{base3}}| = 0.000000$).
- **Apple Silicon JIT Metal Kernel**: Direct register-level decoding via a constant lookup table (`BASE3_LUT`) streaming at **12.06 GB/s**.
- **Memory Footprint Reduction**: Reduces the physical weight footprint of BonsAI 27B from **6.26 GB to 5.00 GB** (saving 1.25 GB of unified memory).
- **Decode Acceleration**: Per-layer GEMV latency reduced from **0.4566 ms to 0.3614 ms** ($1.26\times$ speedup).

Detailed technical documentation and mathematical proofs: [`docs/scheme_s2_base3_compression.md`](docs/scheme_s2_base3_compression.md).

---

## 2. BonsAI v2 27B Dedicated Serving Engine

This sub-release equips `Ternary-Bonsai-2-27B-mlx-2bit` with an end-to-end, production-grade serving stack:

1. **Native Hybrid Architecture Execution**:
   - Seamless handling of 64 layers: 48 Gated Delta Net (GDN) linear attention layers + 16 Full GQA layers with Fast Walsh-Hadamard Transform (FWHT) activations.
2. **Context Guard for 16 GB Apple Silicon Macs**:
   - Strict 10.5 GB active MLX memory ceiling preserving host OS stability.
   - 4-bit quantized KV caching (`kv_bits=4`) and chunked prefill (`prefill_step_size=256`).
3. **OpenAI-Compatible Streaming Gateway (`src/mzsae/server.py`)**:
   - Full support for `GET /health`, `GET /v1/models`, and streaming SSE `POST /v1/chat/completions`.
   - Vision and image-path extraction pipeline for multimodal prompt completions.

---

## 3. Empirical Benchmarks (Apple Silicon M-Series)

### Scheme S2 Single-Layer Equivalence & Speedup (Layer 0 `linear_attn.in_proj_qkv`, $M=10240, K=5120$)

| Metric | Baseline 2-Bit MLX (2.00 b/w) | Scheme S2 Base-3 (1.60 b/w) | Delta / Speedup |
| :--- | :---: | :---: | :---: |
| **Storage / DRAM Footprint** | 12.50 MB | 10.00 MB | **-20.0% (-2.50 MB)** |
| **Total 27B Quantized Weights** | 6.256 GB | 5.005 GB | **-1.251 GB (-20.0%)** |
| **GEMV Layer Latency** | 0.4566 ms | 0.3614 ms | **1.26x Speedup** |
| **Numerical Error ($\max \|Y_{\text{ref}} - Y_{\text{base3}}\|$)** | 0.0 | **0.00000000e+00** | **Bit-Exact Match** |
| **Cosine Fidelity ($\cos$)** | 1.000000 | **1.00000000** | **Bit-Exact Match** |
| **JIT Unpack Speed** | N/A | **12.06 GB/s** | Zero-stall streaming |

---

## 4. Verification Suite

Run the automated bit-exact validation test on any Apple Silicon device:
```bash
./.venv_transmission/bin/python tests/test_base3_equivalence.py \
    --model-path /path/to/Ternary-Bonsai-2-27B-mlx-2bit/model.safetensors
```
