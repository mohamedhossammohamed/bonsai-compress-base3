# MZSAE & Model Weight Quantization Guide

This document explains how **MZSAE** operates seamlessly across all model weight quantization formats—from full precision (FP16/BF16) down to 4-bit (Q4_K_M, AWQ), 2-bit (Q2_K), and 1.58-bit Ternary (BitNet b1.58)—with zero model downloads required for verification.

---

## 1. Architectural Orthogonality: Why MZSAE Supports All Quantizations

A foundational principle of autoregressive transformer inference is that **model weight quantization** and **KV-cache compression** operate at completely orthogonal layers in the execution pipeline:

```
┌────────────────────────────────────────────────────────┐
│ 1. Model Weights (q_proj, k_proj, v_proj, FFN)         │
│    Quantization handled by: GGUF, llama.cpp, AWQ,      │
│    BitsAndBytes, BitNet, etc.                          │
│    Format: FP16, BF16, INT8, Q4_K_M, Q2_K, Ternary     │
└──────────────────────────┬─────────────────────────────┘
                           │ Matrix Multiplication (GEMM)
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Post-Projection Key / Value Activations             │
│    Keys & Values produced in activation space          │
│    Shape: [batch, seq_len, num_kv_heads, head_dim]    │
└──────────────────────────┬─────────────────────────────┘
                           │ Ingestion & Compression
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. KV-Cache Compression & Policy Layer: MZSAE          │
│    - Plane 1: 2-bit Centroid-Residual Quantization     │
│    - Plane 2: 64-byte Cauchy-Schwarz Sentinels         │
│    - Policy: Neuromorphic TD Eviction & Directional Veto│
└──────────────────────────┬─────────────────────────────┘
                           │ Fused Selective Decode
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Attention Computation: Metal / CPU Reference / CUDA │
│    Evaluates active blocks bounded by sentinels        │
└────────────────────────────────────────────────────────┘
```

Because MZSAE receives Key and Value vectors **after** they have been projected by the linear layers, the internal representation of the weights (whether stored as 4-bit indices in DRAM or unpacked on-the-fly into FP16 in registers) is irrelevant to MZSAE's cache format.

---

## 2. Supported Weight Formats Matrix

| Model Weight Format | Bits / Weight | Typical Source / Engine | MZSAE Compatibility | Special Adaptation Needed |
|---------------------|---------------|-------------------------|---------------------|---------------------------|
| **FP16 / BF16**     | 16            | PyTorch, HuggingFace    | ✅ Out of the box   | None (`scale = 1.0`, $\tau = 16.0$) |
| **8-bit (Q8_0 / INT8)** | 8         | llama.cpp, BitsAndBytes | ✅ Out of the box   | None (`scale = 1.0`, $\tau = 16.0$) |
| **4-bit (Q4_K_M / AWQ / GPTQ)** | 4 | llama.cpp, vLLM, AutoAWQ | ✅ Out of the box   | None (`scale = 1.0`, $\tau = 16.0$) |
| **3-bit (Q3_K_M)**  | 3             | llama.cpp               | ✅ Supported        | `dynamic_range_scale = 0.9`, $\tau \times 0.9$ |
| **2-bit (Q2_K)**    | 2             | llama.cpp, QuIP#        | ✅ Supported        | `dynamic_range_scale = 0.8`, $\tau \times 0.8$ |
| **Ternary (BitNet b1.58)** | 1.58 (2) | Microsoft BitNet        | ✅ Supported (Custom) | `dynamic_range_scale = 0.5`, $\tau \times 0.7$, `sentinel_scale = 0.5` |

---

## 3. Deep Dive: Ternary BitNet b1.58 Adaptation

### 3.1 The Biological & Statistical Phenotype of BitNet
BitNet b1.58 replaces traditional floating-point matrix multiplications with ternary weights $W \in \{-1, 0, +1\}$. Because projections are sums and differences of activations:
1. **Tighter Dynamic Range**: The variance of post-projection Key activations $\sigma_K^2$ is roughly $\approx 50\%$ smaller than in standard FP16 or INT4 models.
2. **Narrower Attention Logit Dispersion**: Query-Key dot products produce tighter logit distributions.
3. **Pruning Coarseness Risk**: A standard pruning threshold $\tau = 16.0$ would be too coarse, causing MZSAE to prune too aggressively or fail to discriminate critical context.

### 3.2 MZSAE Adaptation Formulae
To achieve optimal retrieval fidelity on BitNet b1.58 models:

$$\tau_{\text{ternary}} = \tau_{\text{base}} \times 0.7 = 16.0 \times 0.7 = 11.2$$

$$\mu_{\text{centroid}} = \text{mean}(K_{\text{block}}) \times 0.5$$

$$R_{\Delta, \text{ternary}} = R_{\Delta} \times 0.5$$

These adaptations are automatically applied when using `ModelWeightConfig.from_type("ternary")` or `load_config("bitnet_b158")`.

---

## 4. Usage & Integration

### 4.1 PyTorch Module (`MZSAEAttention`)

```python
import torch
from mzsae import MZSAEAttention

# 1. For a standard 4-bit quantized model (e.g. Llama-3-8B Q4_K_M)
attn_4bit = MZSAEAttention(
    embed_dim=4096,
    num_heads=32,
    num_kv_heads=8,
    model_weights="4bit",
)

# 2. For a BitNet b1.58 ternary model
attn_ternary = MZSAEAttention(
    embed_dim=2048,
    num_heads=16,
    num_kv_heads=4,
    model_weights="ternary",
)

# Forward pass (layer execution)
x = torch.randn(1, 128, 2048)
out = attn_ternary(x, causal=True)
```

### 4.2 Loading via YAML Profile

```python
from mzsae import load_config, MZSAEEngine

# Load pre-configured BitNet profile
config = load_config("bitnet_b158")
engine = MZSAEEngine(config=config)
```

### 4.3 Explicit `ModelWeightConfig`

```python
from mzsae import MZSAEConfig, ModelWeightConfig

config = MZSAEConfig()
config.model_weights = ModelWeightConfig(
    quantization_type="ternary",
    quantization_bits=2,
    dynamic_range_scale=0.5,
    tau_multiplier=0.7,
    sentinel_scale=0.5,
)
```

---

## 5. Zero-Storage Testing Strategy

Downloading 4 GB to 16 GB model weights (e.g. GGUF or SafeTensors files) to test attention cache compatibility is wasteful and non-deterministic. MZSAE employs **Zero-Storage Simulation**:

1. **Synthetic Quantized Projections**:
   Linear layers with quantized weights ($\text{round}(W / s) \cdot s$) simulate hardware-quantized GEMMs.
2. **Statistical Variance Matching**:
   Synthetic activations are generated matching the empirically measured variance of quantized layers ($\sigma = 1.0$ for FP16, $\sigma = 0.8$ for 2-bit, $\sigma = 0.5$ for BitNet).
3. **Deterministic Mathematical Verification**:
   The Cauchy-Schwarz sentinel upper bounds ($|q^T k| \le s_{\text{slow}} + R_{\Delta} \|q\|_{\text{fast}}$) are evaluated across hundreds of synthetic blocks to guarantee 100% strictness before any real weights are loaded.
