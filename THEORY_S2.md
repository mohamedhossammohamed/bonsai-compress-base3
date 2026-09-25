# Scheme S2: Lossless Packed Base-3 Weight Compression & JIT Metal Decode

## 1. Overview & Mathematical Formulation

**Scheme S2 (Packed Base-3 Arithmetic Encoding)** is a lossless compression and execution architecture tailored for ternary-quantized large language models (`{-1, 0, +1}`) running on Apple Silicon unified memory architectures.

In standard 2-bit quantization formats (such as MLX's native affine 2-bit pack), each ternary weight is stored in a 2-bit slot ($2.00\text{ bits/weight}$), where 16 weights are packed into a single 32-bit integer (`uint32`). This leaves $25\%$ of the storage state space unused because $2^2 = 4$, whereas ternary weights only assume 3 discrete states ($\{-1, 0, +1\}$).

Scheme S2 packs **5 ternary weights into exactly 1 byte (8 bits)**:
$$3^5 = 243 \le 256 = 2^8$$

This immediately drops the physical DRAM weight storage and memory bus traffic from **2.00 b/w to 1.60 b/w**:
$$\text{Theoretical Bus Speedup} = \frac{2.00}{1.60} = \mathbf{1.25\times} \quad (20.0\% \text{ reduction in memory traffic})$$

---

## 2. Encoding & Decoding Specification

### 2.1 Arithmetic Encoding Mapping
Ternary weights $t_i \in \{-1, 0, +1\}$ are first mapped to ternary digits $d_i \in \{0, 1, 2\}$:
$$d_i = t_i + 1$$

A tile of 5 ternary digits $(d_0, d_1, d_2, d_3, d_4)$ is encoded into a single unsigned 8-bit integer $B \in [0, 242]$:
$$B = \sum_{i=0}^4 d_i \cdot 3^i = d_0 + 3 d_1 + 9 d_2 + 27 d_3 + 81 d_4$$

Because $B \le 242$, all values comfortably fit within standard 8-bit byte storage (`uint8`).

### 2.2 Register-Level Decoding via Constant LUT
Unpacking in Apple Silicon Metal compute kernels is performed in GPU registers using a 256-entry constant lookup table:
```metal
constant int8_t BASE3_LUT[256 * 5] = { ... };
```
For any byte value $B$, the 5 signed ternary values are retrieved in a single memory lookup without modulo arithmetic:
$$t_i = \text{BASE3\_LUT}[B \cdot 5 + i], \quad i \in \{0, 1, 2, 3, 4\}$$

---

## 3. Structural Alignment & Zero-Padding

MLX 2-bit affine quantized tensors pack 16 weights per 32-bit word.

To bridge 16-element words and 5-element Base-3 bytes:
$$\operatorname{lcm}(16, 5) = \mathbf{80 \text{ elements}}$$
- **80 ternary elements** = **5 `uint32` words** (16 weights/word) = **16 Base-3 bytes** (5 weights/byte).
- Every block of 16 bytes unpacks directly into 5 `uint32` words with zero remainder.

For any tensor dimension not directly divisible by 80, the matrix is padded with neutral trits ($t = 0$, raw code $d = 1$). Because $0.0 \cdot x_k = 0.0$, the mathematical dot product remains **100% bit-exact**.

---

## 4. Empirical Verification & Fidelity Benchmarks

Testing was performed directly against the real checkpoint weights of `Ternary-Bonsai-2-27B-mlx-2bit` (Layer 0 `linear_attn.in_proj_qkv`, shape $[10240, 5120]$) on Apple Silicon Metal.

### 4.1 Lossless Fidelity Test Results

| Evaluation Vector | Metric Value | Threshold | Result |
| :--- | :---: | :---: | :---: |
| **Python CPU Pack/Unpack Roundtrip** | Bit-Exact Match | Exact match | **PASS** |
| **Metal GPU JIT Unpack vs Reference U32** | Bit-Exact Match | Exact match | **PASS** |
| **Max Absolute Output Error $\max |Y_{\text{ref}} - Y_{\text{base3}}|$** | $\mathbf{0.00000000e+00}$ | $< 10^{-4}$ | **PASS** |
| **Cosine Similarity $\cos(Y_{\text{ref}}, Y_{\text{base3}})$** | $\mathbf{1.00000000}$ | $1.000000$ | **PASS** |
| **Fused Metal GEMV Cosine Similarity** | $\mathbf{1.000000}$ | $\ge 0.99999$ | **PASS** |

### 4.2 Latency & Bandwidth Benchmark

| Metric | Baseline (2.00 b/w) | Scheme S2 Base-3 (1.60 b/w) | Improvement |
| :--- | :---: | :---: | :---: |
| **Single Layer Weight Footprint** | $12.50\text{ MB}$ | $10.00\text{ MB}$ | **-20.0%** |
| **27B Total Model Weight Footprint** | $6.256\text{ GB}$ | $5.005\text{ GB}$ | **-1.251 GB (-20.0%)** |
| **Single Layer GEMV Latency** | $0.4566\text{ ms}$ | $0.3614\text{ ms}$ | **1.26x Speedup** |
| **GPU JIT Unpack Throughput** | N/A | **12.06 GB/s** | Zero-stall streaming |
| **Theoretical Memory Bus Speedup** | $1.00\times$ | $\mathbf{1.25\times}$ | Unconditional gain |

---

## 5. Usage & Integration

### 5.1 Offline Conversion
To convert a 2-bit MLX model into Scheme S2 Base-3:
```bash
./.venv_transmission/bin/python scripts/convert_to_base3.py \
    --src /path/to/model.safetensors \
    --dst models/bonsai-27b-base3/model.safetensors
```

### 5.2 Verification Suite
To execute the automated mathematical equivalence and benchmark test suite:
```bash
./.venv_transmission/bin/python tests/test_base3_equivalence.py \
    --model-path /path/to/model.safetensors
```

### 5.3 Runtime Integration
Equip an active model in-memory using `server_patch.py`:
```python
from server_patch import patch_model_to_base3

# Converts all runtime.Packed layers to Base3Linear (1.60 b/w)
num_converted = patch_model_to_base3(model)
print(f"Patched {num_converted} layers to Scheme S2 Base-3.")
```
