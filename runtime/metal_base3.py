"""
metal_base3.py
High-performance Apple Silicon Metal kernels for Scheme S2 (Packed Base-3, 5 trits/byte, 1.60 b/w).

Provides:
- JIT Base-3 to MLX uint32 unpacked tensor generator.
- Fused register-level GEMV kernel for M=1 decode tokens.
"""

from __future__ import annotations

import functools
import mlx.core as mx
import numpy as np

# 256-entry precomputed Base-3 LUT
# Maps byte [0..242] to 5 signed trits in {-1, 0, +1}
_TRIT_LUT = np.zeros((256, 5), dtype=np.int8)
for b in range(243):
    v = b
    for i in range(5):
        _TRIT_LUT[b, i] = (v % 3) - 1
        v //= 3

# Maps byte [0..242] to 5 raw affine codes in {0, 1, 2}
_CODE_LUT = np.zeros((256, 5), dtype=np.uint8)
for b in range(243):
    v = b
    for i in range(5):
        _CODE_LUT[b, i] = v % 3
        v //= 3

_TRIT_LUT_STR = ", ".join(map(str, _TRIT_LUT.flatten()))
_CODE_LUT_STR = ", ".join(map(str, _CODE_LUT.flatten()))

_HEADER_UNPACK = f"""
#include <metal_stdlib>
using namespace metal;

constant uint8_t CODE_LUT[256 * 5] = {{{_CODE_LUT_STR}}};
"""

_SOURCE_UNPACK = """
    uint tid = thread_position_in_grid.x;
    if (tid >= total_tiles) return;

    uint in_offset = tid * 16;
    uint out_offset = tid * 5;

    uint8_t c[80];
    #pragma unroll
    for (int b = 0; b < 16; b++) {
        uint8_t byte_val = in_bytes[in_offset + b];
        uint lut_idx = (uint)byte_val * 5;
        c[b * 5 + 0] = CODE_LUT[lut_idx + 0];
        c[b * 5 + 1] = CODE_LUT[lut_idx + 1];
        c[b * 5 + 2] = CODE_LUT[lut_idx + 2];
        c[b * 5 + 3] = CODE_LUT[lut_idx + 3];
        c[b * 5 + 4] = CODE_LUT[lut_idx + 4];
    }

    #pragma unroll
    for (int w = 0; w < 5; w++) {
        uint32_t word = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            word |= ((uint32_t)c[w * 16 + i]) << (2 * i);
        }
        out_u32[out_offset + w] = word;
    }
"""

_HEADER_GEMV = f"""
#include <metal_stdlib>
using namespace metal;

constant int8_t BASE3_LUT[256 * 5] = {{{_TRIT_LUT_STR}}};
"""

_SOURCE_GEMV = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;

    if (row >= M) return;

    device const uint32_t* w_u32_ptr = (device const uint32_t*)(w_packed + row * K_bytes);
    device const half* scale_ptr = scales + row * num_groups;

    uint n_u32_words = K_bytes / 4;
    float acc = 0.0f;

    for (uint w = lane; w < n_u32_words; w += 32) {
        uint32_t four_bytes = w_u32_ptr[w];
        uint k_base = w * 20;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint8_t byte_val = (four_bytes >> (b * 8)) & 0xFF;
            uint lut_idx = (uint)byte_val * 5;
            uint k_sub = k_base + b * 5;

            #pragma unroll
            for (int i = 0; i < 5; i++) {
                uint k = k_sub + i;
                float s = (float)scale_ptr[k >> 7];
                float trit = (float)BASE3_LUT[lut_idx + i];
                acc += (float)x[k] * (trit * s);
            }
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        out[row] = (half)acc;
    }
"""


@functools.lru_cache(maxsize=1)
def get_unpack_kernel():
    return mx.fast.metal_kernel(
        name="unpack_base3_to_u32",
        input_names=["in_bytes", "total_tiles"],
        output_names=["out_u32"],
        header=_HEADER_UNPACK,
        source=_SOURCE_UNPACK,
    )


@functools.lru_cache(maxsize=1)
def get_gemv_kernel():
    return mx.fast.metal_kernel(
        name="gemv_base3_simd32",
        input_names=["x", "w_packed", "scales", "M", "K", "K_bytes", "num_groups"],
        output_names=["out"],
        header=_HEADER_GEMV,
        source=_SOURCE_GEMV,
    )


def unpack_base3_to_u32_gpu(b3_packed: mx.array, target_shape: tuple[int, int]) -> mx.array:
    """
    Unpacks Base-3 packed bytes into native MLX 2-bit uint32 array on GPU.
    target_shape: (M, N_words) where N_words = K // 16.
    """
    total_u32 = target_shape[0] * target_shape[1]
    total_tiles = total_u32 // 5
    kernel = get_unpack_kernel()

    threads_per_tg = 256
    grid = (((total_tiles + threads_per_tg - 1) // threads_per_tg) * threads_per_tg, 1, 1)

    total_tiles_mx = mx.array(total_tiles, dtype=mx.uint32)
    out_u32 = kernel(
        inputs=[b3_packed.reshape(-1), total_tiles_mx],
        grid=grid,
        threadgroup=(threads_per_tg, 1, 1),
        output_shapes=[target_shape],
        output_dtypes=[mx.uint32]
    )[0]
    return out_u32


def base3_gemv(x: mx.array, w_packed: mx.array, scales: mx.array, M: int, K: int) -> mx.array:
    """
    Direct single-token decode GEMV with JIT register Base-3 decoding.
    x: shape (K,) or (1, K)
    w_packed: shape (M, K // 5) uint8
    scales: shape (M, K // 128) float16
    """
    if x.ndim == 2:
        x_1d = x.reshape(-1)
    else:
        x_1d = x

    kernel = get_gemv_kernel()
    K_bytes = K // 5
    num_groups = K // 128

    M_mx = mx.array(M, dtype=mx.uint32)
    K_mx = mx.array(K, dtype=mx.uint32)
    Kb_mx = mx.array(K_bytes, dtype=mx.uint32)
    ng_mx = mx.array(num_groups, dtype=mx.uint32)

    out = kernel(
        inputs=[x_1d, w_packed, scales, M_mx, K_mx, Kb_mx, ng_mx],
        grid=(M * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M,)],
        output_dtypes=[mx.float16]
    )[0]

    if x.ndim == 2:
        return out.reshape(1, -1)
    return out
