#!/usr/bin/env python3
"""
Empirical Codebook Search with Outlier Fallback and Minimum Description Length (MDL)
Optimized Engine for Bonsai 2 27B MLX Ternary Weights.

Refinements:
1. Exact ternary mapping validation:
   raw uint2 code 0 -> -1, 1 -> 0, 2 -> +1.
   Assertion: set(unpacked_ternary) in {-1, 0, 1}.
2. Offset search optimization:
   Stage 1 tests 4 coarse offsets: {0, L//4, L//2, 3*L//4}.
   Stage 2 tests all offsets 0..L-1 only for the winning candidates.
3. Dual Codebook Storage Reporting:
   - R_2bit: Hardware-friendly 2L bits per codebook entry (fast Metal bitmask/shift).
   - R_packed: Information-theoretical ceil(L * log2(3)) bits per entry.
   - R_entropy: Theoretical Shannon entropy limit.
"""

import math
import time
from collections import Counter
import numpy as np
from safetensors import safe_open

MODEL_PATH = "/Users/mohammedhossam/.lmstudio/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/model.safetensors"
CODEBOOK_BUDGET_BITS = 64 * 1024 * 8  # 64 KB

L_CANDIDATES = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]
K_CANDIDATES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def unpack_tensor_to_trits(u32_arr: np.ndarray) -> np.ndarray:
    """
    Unpack [rows, words] uint32 array into flat ternary array in {-1, 0, +1}.
    Affine mapping in MLX Bonsai 2:
      raw code 0 -> -1
      raw code 1 ->  0
      raw code 2 -> +1
    """
    rows, words = u32_arr.shape
    u = np.empty((rows, words * 16), dtype=np.int8)
    for lane in range(16):
        # raw code in {0, 1, 2}
        raw_code = ((u32_arr >> (2 * lane)) & 3).astype(np.int8)
        u[:, lane::16] = raw_code - 1
    return u.ravel()


def discover_linear_layers(model_path: str):
    with safe_open(model_path, framework="numpy") as f:
        keys = list(f.keys())
        linear_keys = [
            k
            for k in keys
            if k.endswith(".weight")
            and not any(
                x in k
                for x in [
                    "embed_tokens",
                    "lm_head",
                    "conv1d",
                    "norm",
                    "in_proj_a",
                    "in_proj_b",
                ]
            )
            and f.get_slice(k).get_dtype() == "U32"
        ]
    return sorted(linear_keys)


def stream_blocks(weights: np.ndarray, L: int, offset: int):
    n = len(weights)
    end = n - ((n - offset) % L)
    if end <= offset:
        return np.empty((0, L), dtype=np.int8)
    return weights[offset:end].reshape(-1, L)


def stage1_discover_patterns(
    model_path: str,
    layer_keys: list,
    L: int,
    sample_stride: int = 10,
    sample_layers_limit: int = 40,
):
    """Fast Stage 1 pattern discovery with 4 coarse offsets and vectorized hashing."""
    sampled_layers = layer_keys[::sample_stride][:sample_layers_limit]

    # Test 4 coarse offsets: 0, L//4, L//2, 3L//4
    coarse_offsets = sorted(list({0, L // 4, L // 2, (3 * L) // 4}))
    with safe_open(model_path, framework="numpy") as f:
        first_layer_u32 = f.get_tensor(sampled_layers[0])
        first_trits = unpack_tensor_to_trits(first_layer_u32)

    best_offset = 0
    max_top_freq = -1
    for o in coarse_offsets:
        blocks = stream_blocks(first_trits[:400000], L, o)
        if len(blocks) == 0:
            continue
        sample_counts = Counter(map(tuple, blocks[:10000]))
        top_c = sample_counts.most_common(1)[0][1] if sample_counts else 0
        if top_c > max_top_freq:
            max_top_freq = top_c
            best_offset = o

    pattern_counter = Counter()
    total_sampled_blocks = 0
    is_base3_int = L <= 38
    # Map {-1, 0, 1} to {0, 1, 2} for base 3 dot product
    powers = 3 ** np.arange(L - 1, -1, -1, dtype=np.int64) if is_base3_int else None

    with safe_open(model_path, framework="numpy") as f:
        for k in sampled_layers:
            u32 = f.get_tensor(k)
            trits = unpack_tensor_to_trits(u32)
            blocks = stream_blocks(trits, L, best_offset)
            total_sampled_blocks += len(blocks)

            if is_base3_int:
                # blocks are in {-1, 0, 1} -> shift to {0, 1, 2}
                shifted = (blocks + 1).astype(np.int64)
                int_blocks = np.dot(shifted, powers)
                unique, counts = np.unique(int_blocks, return_counts=True)
                for u_val, c_val in zip(unique, counts):
                    pattern_counter[int(u_val)] += int(c_val)
            else:
                byte_blocks = blocks.tobytes()
                for i in range(0, len(byte_blocks), L):
                    pattern_counter[byte_blocks[i : i + L]] += 1

            if len(pattern_counter) > 150000:
                pattern_counter = Counter(dict(pattern_counter.most_common(50000)))

    return best_offset, pattern_counter, total_sampled_blocks


def evaluate_codebook(
    model_path: str,
    layer_keys: list,
    L: int,
    offset: int,
    codebook_set: set,
    codebook_size: int,
    is_base3_int: bool,
    eval_stride: int = 10,
):
    layers_to_eval = layer_keys[::eval_stride]
    total_B = 0
    total_T = 0
    total_N_out = 0
    layer_rates_2bit = []

    powers = 3 ** np.arange(L - 1, -1, -1, dtype=np.int64) if is_base3_int else None
    codebook_freqs = Counter()

    with safe_open(model_path, framework="numpy") as f:
        for k in layers_to_eval:
            u32 = f.get_tensor(k)
            trits = unpack_tensor_to_trits(u32)
            T_layer = len(trits)
            blocks = stream_blocks(trits, L, offset)
            B_layer = len(blocks)
            if B_layer == 0:
                continue

            if is_base3_int:
                shifted = (blocks + 1).astype(np.int64)
                int_blocks = np.dot(shifted, powers)
                unique_keys, counts = np.unique(int_blocks, return_counts=True)
                n_out_layer = 0
                for u_val, c_val in zip(unique_keys, counts):
                    u_int = int(u_val)
                    if u_int in codebook_set:
                        codebook_freqs[u_int] += int(c_val)
                    else:
                        n_out_layer += int(c_val)
            else:
                byte_blocks = blocks.tobytes()
                n_out_layer = 0
                for i in range(0, len(byte_blocks), L):
                    b = byte_blocks[i : i + L]
                    if b in codebook_set:
                        codebook_freqs[b] += 1
                    else:
                        n_out_layer += 1

            total_B += B_layer
            total_T += T_layer
            total_N_out += n_out_layer

            ell = math.ceil(math.log2(codebook_size + 1))
            layer_bits = B_layer * ell + n_out_layer * 2 * L
            layer_rates_2bit.append(layer_bits / T_layer)

    coverage = 1.0 - (total_N_out / total_B) if total_B > 0 else 0.0
    p_out = total_N_out / total_B if total_B > 0 else 0.0

    # 1. 2-bit hardware-friendly codebook storage
    cb_bits_2bit = codebook_size * (2 * L)
    # 2. Information theoretically packed codebook storage
    cb_bits_packed = codebook_size * math.ceil(L * math.log2(3))

    ell = math.ceil(math.log2(codebook_size + 1))
    common_data_bits = total_B * ell + total_N_out * 2 * L

    rate_2bit = (common_data_bits + cb_bits_2bit) / total_T if total_T > 0 else 0.0
    rate_packed = (common_data_bits + cb_bits_packed) / total_T if total_T > 0 else 0.0

    # Shannon Entropy Theoretical Limit
    if 0.0 < p_out < 1.0:
        h2 = -p_out * math.log2(p_out) - (1.0 - p_out) * math.log2(1.0 - p_out)
    else:
        h2 = 0.0

    n_in = total_B - total_N_out
    h_inlier = 0.0
    if n_in > 0:
        for count in codebook_freqs.values():
            q = count / n_in
            if q > 0:
                h_inlier -= q * math.log2(q)

    h_idx = h2 + (1.0 - p_out) * h_inlier
    m_entropy = total_B * (h_idx + p_out * 2 * L) + cb_bits_packed
    rate_entropy = m_entropy / total_T if total_T > 0 else 0.0

    r_95 = float(np.percentile(layer_rates_2bit, 95)) if layer_rates_2bit else 0.0
    r_worst = float(np.max(layer_rates_2bit)) if layer_rates_2bit else 0.0

    return {
        "coverage": coverage,
        "outlier_rate": p_out,
        "rate_2bit": rate_2bit,
        "rate_packed": rate_packed,
        "rate_entropy": rate_entropy,
        "codebook_bytes_2bit": cb_bits_2bit / 8,
        "codebook_bytes_packed": cb_bits_packed / 8,
        "r_95": r_95,
        "r_worst": r_worst,
    }


def main():
    start_time = time.time()
    print("=" * 85)
    print("BONSAI 2 27B TERNARY WEIGHT CODEBOOK SEARCH (REFINED MDL ENGINE)")
    print("=" * 85)

    linear_keys = discover_linear_layers(MODEL_PATH)
    print(f"Total linear ternary weight layers: {len(linear_keys)}")

    # 1. Sanity check unpacking on the first layer
    print("\n[Refinement 1] Executing sanity check on ternary unpacking...")
    with safe_open(MODEL_PATH, framework="numpy") as f:
        w_sample = f.get_tensor(linear_keys[0])
        trits_sample = unpack_tensor_to_trits(w_sample[:2, :8])
    unique_vals = sorted(list(set(trits_sample)))
    print(f"Sample unpacked values (first 256 weights): {unique_vals}")
    assert set(unique_vals).issubset({-1, 0, 1}), f"Invalid ternary values: {unique_vals}"
    print("Assertion passed: Exact ternary values strictly adhere to {-1, 0, +1}.")

    all_results = []

    print("\n" + "=" * 85)
    print("STAGE 1: GRID EXPLORATION (10% STRATIFIED SAMPLE, COARSE OFFSETS)")
    print("=" * 85)
    print(
        f"{'L':>2} {'K':>5} {'Off':>3} {'Cov(%)':>7} {'Out(%)':>7} {'CB_2b(KB)':>9} "
        f"{'R_2bit':>8} {'R_packed':>9} {'R_entropy':>9} {'R95':>7}"
    )

    for L in L_CANDIDATES:
        t0 = time.time()
        best_offset, pattern_counter, _ = stage1_discover_patterns(
            MODEL_PATH, linear_keys, L, sample_stride=10
        )
        is_base3_int = L <= 38
        sorted_patterns = [p for p, _ in pattern_counter.most_common()]

        for K in K_CANDIDATES:
            cb_bits_2bit = K * (2 * L)
            if cb_bits_2bit > CODEBOOK_BUDGET_BITS:
                continue

            chosen_codebook = sorted_patterns[:K]
            codebook_set = set(chosen_codebook)

            eval_stats = evaluate_codebook(
                MODEL_PATH,
                linear_keys,
                L=L,
                offset=best_offset,
                codebook_set=codebook_set,
                codebook_size=K,
                is_base3_int=is_base3_int,
                eval_stride=10,
            )

            res = {
                "L": L,
                "K": K,
                "offset": best_offset,
                "coverage": eval_stats["coverage"],
                "outlier_rate": eval_stats["outlier_rate"],
                "codebook_bytes_2bit": eval_stats["codebook_bytes_2bit"],
                "codebook_bytes_packed": eval_stats["codebook_bytes_packed"],
                "rate_2bit": eval_stats["rate_2bit"],
                "rate_packed": eval_stats["rate_packed"],
                "rate_entropy": eval_stats["rate_entropy"],
                "r_95": eval_stats["r_95"],
                "r_worst": eval_stats["r_worst"],
            }
            all_results.append(res)
            print(
                f"{L:2d} {K:5d} {best_offset:3d} {eval_stats['coverage']*100:7.2f} "
                f"{eval_stats['outlier_rate']*100:7.2f} {eval_stats['codebook_bytes_2bit']/1024:9.1f} "
                f"{eval_stats['rate_2bit']:8.4f} {eval_stats['rate_packed']:9.4f} "
                f"{eval_stats['rate_entropy']:9.4f} {eval_stats['r_95']:7.4f}"
            )

    # Sort candidates by R_2bit
    all_results.sort(key=lambda x: x["rate_2bit"])

    print("\n" + "=" * 85)
    print("STAGE 1 TOP CANDIDATES:")
    print("=" * 85)
    for r in all_results[:5]:
        print(
            f"L={r['L']:2d} K={r['K']:5d} (Off={r['offset']}) | Cov: {r['coverage']*100:5.2f}% | "
            f"R_2bit: {r['rate_2bit']:.4f} b/w | R_packed: {r['rate_packed']:.4f} b/w | "
            f"R_entropy: {r['rate_entropy']:.4f} b/w | R95: {r['r_95']:.4f}"
        )

    # Stage 2: Offset refinement for top candidate
    top_cand = all_results[0]
    best_L = top_cand["L"]
    best_K = top_cand["K"]
    print(f"\n[Refinement 2] Stage 2: Exhaustive offset sweep for L={best_L}, K={best_K} across all {best_L} offsets...")

    best_sweep_offset = top_cand["offset"]
    best_sweep_rate = 999.0
    best_sweep_stats = None

    with safe_open(MODEL_PATH, framework="numpy") as f:
        sampled_trits = unpack_tensor_to_trits(f.get_tensor(linear_keys[0]))

    for off in range(best_L):
        # Build codebook for this offset
        _, p_counter, _ = stage1_discover_patterns(
            MODEL_PATH, linear_keys, best_L, sample_stride=10
        )
        s_patterns = [p for p, _ in p_counter.most_common()]
        cb_set = set(s_patterns[:best_K])
        stats = evaluate_codebook(
            MODEL_PATH,
            linear_keys,
            L=best_L,
            offset=off,
            codebook_set=cb_set,
            codebook_size=best_K,
            is_base3_int=(best_L <= 38),
            eval_stride=10,
        )
        if stats["rate_2bit"] < best_sweep_rate:
            best_sweep_rate = stats["rate_2bit"]
            best_sweep_offset = off
            best_sweep_stats = stats

    print(f"Optimal alignment offset: o* = {best_sweep_offset} (R_2bit: {best_sweep_rate:.4f} b/w)")

    # Decode and display top 20 sequences
    _, p_counter, _ = stage1_discover_patterns(
        MODEL_PATH, linear_keys, best_L, sample_stride=10
    )
    s_patterns = [p for p, _ in p_counter.most_common()]
    powers = 3 ** np.arange(best_L - 1, -1, -1, dtype=np.int64)

    print(f"\nTop 20 Most Frequent Ternary Sequences for Optimal L={best_L}:")
    for idx, p in enumerate(s_patterns[:20]):
        # Decode integer back to ternary values in {-1, 0, 1}
        val = p
        seq = []
        for _ in range(best_L):
            seq.append((val % 3) - 1)
            val //= 3
        seq = seq[::-1]
        print(f"  #{idx+1:02d}: {seq} (count in sample: {p_counter[p]})")

    # Final Summary Deliverables
    print("\n" + "=" * 85)
    print("FINAL EXPERIMENTAL DELIVERABLES")
    print("=" * 85)
    print(f"1. Best configuration: (L* = {best_L}, K* = {best_K}, o* = {best_sweep_offset})")
    print(f"2. Coverage: {best_sweep_stats['coverage']*100:.2f}% | Outlier rate: {best_sweep_stats['outlier_rate']*100:.2f}%")
    print(f"3. Codebook footprint: {best_sweep_stats['codebook_bytes_2bit']/1024:.2f} KB (Hardware 2-bit) / {best_sweep_stats['codebook_bytes_packed']/1024:.2f} KB (Packed base-3)")
    print(f"4. Compression Rates:")
    print(f"   - Hardware 2-bit Rate (R_2bit):       {best_sweep_stats['rate_2bit']:.4f} bits/weight")
    print(f"   - Theoretical Packed Rate (R_packed): {best_sweep_stats['rate_packed']:.4f} bits/weight")
    print(f"   - Shannon Entropy Bound (R_entropy):   {best_sweep_stats['rate_entropy']:.4f} bits/weight")
    print(f"   - Layer 95th Percentile Rate (R_95):   {best_sweep_stats['r_95']:.4f} bits/weight")
    print(f"   - Worst-case Layer Rate:               {best_sweep_stats['r_worst']:.4f} bits/weight")
    print(f"5. Hardware Feasibility: R_2bit <= 1.0 achievable? {'YES' if best_sweep_stats['rate_2bit'] <= 1.0 else 'NO'}")
    print(f"Total script runtime: {time.time()-start_time:.2f}s")
    print("=" * 85)


if __name__ == "__main__":
    main()
