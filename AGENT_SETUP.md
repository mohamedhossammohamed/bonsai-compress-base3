# Plug-and-play prompt for a coding agent

Clone https://github.com/mohamedhossammohamed/bonsai-compress-base3 and
reproduce the Scheme S2 result on a local ternary checkpoint. Logs under
./logs/.

1. `pip install -r requirements.txt` (numpy, safetensors; mlx for the engine
   bench on Apple Silicon only).
2. Supply your own ternary {-1,0,+1} checkpoint as 2-bit-packed uint32
   safetensors — record its source + SHA.
3. Repack: `python code/convert_to_base3.py --input <2bit> --output <base3>`
   (check `--help` for exact flags). Confirm: output ≈ 0.8× input bytes,
   peak RSS < 1 GB.
4. Verify: `python code/benchmark_ternary_transmission_suite.py --weights
   <base3>` — every tensor must reconstruct bit-exact (zero mismatches).
5. Optional (Apple Silicon): `python code/bench_bonsai_vs_mlx.py` and report
   the speedup table.
6. Send back: full stdout, byte counts before/after, mismatch count (must be
   0), wall time, machine specs.

Keep weights local — upload nothing. Measure the scheme as published; leave the
packing untouched.
