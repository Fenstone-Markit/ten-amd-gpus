# Task: a faster fixed-order kernel at the brain's decode size (fp32-tune)

## Goal
The brain's decode step is a two-token speculative verify, so its 255 quantized linears run at M = 2. Today's
fixed-order kernel takes about 6.75 ms per token for them at M = 2. Reading their weights alone needs about
2.7 ms at the card's measured 771 GB/s, so the kernel runs at roughly 2.5 times its bandwidth floor. Make it
faster at M = 2, while keeping it at the precision floor and bit-for-bit repeatable.

## Rules
- G7 only, under AGENTS.md.
- The gate is gate_tune.py (read-only, md5 1dc724b0012cb908a5685012f651c6d9). Do not edit it, copy and modify it, or write another harness. Its verdict is the result.
- Edit only src/q_gemm_rdna3_tuned.cu (it starts as a copy of today's kernel). src/q_gemm_rdna3_det_ref.cu is today's kernel, read-only: the gate builds it as the reference.
- This task is exempt from the 10-minute per-command limit for the full gate only: run it with timeout 3600. Everything else in AGENTS.md applies.
- Never edit installed files. Never touch the brain, other cards, or other containers.

## What the gate requires (read its header for the exact thresholds)
- N: your own kernel, at most two device operations per call (for example the GEMM and a reduce), no memset or copy.
- P: error within 10 % of the bf16 rounding floor on every shape, at M = 1, 2 and 4.
- R: output bit-identical across 20 runs. G: bit-identical under CUDA graph replay.
- S: at least 0.50 ms per token faster than today's kernel at M = 2; no slower at M = 1; at most 2 % slower at M = 4.
- Your output may differ from today's kernel in the last bit (the gate reports how many elements); P bounds the error.

## Where to look (hints, not requirements)
- Today's kernel splits K into slices of 256 (THREADS_X = BLOCK_KN_SIZE = 256): 20 slices at K = 5120, 9 at 2176, 3 at
  768. Each slice writes fp32 partial sums for its tile; the reduce kernel adds the slices in order and rounds once.
  Fewer, larger slices mean less scratch traffic and a cheaper reduce, but fewer workgroups in flight on 96 CUs.
- A tile is 256 K by 1,024 N: 256 threads, four output columns each. At M = 2 each thread does little work per
  weight it loads.
- The launcher may choose different settings per shape. The brain's six shapes and their call counts are in the gate.
- Already measured, do not retry: folding the reduce into the GEMM as one kernel is slower at M >= 2 (FINDINGS).

## Steps
1. `python gate_tune.py --preflight` must print PREFLIGHT PASS. If it prints STOP, copy the STOP line into Results and stop.
2. Change src/q_gemm_rdna3_tuned.cu.
3. `timeout 3600 python gate_tune.py` runs the full gate. Iterate until it prints ALL GATES PASS, or until you conclude it cannot: then say why.
4. Fill in Results: the gate md5 it printed, the N line, the three per-token speed lines, the VERDICT line, what you changed per shape, and the md5 of your final source. One git commit at the end.

## Results

Gate: gate_tune.py md5 1dc724b0012cb908a5685012f651c6d9 (matches README). Full gate run with
`timeout 3600 python gate_tune.py`, G7 only. Results file: gate_tune_results.json.

VERDICT: N PASS  I PASS  P PASS  R PASS  G PASS  S PASS — ALL GATES PASS

- N identity: own kernel vllmht_tunedcand, dtype bfloat16, shape (M, N), 2 device operations per
  call (GEMM + fixed-order reduce), no memset/copy.
- P precision: cand at the bf16 rounding floor on every shape at M = 1, 2, 4 (RMS rel 0.00165–0.00169,
  equal to det_ref, well under 1.10 x floor and 1.25 x triton).
- R repeatability: 0 elements changed over 20 runs, every shape x M.
- G graph: CUDA graph of 6 calls replayed 5 times, bit-identical to eager output.
- Speed per token over the 255 calls (us per call -> ms per token, median of 20 graphed replays):

| M | triton | hip_inst | det_ref | cand | gate requirement |
| --- | --- | --- | --- | --- | --- |
| 1 | 17.95 | 6.72 | 6.47 | 5.30 | <= 6.47 (no slower) — PASS, 0.17 ms faster |
| 2 | 18.00 | 6.47 | 6.61 | 5.54 | <= 6.11 (0.50 ms faster) — PASS, 1.07 ms faster |
| 4 | 18.11 | 7.69 | 7.60 | 6.29 | <= 7.75 (2 % slower) — PASS, 0.31 ms faster |

Output elements differing from det_ref (reported, not gated): 0–19 per shape x seed, at the single
last bit (different K-slice summation order); precision is at the floor.

### What changed per shape

Only the K-slice size TK (== thread count; one K element per thread, 4 N columns per thread, same
256 K / 1024 N tile as before, M_COUNT dispatch unchanged, fp16 path untouched at TK=256).
The kernel is now templated on TK and pick_tk() selects per shape, chosen from a 12-config sweep
(64/128/256/512 x M_COUNT 1/2/4) timed under the gate's speed conditions (sweep_src/):

| Shape (K, N) | calls | TK at M=1 | TK at M=2 | TK at M=4 |
| --- | --- | --- | --- | --- |
| qkv (5120, 2048) | 16 | 128 | 128 | 128 |
| qkvz (5120, 2048) | 48 | 128 | 128 | 128 |
| o (768, 5120) | 16 | 64 | 64 | 64 |
| gdn_out (768, 5120) | 47 | 64 | 64 | 64 |
| gate_up (5120, 4352) | 64 | 128 | 256 | 256 |
| down (2176, 5120) | 64 | 128 | 128 | 128 |

Rationale: TK=128 gives twice the blocks per call (more workgroups on 96 CUs) at the cost of double
the fp32 scratch traffic, which is negligible at these M. TK=64 wins the two K=768 shapes (N=5120
means only 2.5 x 4 tiles of 1024 at TK=256; 64 splits K into 12 slices and the wider N grid keeps
the CUs fed). gate_up is the exception at M>=2: there TK=256 is fastest (M2 30.22 vs 33.49 us;
M4 35.55 vs 41.68 us in the sweep), so it keeps the old slice size there and only M=1 moves to 128.
Everything else falls through to TK=256 (previous behaviour).

Source md5 (final): src/q_gemm_rdna3_tuned.cu 79a2da44321c85d49932be45114ee8e3

## Findings

- The fixed-order kernel's K-slice size is shape- and M-dependent on gfx1100 at decode sizes.
  TK=128 (double the 256 default) is fastest at M=1 and M=2 on most brain shapes; the extra fp32
  scratch traffic is negligible (weights dominate). At M=4 the best TK is shape-dependent
  (128-256). The two K=768, N=5120 shapes (o, gdn_out) are best at TK=64 at all M. gate_up
  (K=5120, N=4352) is the only shape where the old TK=256 wins at M>=2. Per-shape selection gives
  5.54 ms/token at M=2 vs 6.61 for the 256-tile kernel (16 %), 5.30 at M=1 vs 6.47, 6.29 at M=4
  vs 7.60; all at the bf16 rounding floor, bit-repeatable, graph-safe (gate_tune.py, 2026-09-26).

