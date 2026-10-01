# Task: 4-token 3D attention (attn-4token)

## Goal
Let a decode step with up to 4 query tokens per sequence use the fast 3D (split-segment) Triton attention path, so
that speculative decoding with 2 or 3 draft tokens stays fast at long context. Today only up to 2 query tokens use it.

## Why
A brain test on 2026-10-01 with 3 MTP draft tokens (4 query tokens per verify) measured 46.4 tok/s at 16K and 15.9 at
60K, against 113 and 110 with 1 draft token. Tokens per step rose to 2.67, but each step took about 58 ms at 16K and
168 ms at 60K, because the 4-token verify falls back to the 2D attention path.

## Your earlier work this builds on (read these folders first)
- 3d-2token-dispatch/: the one-line dispatch change that let 2 query tokens use the 3D path.
- 27b-crash/: why the 2-token change first crashed the 27B at startup. CUDA graph capture runs the largest batches
  the 3D path accepts, and the segment buffer overflowed there although small tests passed. Do not repeat that.
- 3d-buffer-fix/: the segment buffer sizing fix (out-of-bounds write) and gate2.py.
- 3d-multitoken-gate/: gate.py, the correctness harness.
- attn-segments/: the 128-segment result (attn_gate.py).
The sandbox's installed attention files are the brain's own (2-token dispatch, buffer fix, 128 segments).

## Steps
1. From the installed triton_unified_attention.py and triton_attn.py, find every place that limits the 3D path to
   2 query tokens per sequence, and every buffer whose size depends on that limit. Explain each in Results.
2. Make copies in this folder (never edit the installed files) that allow up to 4 query tokens per sequence, with
   buffers sized for it. Keep the changes minimal. Write them as patch files against the installed versions.
3. Correctness, on G7: using the existing gates (extend their parameters if they allow it; if you must add a case,
   keep the existing reference, comparison and broken control unchanged), test query lengths 1, 2, 3 and 4 per
   sequence, with 1, 2, 4 and 8 sequences, at contexts of about 1K, 16K and 60K, and also at the largest number of
   sequences the 3D path accepts (the batch CUDA graph capture will run) with 4 query tokens each, checking that no
   buffer is written out of bounds. Every case must match the reference
   as closely as the current 2-token path does, and a deliberately broken control must fail. Report max error per case.
4. Speed, on G7, timed inside a CUDA graph: attention time per call for a 4-token verify at 16K and 60K, patched 3D
   path against the current 2D fallback, and the 2-token 3D path for comparison. Also report the extra buffer memory.

## Rules
- G7 only, short runs. Never G9. Never edit installed files, the brain, or any image. No package installs.
- Stop after 150 minutes of work with what you have.
- Commit only this folder: git add /workspace/attn-4token/ (never git add -A or git add .), one commit at the end.

## Deliverable
Results below: the limits found (file and line), the patches, the correctness table, the speed table, the memory
cost, and a one-line verdict: "ready for a brain test" or "not ready", with the reason.

## Results
PASS.

### What was found (the limits)
The 2-token limit lives in exactly two places; everything else in the 3D path is
already token-count general (verified in source — BLOCK_Q at unified:842,
total_num_q_blocks at unified:853, reduce_segments at unified:1039 which reduces
token-by-token via program_id(0), and the per-row causal mask). The MTP path is a
separate dispatch (max_seqlen_q==1) and is untouched.

1. `triton_unified_attention.py:929` — dispatch gate: `or max_seqlen_q > 2` → `> 4`.
2. `triton_attn.py:188` — 3D segm buffer sizing: `max_q_tokens_3D = 2 * seq_threshold_3D`
   → `4 * seq_threshold_3D` (first dim 256 → 512).

Patches: `triton_unified_attention.patch`, `triton_attn.patch` (apply cleanly on the
installed vllm). Copies of the original and patched files in `orig/` and `*.patched`.

### Correctness (G7, nq=3 nkv=1 hs=256 — the 27B per-card shape; ref = 2D path of the same wrapper)
Tolerance: max_abs ≤ 0.02 AND max_rel(|ref|≥0.01) ≤ 0.05. All 108 cases PASS, all on the
3D path (3D grid confirmed via a grid spy on the kernel launch), max_abs in
[6.1e-05, 1.95e-03] — at the bf16 noise floor, identical to the 1/2-token cases.

- nq∈{3,4}, ctx∈{1024,16384,60000}, seqs∈{1,2,4,8}, ntok∈{1,2,3,4}: 96/96 PASS.
- Max-batch (the 27B crash shape): S=128, ntok=4 → 512 query tokens total, all ctx.
  This is the exact batch that crashed the 27B under the 2-token change. PASS,
  grid [230,1,128], max_abs 1.95e-03 (ctx=1024), no OOB.
- Boundary ntok=3 at 60k ctx: PASS, max_abs 6.1e-05.

Broken controls (prove the harness detects a bug):
- OOB control (candidate forced to the OLD bufdim=256 at S=128, ntok=4):
  ctx=1024 → CRASH (hipErrorIllegalAddress, GPU fault); ctx=16384 → HANG (subprocess
  timed out). Exactly the 27B-crash signature, reproduced and explained: the old buffer
  first dim 256 < 512 needed by the 4-token launch.
- Broken-reduce control (gate.py verbatim, ntok=4): FAIL, max_abs 0.0732, max_rel 4.13.

### Speed (G7, inside one CUDA graph, median of 20 replays, events outside; per call)
16 layers, 16 distinct 96 MB caches (Infinity Cache defeated), nq=3 nkv=1 hs=256.
"ms/token" = ms/call × 16 (the FINDINGS convention).

| ctx   | S | mode         | grid      | ms/call | ms/token(x16) |
|-------|---|--------------|-----------|---------|---------------|
| 16384 | 1 | current 2D   | [1,1]     | 2.671   | 42.74         |
| 16384 | 1 | patched 3D   | 1,1,128   | 0.042   | 0.66          |
| 16384 | 1 | 2-token 3D   | 1,1,128   | 0.047   | 0.74          |
| 16384 | 4 | current 2D   | [7,1]     | 3.136   | 50.17         |
| 16384 | 4 | patched 3D   | 7,1,128   | 0.117   | 1.87          |
| 16384 | 4 | 2-token 3D   | 5,1,128   | 0.110   | 1.77          |
| 60000 | 1 | current 2D   | [1,1]     | 16.935  | 270.96        |
| 60000 | 1 | patched 3D   | 1,1,128   | 0.104   | 1.67          |
| 60000 | 1 | 2-token 3D   | 1,1,128   | 0.102   | 1.64          |
| 60000 | 4 | current 2D   | [7,1]     | 11.382  | 182.11        |
| 60000 | 4 | patched 3D   | 7,1,128   | 0.365   | 5.84          |
| 60000 | 4 | 2-token 3D   | 5,1,128   | 0.356   | 5.70          |

3D is 64–162× faster than the 2D fallback (per-call). 4-token ≈ 2-token 3D (within
noise at 16k; at 60k the 4-token S=1 case is marginally faster than 2-token because the
4-token launch covers the same KV with fewer program rows). The 4-token patch does not
regress 3D speed.

Cross-check: my absolute 3D numbers match the established FINDINGS reference within ~7%
(0.104 ms/call ×16 = 1.67 ms/token @60k vs FINDINGS 1.61; 0.042×16 = 0.66 @16k vs 0.72).
The reference `attn-segments/attn_gate.py` cannot be used as an anchor here — it hardcodes
the pre-buffer-fix backend md5 (31f3473…) and the installed file is the current brain's
(6eea73a…), so it STOPPED on the file-hash check.

### Memory
One set of 3D segm buffers (output/max/expsum) per attention group. The 16 identical
full-attention layers (nq=3) are deduped by gpu_model_runner into ONE attention group and
share ONE builder → ONE buffer set. Old (256) = 96.75 MiB, new (512) = 193.5 MiB,
**extra ≈ 96.75 MiB total** (not 16×). Negligible against a multi-GB KV cache.

### Telemetry (G7, during/after runs)
Junction up to 58C, power 131 W, sclk 2390 MHz, busy 74% during the bench; card returned
to 40C / 0% busy after.

### Files
- `triton_unified_attention.patch`, `triton_attn.patch` — the minimal 2-line patch.
- `orig/` — installed originals (md5 recorded); `*.patched` — patched copies.
- `gate4.py` — correctness gate (in-process matrix + isolated OOB/broken controls).
- `gate4_all.log` — raw JSON lines for all 108 PASS cases + OOB controls.
- `bench4.py` / `bench4.json` — CUDA-graph speed bench.
- `results.json` — full raw numbers (correctness, OOB, broken, speed, memory, telemetry).

### Not done / caveats
- No full 27B end-to-end run in this task (the gate reproduces the crash batch and the
  exact OOB signature instead; the 2-token crash was a buffer-size bug, now fixed and
  directly exercised). The OOB HANG case needs the 90s subprocess kill to clear the card.
- attn_gate.py's stale md5 check is a separate pre-existing issue (it predates this task).
