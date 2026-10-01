# rc3 patches for the public repository (rc2 to rc3c)

This folder documents every change from rc2 to rc3c so that anyone with eight
RX 7900 XTX cards (gfx1100) can rebuild the brain image and check each change.
It is documentation and file assembly only. It contains no compiled binaries:
the two `.so` libraries the images install (rdna_ar8.so, fen_fp32_op.so) are
built from the sources here (section e). Every number below is taken from the
release manifests, the results files, or the task folders' Results; the source
is named beside it. Nothing is from memory. Written by Itko (the agent) from the
frozen release; audited before publication, with five corrections.

The model is Qwen3.8-27B AWQ INT4, tensor-parallel 8, one request, exact
token rate (tokens counted per streamed chunk).

## The three releases in order

| Release | What it added | 16K tok/s | 60K tok/s | Gate it passed |
| --- | --- | --- | --- | --- |
| rc2 (baseline, 2026-09-26) | Triton 3D decode attention at 128 segments | 80.4 | 76.4 | op gate, fixed-order gate, attention gate (S=128) |
| rc3a (2026-10-01) | 8-card RDNA3 all-reduce (butterfly) | 108.2 | 106.1 | rdna_ar_test8 (w8) + 4-gate window |
| rc3b (2026-10-01) | tuned fixed-order fp32 kernel as the op library | 113.3 | 110.4 | gate_tune, gate_cover (one exception), fp32_op_gate_tuned |
| rc3c (2026-10-01) | 4-token 3D attention path, served with 3 draft tokens | 157.0 | 161.7 | 108/108 correctness gate |

Sources: rc2 80.4/76.4 from inputs/results/MANIFEST-rc2.md. rc3a 108.2/106.1 from
inputs/results/MANIFEST-rc3a.md (test window; that manifest's own rc2 baseline is
81.0/77.1). rc3b 113.3/110.4 from inputs/results/MANIFEST-rc3b.md (test window
2026-10-01 10:17-10:26). rc3c 157.0/161.7 for 3 draft tokens from inputs/results/MANIFEST.md
(test window 2026-10-01); rc3c with 1 and 2 draft tokens is 112.0/107.5 and
141.6/137.2 in the same file.

## a. The 8-card RDNA3 all-reduce (rc3a)

What it is. The kernel from vLLM PR #57767 (credited by PR number only; no
author name is supplied here) extended with a three-round butterfly for TP=8
(`allreduce/rdna_custom_all_reduce_tp8.cu`, namespace `rdna3_tp8`). The PR's
wrapper is ported to this image and allowed to handle uniform speculative
batches (`allreduce/rdna_custom_all_reduce.py`). It is switched on by the
environment variable FENSTONE_RDNA_AR=1; without it the image behaves as rc2.

What the butterfly does. Round k (k = 0, 1, 2) pairs each rank with
`rank ^ (1 << k)`. Each round: a rank pushes its current partial sum into the
partner's slot k, synchronises with that partner only (the TP=4 pair barrier),
then adds the two partials with the lower rank's value first. Because every
rank adds in the same order, both partners of a round (and therefore all eight
ranks after three rounds) hold bit-identical results. The three rounds use
slots 0, 1, 2 of each epoch half.

Why decode-sized messages only.
- The TP=8 kernel refuses any message larger than kMaxNumel = 128*1024
  elements (`all_reduce` TORCH_CHECK); larger messages stay on RCCL.
- The wrapper's `should_custom_ar` also returns False when, on gfx1100,
  `inp.numel() >= 128*1024`, to stay below RDNA3's variable-performance
  tagged/bulk ring region. (The 2 MiB IPC buffer itself holds 1,048,576 bf16
  elements; the 128K cap is a separate, deliberate limit to decode-sized messages.)
- It only fires while the current stream is capturing a CUDA graph
  (`torch.cuda.is_current_stream_capturing()`), which is the decode path;
  eager prefill never takes it. A decode verify step of 3 draft tokens on this
  model (hidden size 5,120) is 4 x 5,120 = 20,480 elements, well under the cap;
  prefill (thousands of tokens) is not. (8,192 is the kernel's maximum hidden
  size, not this model's.)

Exactness and timing against RCCL, 8 cards (results/rdna_ar_test_w8.json):
exact integer sums, every rank bit-identical, RMS error 0.0031 against
RCCL's 0.0037. Per call, 17.7 us at 2 tokens against RCCL's 67.8, and 17.9
against 40.2 at 1 token (inputs/results/MANIFEST-rc3a.md; raw rows in the
JSON: 1 token 17.938/40.164 us, 2 tokens 17.674/67.778 us). At 4, 8, 16 and 24
tokens the RDNA kernel is 23.2/33.6/51.5/64.0 us against RCCL's
69.1/58.1/66.3/78.9 us.

## b. The communicator hook (rc3a)

`allreduce/patch_comm.py` applies four counted edits to the image's
`cuda_communicator.py`. It refuses to run unless every anchor matches exactly
once, and it refuses to patch a file that already carries the hook (it checks
for the marker string "Fenstone: graph-only RDNA" and exits STOP if present, so
it cannot patch twice). The patched file is `allreduce/cuda_communicator.py`.

1. Add `import os` to the imports (needed to read FENSTONE_RDNA_AR).
2. In `__init__`, after `self.use_flashinfer_allreduce = ...`, add
   `self.rdna_ar_comm = None` and a block that, when FENSTONE_RDNA_AR=1 and the
   group is a TP group with world_size > 1, constructs
   `RdnaCustomAllreduce(self.cpu_group, self.device, enabled=True)`.
3. At the top of `all_reduce`, call `rdna_ar_comm.custom_all_reduce(input_)`
   first and return it if it is not None (so it wins over every other backend).
4. In `destroy`, call `rdna_ar_comm.close()` and set it to None.

Why the wrapper accepts uniform batches. `should_custom_ar` reads
`get_forward_context().batch_descriptor` and accepts any batch with
`batch_descriptor.uniform` true and `num_tokens` up to 128 (not only one token
per request). That is why two-token speculative verify steps (and the 4-token
step that rc3c adds) use the RDNA all-reduce instead of falling through to RCCL.
The docstring of `rdna_custom_all_reduce.py` states this explicitly.

## c. The tuned fixed-order kernel (rc3b)

The op library `fen_fp32_op.so` (md5 7cfa693db0a3483b6cf82921b827225c) is built
from `tuned/q_gemm_rdna3_tuned.cu` (md5 74f2b097ec0796c37a7a2c8e2aa9fcb3) plus
`tuned/qdq_4_rdna3.cuh`. It keeps the fixed-order accumulate of rc2 (each K-slice
writes its fp32 partial sums into its own slice of a [splits, M, N] scratch
buffer; a separate reduce kernel adds the slices in order 0, 1, 2, ... and casts
to bf16 once). The only change is the per-shape K-slice size TK (slice width
equals the thread count) and the per-M tile height for M >= 8.

Slice size per shape (TK), bf16 path only (fp16 stays at 256):

M = 1..7 (fp32-tune/README.md Results, from a 12-config sweep on G7):
| Shape (K, N) | calls/token | TK at M=1 | TK at M=2 | TK at M=4 |
| --- | --- | --- | --- | --- |
| qkv (5120, 2048) | 16 | 128 | 128 | 128 |
| qkvz (5120, 2048) | 48 | 128 | 128 | 128 |
| o (768, 5120) | 16 | 64 | 64 | 64 |
| gdn_out (768, 5120) | 47 | 64 | 64 | 64 |
| gate_up (5120, 4352) | 64 | 128 | 256 | 256 |
| down (2176, 5120) | 64 | 128 | 128 | 128 |

M >= 8 (fp32-cover extension). The dispatch in `q_gemm_rdna3_tuned.cu`
`pick_tk()` is the source of truth for the build:
| Shape (K, N) | TK at M=8 | TK at M=9..15 | M_COUNT |
| --- | --- | --- | --- |
| o, gdn_out (768, 5120) | 64 | 128 | 8 (M8-11), 4 (M12-13), 8 (M14-15) |
| gate_up (5120, 4352) | 128 | 256 | as above |
| qkv, qkvz (5120, 2048) | 128 | 256 | as above |
| down (2176, 5120) | 128 | 256 | as above |
| anything else | 256 | 256 | 8 |
Note: the file's header prose lists gate_up as "TK=128 at M=8 and M>=15", but
the code dispatches TK=256 at M=15 (the code's `size_m <= 8 ? 128 : 256` gives
128 only at M=8). The code is what ships; the prose is imprecise.

Results per batch size (ms per token over the 255 quantized GEMM calls, median
of 3 shuffled runs, graphed, >=401 MB distinct weight copies; source
fp32-cover/README.md Results, gate_cover.py FULL mode). cand = tuned kernel,
det_ref = the rc2 fixed-order kernel it replaces:

| M | cand | det_ref | triton |
| --- | --- | --- | --- |
| 1 | 5.32 | 6.46 | 18.24 |
| 2 | 5.54 | 6.66 | 18.15 |
| 3 | 6.19 | 6.97 | 18.21 |
| 4 | 6.33 | 7.77 | 18.27 |
| 6 | 7.72 | 8.61 | 18.28 |
| 8 | 7.78 | 9.66 | 18.30 |
| 12 | 9.72 | 11.58 | 18.33 |
| 15 | 11.20 | 11.80 | 18.37 |

The M = 1, 2, 4 figures match the fp32-tune gate (5.30, 5.54, 6.29) within
run-to-run. `tuned/op_gate_run.log` confirms the build (source md5 74f2b097,
library md5 7cfa693d) and prints "ALL OP GATES PASS" (B, D, X, F, C, G), with
decode precision at the bf16 rounding floor and M=1 bit-identical across calls.

The 8-row bug the coverage task found. The kernel as it first passed gate_tune
failed gate_cover at M = 8, 12 and 15. Root cause (fp32-cover/README.md
Results): the M_COUNT=8 dispatch branch (size_m > 7) hardcoded TK=256 and
ignored the tk argument, so pick_tk()'s per-shape choice was never applied at
M = 8, 12 or 15. The fix (per-shape / per-M dispatch only) moved M = 8 from
about 9.6 to 7.78 and M = 12 from 11.54 to 9.72, turning both from S-FAIL to
S-PASS.

The recorded 15-row exception. gate_cover S failed only the
triton-minus-8.0 ms margin at M = 15: cand 11.20, needed <= 10.37. It passes
the other S sub-rule at M = 15 (1.02 x det_ref = 1.02 x 11.80 = 12.04, and
11.20 <= 12.04). At M = 15 the brain's comparison is det_ref, which this kernel
beats, so the failure is accepted by the operator as a documented exception to
be judged separately (fp32-cover/README.md Results; MANIFEST-rc3b.md).

## d. The 4-token 3D attention path (rc3c)

The change is two lines. The 2-token limit lived in exactly two places
(attn-4token/README.md Results); everything else in the 3D path is already
token-count general.
1. `triton_unified_attention.py:929` dispatch gate: `or max_seqlen_q > 2`
   changed to `or max_seqlen_q > 4`.
2. `triton_attn.py:188` segment buffer sizing:
   `max_q_tokens_3D = 2 * self.seq_threshold_3D` changed to
   `max_q_tokens_3D = 4 * self.seq_threshold_3D` (first dimension 256 to 512).

The patches are `attn4/triton_unified_attention.patch` and
`attn4/triton_attn.patch`; the patched files are
`attn4/triton_unified_attention.py` and `attn4/triton_attn.py` (what the
Containerfile COPYs into the image).

The 108-case gate (attn-4token/README.md Results, G7, nq=3 nkv=1 hs=256, the
27B per-card shape; reference is the 2D path of the same wrapper). Tolerance
max_abs <= 0.02 AND max_rel (|ref| >= 0.01) <= 0.05. All 108 cases PASS, all on
the 3D path (confirmed via a grid spy on the kernel launch), max_abs in
[6.1e-05, 1.95e-03] at the bf16 noise floor. The 108 are: 96 in the grid
nq in {3,4} x ctx in {1024,16384,60000} x seqs in {1,2,4,8} x ntok in {1,2,3,4},
plus the max-batch cases and the boundary ntok=3 at 60k. A deliberately broken
reduce control fails (max_abs 0.0732, max_rel 4.13), proving the harness
detects a bug. MANIFEST.md and Containerfile.rc3c both record 108 of 108.

The largest-batch test (the 27b-crash lesson). CUDA graph capture runs the
largest batch the 3D path accepts. The max-batch case is S=128 with ntok=4
(512 query tokens total), all contexts: this is the exact batch that crashed the
27B under the earlier 2-token change, because the segment buffer overflowed
there although small tests passed. It now PASSes with no out-of-bounds write
(grid [230,1,128], max_abs 1.95e-03 at ctx=1024). The broken OOB control
(forcing the candidate to the old buffer first-dim 256 at S=128, ntok=4)
reproduces the crash signature exactly: ctx=1024 crashes with
hipErrorIllegalAddress and ctx=16384 hangs. The old buffer first-dim 256 is
less than the 512 the 4-token launch needs.

Why it matters. Before the fix, 3 draft tokens on rc3b gave 46.4 tok/s at 16K
and 15.9 at 60K (results/ref_rc3b-k3-A.json is that run's text; MANIFEST.md),
because the 4-token verify fell back to the 2D attention path. With the fix,
3 draft tokens give 157.0/161.7 in the test window (MANIFEST.md).

## e. Building rc3a, rc3b, rc3c in order

rc2 itself is built from the public base image by the repository's existing
`patches/` folder (published 2026-09-26). Refer to it for the rc2 build; it is
not rewritten here. The three images below are each built from the previous
one. The build contexts are the directories the Containerfiles expect
(~/brain/pkg/rc3a, .../rc3b, .../rc3c); place the sources here in the matching
subfolder (allreduce/, tuned/, attn4/).

These commands are written, not run: podman is not available in this sandbox,
so none of them were executed here.

Build rc3a (from rc2). The Containerfile COPYs rdna_ar8.so plus two .py files.
Build rdna_ar8.so first from allreduce/rdna_custom_all_reduce_tp8.cu (it is not
in this folder; it is the compiled TP8 kernel). This is the command that built
it on Node02, inside the rc2 image so the compiler and PyTorch match, with the
folder holding the .cu file mounted as /w:
    podman run --rm --network=none -v "$PWD/allreduce:/w" -e PYTORCH_ROCM_ARCH=gfx1100 \
      --entrypoint python3 localhost/n02-brain:rc2 -c "from torch.utils.cpp_extension import load; \
      load(name='rdna_ar8', sources=['/w/rdna_custom_all_reduce_tp8.cu'], build_directory='/w', \
      extra_cuda_cflags=['-O3'], is_python_module=False)"
Then test it on eight cards with allreduce/rdna_ar_test8.py (exact sums, every
rank bit-identical, error against RCCL), and build the image:
    podman build -t localhost/n02-brain:rc3a -f allreduce/Containerfile.rc3a <rc3a context>
Verify each file by hash inside the image (md5, matching the frozen
MANIFEST-rc3a.md):
    podman run --rm --entrypoint sh localhost/n02-brain:rc3a -c \
      "md5sum /opt/fenstone/rdna_ar8.so /opt/python/lib/python3.14/site-packages/vllm/distributed/device_communicators/rdna_custom_all_reduce.py /opt/python/lib/python3.14/site-packages/vllm/distributed/device_communicators/cuda_communicator.py"
Expected: rdna_custom_all_reduce.py 0b06c90ad213c51386f7548e72ce5e6e,
cuda_communicator.py bbf33ba30a5b45a94b22c49502270390. The Node02 build of
rdna_ar8.so was 338a041c0ac8b406529d2b8f8c0377c1, but a compiled library
generally does not rebuild to the same hash: check the .cu source by hash
(SHA256SUMS) and the built library by running rdna_ar_test8.py.

Build rc3b (from rc3a). The Containerfile COPYs only fen_fp32_op.so. Build it
from tuned/q_gemm_rdna3_tuned.cu + tuned/qdq_4_rdna3.cuh exactly as
tuned/fp32_op_gate_tuned.py step B does: disable the WMMA branch
(`if (false && ...)`), rename the symbols (namespace vllm to vllmfen, entry to
fen_fp32_entry), and append the _fenstone_C::gptq_gemm_rdna3_fp32 glue (the op
with the FENSTONE_FP32_PREFILL switch). That produces the op the brain runs.
    podman build -t localhost/n02-brain:rc3b -f tuned/Containerfile.rc3b <rc3b context>
Verify by hash inside the image (md5, matching MANIFEST-rc3b.md):
    podman run --rm --entrypoint sh localhost/n02-brain:rc3b -c \
      "md5sum /opt/fenstone/fen_fp32_op.so"
The Node02 build was 7cfa693db0a3483b6cf82921b827225c; as with rdna_ar8.so, a
rebuild will usually differ in hash, so check the sources by hash and the built
library by its gate (fp32_op_gate_tuned.py must print ALL OP GATES PASS). The
rc3a files are inherited and unchanged.

Build rc3c (from rc3b). The Containerfile COPYs the two patched attention
backends.
    podman build -t localhost/n02-brain:rc3c -f attn4/Containerfile.rc3c <rc3c context>
Verify by hash inside the image (md5, matching MANIFEST.md):
    podman run --rm --entrypoint sh localhost/n02-brain:rc3c -c \
      "md5sum /opt/python/lib/python3.14/site-packages/vllm/v1/attention/ops/triton_unified_attention.py /opt/python/lib/python3.14/site-packages/vllm/v1/attention/backends/triton_attn.py"
Expected: triton_unified_attention.py c84f3ca9848b4fc7e590ca5d7900ec92,
triton_attn.py 105986eb1d88abede7df3a46a82570fb.

At run time all three use FENSTONE_RDNA_AR=1; rc3c is served with 3 MTP draft
tokens. The per-release test windows are allreduce/rc3a-test.sh,
tuned/rc3b-test.sh and attn4/rc3c-spec.sh.

## What is not done

- Three draft tokens' text check shows more near-tie divergence than one or
  two drafts: against rc3b, 1 draft diverges on 6 of 12 prompts, 2 drafts on 5
  of 12, 3 drafts on 9 of 12; largest logprob difference 0.37, 0.32, 0.29 and
  top-5 agreement 95.9, 95.6, 94.1 percent respectively (MANIFEST.md).
- Concurrency is unmeasured: the manifests mark concurrent requests as not yet
  proven.
- The INT8 cache and the all-reduce/norm fusion are not attempted.

## Files and licenses

Each copied file is listed with its license exactly as stated in its own
header. "No header" means the file has no license statement. Do not relabel:
the vLLM-derived files carry an Apache-2.0 SPDX header attributing "Copyright
contributors to the vLLM project".

| File | License (as in header) |
| --- | --- |
| allreduce/rdna_custom_all_reduce_tp8.cu | Apache-2.0 (SPDX; Copyright contributors to the vLLM project) |
| allreduce/rdna_custom_all_reduce.py | Apache-2.0 (SPDX; Copyright contributors to the vLLM project) |
| allreduce/cuda_communicator.py | Apache-2.0 (SPDX; Copyright contributors to the vLLM project) |
| allreduce/patch_comm.py | no header |
| allreduce/rdna_ar_test8.py | no header |
| allreduce/Containerfile.rc3a | no header |
| allreduce/rc3a-test.sh | no header |
| tuned/q_gemm_rdna3_tuned.cu | Apache-2.0 (SPDX; Copyright contributors to the vLLM project) |
| tuned/qdq_4_rdna3.cuh | Apache-2.0 (SPDX; Copyright contributors to the vLLM project) |
| tuned/fp32_op_gate_tuned.py | no header |
| tuned/op_gate_run.log | no header |
| tuned/Containerfile.rc3b | no header |
| tuned/rc3b-test.sh | no header |
| attn4/triton_unified_attention.py | Apache-2.0 (SPDX; Copyright contributors to the vLLM project) |
| attn4/triton_attn.py | Apache-2.0 (SPDX; Copyright contributors to the vLLM project) |
| attn4/triton_unified_attention.patch | no header |
| attn4/triton_attn.patch | no header |
| attn4/Containerfile.rc3c | no header |
| attn4/rc3c-spec.sh | no header |
| attn4/itko-attn-4token-README.md | no header |

## Files in this folder and hash verification

20 files are copied from inputs/ (sources, patches, Containerfiles, tests,
gates, and the attn4 README), in three subfolders. SHA256SUMS lists them.
Every copy was verified against inputs/ by md5 (20 of 20 identical) and, where
present in the frozen release, cross-checked against inputs/results/MANIFEST.md
(20 of 20 md5 match). Nothing from inputs/results/task01/ (evaluation material)
is included. The two compiled .so files are not present in inputs/ and are not
copied; they are built from the sources (section e).
