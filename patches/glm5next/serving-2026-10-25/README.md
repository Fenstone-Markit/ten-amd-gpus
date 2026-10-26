# GLM-5.3-Flash on twelve RX 7900 XTX: the serving patch set (2026-10-25)

What [Chapter 9](../../../paper/09-what-we-were-given-and-what-we-built.md) serves on: 25 files over the community RDNA4 overlay, our FP8 and indexer kernels, the start script, the builders that made each change, and the test tools. It supersedes the 14-file port in [the folder above](../README.md) for serving; that port's files are all in here, some changed. Every file's hash is in [MANIFEST.md](MANIFEST.md).

## What it is built from

| Layer | What | From |
| --- | --- | --- |
| Image | `docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0` (vLLM 0.27.1, torch 2.12, ROCm 10, Python 3.14) plus the one-line load fix in [../vllm/model_executor/layers/fused_moe/](../vllm/model_executor/layers/fused_moe/routed_experts.py) | AMD; the fix is ours |
| Overlay | `djdeniro/GLM-5.3-Flash-rocm-r9700`, 72 whole-file replacements over vLLM for the R9700 (RDNA4) | its author, MIT; not included here |
| This set | `vllm/`: 25 files, each with a `.diff` against the overlay's copy (marked `a/overlay/`), AMD's image (`a/image/`), or nothing (`a/new/`) | ours |
| FP8 kernels | `kernels/fp8/`: the source of `libfp8rdna3.so` and its build line | ours |
| Weights | `wtdcode/GLM-5.3-Flash-AWQ-W4A16` (revision `abd7b077`) plus official FP8 experts from `zai-org/GLM-5.3-Flash` (revision `46a050be`) for layers 3, 6, 12, 18, 27, 36, 42 and the draft layer 45, joined by [`builders/mixed-ckpt-01.py`](builders/mixed-ckpt-01.py) and [`builders/mtp-ckpt-01.py`](builders/mtp-ckpt-01.py) (links the 4-bit files, copies the FP8 experts, merges the index) | Hugging Face |

The tree is built inside the image the same way as the port: copy the installed vLLM to `/tmp/ov/vllm`, copy the overlay's `app/vllm/vllm/` over it, copy `vllm/` from here over that, delete `__pycache__` in every touched folder, run with `PYTHONPATH=/tmp/ov`. [`serve/glm-serve-51.sh`](serve/glm-serve-51.sh) does exactly that in its container command, and checks every file's md5 first.

Licences: files marked `a/overlay/` in their diff are modified copies of overlay files and stay under the overlay's MIT licence ([../LICENSE-overlay](../LICENSE-overlay)); files marked `a/image/` are modified vLLM files under Apache-2.0 ([../../../LICENSE-APACHE-2.0](../../../LICENSE-APACHE-2.0)); new files are MIT, as the repository.

## The switches

Every change since the 14-file port sits behind an environment switch. The start sets these:

| Switch | Set to | What it does |
| --- | --- | --- |
| `GLM5_DSA_BREAK=indexer`, `GLM5_PROFILE_ALWAYS=1`, `GLM5_KDA_NOCACHE=1` | on | Chapter 5's three |
| `GLM5_WS_FIT=1` | on | the indexer's gather workspace sized to one window (52 MB at 393k instead of 2.08 GB) |
| `GLM5_FP8_LAYERS`, `GLM5_FP8_LIB` | `3,6,12,18,27,36,42,45`, the library path | these layers' experts run on our FP8 kernels; the loader skips the 4-bit tensors of those layers and the FP8 tensors of the others |
| `GLM5_NEXT_MTP_PROPOSER=1` | on | the overlay's GLM draft proposer |
| `GLM5_MHC_FUSED_MAX_T` | 1000000 | the overlay's fused mHC kernels for every step (its own limit was 64 tokens) |
| `GLM5_MQA_TRITON=1` | on | our Triton kernel for the indexer's prompt scores |
| `GLM5_CANARY=1`, `GLM5_PREWARM_SCAN=1` | on | the memory canary and the start-up fix from Chapters 6 and 8 |

Off unless set: `GLM5_SPEC_MIXED_TORCH=1` (old path for mixed MTP steps), `GLM5_PREFILL_EVERY_MS` (fair prompt chunks with two users; built, not tested), `GLM5_STAGE_TIMING`, `GLM5_MODULE_TIMING` (timing), `GLM5_TB_SIGNAL` (stack dumps every 90 s), `GLM5_STEP_TIMING`, `GLM5_MEM_TRACE`, `GLM5_EXPERT_COUNT`.

## The files changed or added after the 14-file port

| File | Tag | What and why |
| --- | --- | --- |
| `models/glm5next/nvidia/mtp.py` | mtp-01, mtp-05 | the draft model declares pipeline support; the main model's word embedding is loaded into the drafter (without stages vLLM shares it; with stages the drafter ran on an empty one, 11-18 % of drafts accepted) |
| `v1/worker/gpu_model_runner.py` | mtp-02, mtp-07 | drafter uses guarded on stages without one; the last stage sends every stage the next token, valid and accepted counts and the drafts, which the earlier stages apply to their KDA state and bookkeeping (vLLM 0.27 sent one token, and earlier stages' KDA state went wrong whenever a draft was accepted) |
| `v1/engine/core.py` | mtp-04, -06, -08 | with a draft model and async scheduling off: prompt chunks still overlap across stages; while a request decodes, one step in flight; drafts fetched only after a step's token is recorded (the hang); `GLM5_STEP_TIMING` |
| `v1/worker/gpu_worker.py` | mtp-03, stt-01, mt-01 | stack dumps; per-stage call timing; GPU-event timing per module during prompt steps |
| `v1/attention/backends/mla/rocm_aiter_mla_sparse.py` | mixspec-01 | a fourth dispatch gate: steps mixing MTP verify rows and a prompt chunk take the ragged Triton kernel instead of a per-token Python loop (2.2 s to 0.64 s per step on stage 1) |
| `model_executor/layers/mhc.py` | mhc-01 | `GLM5_MHC_FUSED_MAX_T` (18.9 s to 3.0 s per stage over 230k prompt tokens) |
| `v1/attention/ops/glm5_mqa_logits_rdna3.py` (new), `v1/attention/ops/rocm_aiter_mla_sparse.py` | mqa-01 | the indexer-score kernel and its switch; [`kernels/mqa/`](kernels/mqa/) has the same kernel with its tests |
| `model_executor/layers/quantization/fp8.py`, `compressed_tensors/compressed_tensors.py` | fp8 | hand FP8 layers (official checkpoint) or the listed layers (mixed checkpoint) to the methods below |
| `model_executor/layers/quantization/fp8_rdna3.py` (new) | fp8 | FP8 linear (dequantised once after loading) and FP8 MoE on our kernels, as custom op `glm5_fp8_rdna3_moe` |
| `model_executor/layers/quantization/glm5_fp8_rdna3.py` (new) | fp8 | ctypes wrapper of `libfp8rdna3.so`: decode path up to 32 tokens, prompt path in vLLM's sorted-token format |
| `models/glm5next/nvidia/model.py` | fp8 loader, ecount | skips the tensors a mixed checkpoint carries for the other precision; expert-count hooks (off) |
| `v1/worker/glm5_ecount.py`, `glm5_memsnap.py`, `v1/attention/ops/glm5_meta_watch.py` (new) | diagnostics | the expert count (Chapter 8), the allocation recorder and metadata watch (Chapter 6); off |

`vision/` holds the two files of the next set (claude-vis-01): with `GLM5_VISION_FIRST_STAGE_ONLY=1` only the first pipeline stage builds the vision tower and runs its profiling pass. Serve with images by dropping `--language-model-only` and adding `--limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_image_tokens":1024}'` ([`serve/glm-serve-54.sh`](serve/glm-serve-54.sh) `VIS=1`). Tested, not the serving default.

## The FP8 kernels (`kernels/fp8/`)

`fp8-rdna3-lib-01.hip`: e4m3 to fp16 or bf16 by bit masks (exact for all 256 values; [`fp8-convert-check-01.c`](kernels/fp8/fp8-convert-check-01.c) proves it), sums in fp32 with `v_dot2`, 128 x 128 block scales applied per K block. Decode: gate-up with SiLU, then down with the routing weights and the top-k sum, no atomics. Prompts: a WMMA GEMM in vLLM's sorted-token MoE format. Build: [`fp8-lib-build-01.sh`](kernels/fp8/fp8-lib-build-01.sh) (hipcc from the image, `--offload-arch=gfx1100`). The WMMA builtins are gfx1100's; RDNA4 needs its own build.

## Serving it

[`serve/glm-serve-51.sh`](serve/glm-serve-51.sh) is the exact start (paths are relative to the home folder of the machine it ran on). The flags that matter:

```bash
--tensor-parallel-size 4 --pipeline-parallel-size 3 --distributed-executor-backend mp \
--gpu-memory-utilization 0.97 --max-num-seqs 2 --max-model-len 393216 --max-num-batched-tokens 2048 \
--kv-cache-memory-bytes 1932735283 --kv-cache-dtype auto --language-model-only \
--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
--default-chat-template-kwargs '{"reasoning_effort":"low"}' \
--no-async-scheduling --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-prefix-caching \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6]}'
```

with `VLLM_PP_LAYER_PARTITION=16,16,13`, `VLLM_ROCM_USE_AITER=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, a writable MIOpen cache, the switches above, and the cards ordered by unique id with `ROCR_VISIBLE_DEVICES`. GLM's cache layout limits splits: with the draft layer the linear-attention layers fall into 3 groups, and each stage may carry at most one linear layer per group for each sparse layer it holds; 16,16,13 for three stages and 8,8,8,8,8,5 for six satisfy it.

The script's gates: every patch file's md5, the cards free, a PCIe error watch (30 s idle with no new error), a 107 C stop, warm-up answers checked, and a long-read gate (two cold reads of 74k and 116k tokens under a private cache key, answers checked, up to three tries) because of the start-up fault in Chapter 9.

## The builders and tools

`builders/` holds the scripts that made each change from the set before (each refuses if its anchor text is missing or not unique), so every change can be rebuilt and read on its own. `tools/` holds the tests the chapter's numbers come from: `prefix-check-01.py`, `prefix-stress-01.py` and `-02.py`, `mixed-check-02.py`, `needle-set-01.py`, `prefill-speed-01.py`, `first-read-01.py`, `mtp-probe-02.py`, `lc-check-01.py`, `vision-check-01.py`, and `mt-diff-01.py` for the timing tables. They talk to a server on `127.0.0.1:8000`. Three of them (`first-read-01.py`, `prefix-stress-01.py`, `-02.py`) and `builders/mqa-triton-01-edit.py` had the machine's home path written out; here it is `~` instead, the only change from the copies that ran, checked to compile but not re-run in this form.
