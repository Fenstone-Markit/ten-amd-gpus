# GLM-5.3-Flash on ten RX 7900 XTX: the port

Everything [Chapter 5](../../paper/05-a-321b-model-on-ten-cards.md) ran on, in the order it is applied. Measured from 2026-10-14 to 2026-10-19 on ten RX 7900 XTX (gfx1100). Every file's hash is in [MANIFEST.md](MANIFEST.md).

## What it is built from

| Layer | What | Where it comes from |
| --- | --- | --- |
| Image | `docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`: vLLM 0.27.1, torch 2.12, ROCm 10, Python 3.14 | AMD |
| Overlay | `djdeniro/GLM-5.3-Flash-rocm-r9700`: 72 whole-file replacements over vLLM, written for the R9700 (RDNA4) | its author, MIT; not included here |
| Load fix | `vllm/model_executor/layers/fused_moe/routed_experts.py`: one class name added so compressed-tensors W4A16 MoE checkpoints load (without it: KeyError 'intermediate_size_full') | this repository |
| The port | 14 whole files | this repository |
| Weights | `wtdcode/GLM-5.3-Flash-AWQ-W4A16`, revision `abd7b07719111f137e1de8a0c1b7e01c11b74d1a`, 177 GiB | Hugging Face |

`vllm/` holds those fifteen files, each with a `.diff` beside it: against the overlay's copy where the overlay has one, otherwise against the file in AMD's image. Twelve are modified copies of overlay files and stay under the overlay's MIT license ([LICENSE-overlay](LICENSE-overlay)); the other three are modified vLLM files under Apache-2.0.

[`apply_port.sh`](apply_port.sh) builds the patched tree in `/tmp/ov/vllm` inside the image: the installed package, then the overlay, then `vllm/` from here. Serve with `PYTHONPATH=/tmp/ov`.

## The switches

The patches added while chasing the long-window fault are behind environment switches that are off by default. Three are needed to serve:

| Switch | Off | On |
| --- | --- | --- |
| `GLM5_DSA_BREAK=indexer` | The first decode step replayed from a CUDA graph reads unmapped memory | The sparse indexer runs outside the captured graph; graphs work |
| `GLM5_PROFILE_ALWAYS=1` | A fixed cache size (`--kv-cache-memory-bytes`) skips the start-up memory profile | The profile runs, then the fixed size is used |
| `GLM5_KDA_NOCACHE=1` | KDA chunk indices built on the GPU behind an identity-keyed cache; every window above 32,768 fails | Built on the CPU, as upstream vLLM does; windows up to 131,072 work |

Diagnostics from the investigation, all off: `GLM5_WS_FIT`, `GLM5_DSA_AUDIT`, `GLM5_DSA_BREAK_PARTS`, `GLM5_MEM_TRACE`, `GLM5_ENABLE_HC_RMS`. The overlay's own switches (`GLM5_DSA_TORCH`, `GLM5_DSA_DENSE`, `GLM5_MHC_TORCH`, `GLM5_PROFILE_*`, `GLM5_DEBUG_*` and others) stay off too; `GLM5_DSA_DENSE` cannot start on gfx1100, which has no dense MLA prefill backend.

The comment at the top of `third_party/flash_linear_attention/ops/index.py` explains the fault as the decode graph writing into memory the prompt step uses. Later runs did not support that: eager mode, with no graphs, fails the same way. The code is right; the comment's explanation is withdrawn (Chapter 5, "Withdrawn explanations").

## The files

| File | Replaces | What it does |
| --- | --- | --- |
| `v1/attention/ops/rocm_aiter_mla_sparse.py` | overlay | A capture-time scalar built with `torch.ones(())` instead of `torch.tensor(1.0)`, whose host-to-device copy is illegal during graph capture; `GLM5_DSA_AUDIT` bounds check |
| `model_executor/layers/mhc.py` | overlay | Turns the image's tilelang mHC kernel off on gfx1100, where its output is wrong, so the fused Triton kernel runs |
| `v1/attention/backends/registry.py` | overlay | Puts back two backend names that vLLM 0.27 refers to and the overlay's copy drops |
| `platforms/rocm.py` | overlay | Removes a doubled `@classmethod` |
| `v1/core/sched/scheduler.py` | overlay | Removes the overlay's degenerate-loop detection, which needs scheduler code newer than 0.27 |
| `v1/core/kv_cache_utils.py` | overlay | Groups the KV cache per pipeline stage, padding each stage's KDA state pages to its own sparse-attention page |
| `models/glm5next/nvidia/model.py` | overlay | Pipeline parallelism: the multi-stream mHC residual passed between stages; diagnostics |
| `model_executor/layers/mla.py` | overlay | `GLM5_DSA_BREAK` |
| `compilation/breakable_cudagraph.py` | image | `GLM5_DSA_BREAK` |
| `model_executor/layers/sparse_attn_indexer_kpool.py` | overlay | `GLM5_MEM_TRACE` (diagnostic) |
| `v1/attention/backends/mla/indexer.py` | overlay | `GLM5_WS_FIT` (diagnostic) |
| `v1/worker/gpu_model_runner.py` | overlay | `GLM5_MEM_TRACE` (diagnostic) |
| `v1/worker/gpu_worker.py` | image | `GLM5_PROFILE_ALWAYS` |
| `third_party/flash_linear_attention/ops/index.py` | overlay | `GLM5_KDA_NOCACHE` |
| `model_executor/layers/fused_moe/routed_experts.py` | image | The load fix above |

## Serving it on ten cards

```bash
export VLLM_ROCM_USE_AITER=0 GPU_MAX_HW_QUEUES=1 HSA_ENABLE_INTERRUPT=1 HSA_ENABLE_MWAITX=1 TRUST_REMOTE_CODE=1 \
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 VLLM_CLAV_CONV1D=1 OMP_NUM_THREADS=8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_RPC_TIMEOUT=3600000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
  VLLM_PP_LAYER_PARTITION=10,9,9,9,8 GLM5_DSA_BREAK=indexer GLM5_PROFILE_ALWAYS=1 GLM5_KDA_NOCACHE=1 PYTHONPATH=/tmp/ov
cd /tmp/ov && python -m vllm.entrypoints.cli.main serve /path/to/GLM-5.3-Flash-AWQ-W4A16 \
  --served-model-name glm --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 5 --distributed-executor-backend mp \
  --gpu-memory-utilization 0.97 --max-num-seqs 2 --max-model-len 131072 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 1073741824 --kv-cache-dtype auto \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'
```

All ten cards visible (no `ROCR_VISIBLE_DEVICES` fence). At utilisation 0.97 it starts only with `expandable_segments` and the image's breakable CUDA graph, which the image turns on by itself. Start with empty `VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR` after any patch change. A cold start runs every card at its power cap for several minutes while Triton tunes.
