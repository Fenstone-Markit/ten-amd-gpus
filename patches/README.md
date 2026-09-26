# Patches

Everything that turns the stock image into the configuration measured in Chapters 1.3 and 2.2
(release rc2 on this machine). Every file's hash is in [MANIFEST.md](MANIFEST.md).

Base image: `docker.io/rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`
(vLLM 0.23.1.dev1, ROCm 7.14.1, PyTorch 2.11). The diffs apply to its files under
`/opt/python/lib/python3.14/site-packages/vllm/`.

## What each patch changes

| File | What it changes | Chapter |
| --- | --- | --- |
| [`vllm/triton_unified_attention.py.diff`](vllm/triton_unified_attention.py.diff) | Lets a two-token speculative check use the fast 3D attention path | 2.1 |
| [`vllm/triton_attn.py.diff`](vllm/triton_attn.py.diff) | Sizes the 3D path's scratch buffers for what the kernel indexes (the out-of-bounds fix), and sets 128 attention segments instead of 16 | 2.1, 1.3 |
| [`vllm/rdna3_w4a16.py.diff`](vllm/rdna3_w4a16.py.diff) | Accepts asymmetric uint4 checkpoints, converts their zero points, and sends batches under 16 tokens to the fixed-order kernel below | 1.2, 2.2 |
| [`kernel/q_gemm_rdna3_det.cu`](kernel/q_gemm_rdna3_det.cu) | The fixed-order kernel: 32-bit partial results in slots of their own, added in order, rounded once | 2.2 |
| [`kernel/qdq_4_rdna3.cuh`](kernel/qdq_4_rdna3.cuh) | Its dequantisation header, unchanged from vLLM | 2.2 |
| [`container/Containerfile.rc2`](container/Containerfile.rc2) | Builds the measured configuration as an image | 1.3 |
| [`container/brain-rc2.sh`](container/brain-rc2.sh) | Starts it: refuses unless the image carries these exact files, keeps both compile caches on the host, proves the compiled graph, sends a warm-up request | 1.3 |

## Building the kernel library

The patched `rdna3_w4a16.py` loads the kernel from `/opt/fenstone/fen_fp32_op.so` (or the path in
`FENSTONE_FP32_OP`). [`tools/kernel/fp32_op_gate.py`](../tools/kernel/fp32_op_gate.py) builds that library from
the kernel source above, registers it as `_fenstone_C::gptq_gemm_rdna3_fp32`, and gates it: precision
against a float64 reference, repeatability, and proof that a compiled server actually calls it. Its
paths at the top are this machine's; set them to yours.

## Settings the patches expect

`--attention-backend TRITON_ATTN`, `--enable-prefix-caching`, `NCCL_ALGO=Ring` at eight cards (measure
at yours), `FENSTONE_FP32_PREFILL=installed`, and for speculative decoding
`--speculative-config '{"method":"mtp","num_speculative_tokens":1,"attention_backend":"TRITON_ATTN"}'`.
The launcher sets all of them.

After changing which kernel vLLM selects, start with an empty `VLLM_CACHE_ROOT`. vLLM otherwise reuses
the graph compiled for the old kernel and serves garbage while reporting healthy (Chapter 2.1).

## Scope

Tested on RX 7900 XTX (gfx1100) with Qwen3.8-27B AWQ INT4, at two, four and eight cards. Nothing here
is tested on other cards, other models or other vLLM versions. vLLM pull request #54706 fixes the same
precision problem upstream, for the prefill kernel as well, and is the better long-term route.
