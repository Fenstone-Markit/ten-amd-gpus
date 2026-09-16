# Chapter 1.1: What the Image Costs

Same hardware, same weights, same prompts. Two container images published by AMD
five days apart. Long-context decode throughput differs by up to 53 percent.

Chapter 1 treated the software stack as a constant, named once in the setup and
never examined. It is not a constant. It is the largest single variable in these
measurements, larger than tensor parallel degree and larger than the choice
between pipeline and tensor parallelism.

## The two images

Both are on Docker Hub. Both are what an RDNA3 user would pull today.

| | Older | Newer |
|---|---|---|
| Tag | `rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0` | `rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0` |
| Built | 2026-09-01 | 2026-08-27 |
| ROCm | 7.14.1 | 10.0.0 |
| PyTorch | 2.11.0 | 2.12.0 |
| vLLM | 0.23.1.dev1+g9ddef7117 | 0.27.1.dev5+gf46a9dfe2 |

The newer image is older. Higher version numbers, earlier source snapshot. That
distinction matters twice below.

## Method

`n02-bench`, in this repository. Prompts padded to an exact token count verified
against the server's `/tokenize` endpoint. Output pinned at 256 tokens with
`min_tokens` and `ignore_eos`. One warm-up request per point, discarded. Three
timed samples, median reported. TTFT from the stream; decode measured as
completion tokens over wall time minus TTFT, so prefill is never folded into a
decode figure.

`cyankiwi/Qwen3-Coder-Next-AWQ-4bit`, compressed-tensors W4A16, symmetric, group
32. Four RX 7900 XTX, 32k max model length, prefix caching off, temperature 0.
The same four physical cards throughout.

Sample spread ran between 0.0 and 2.1 percent.

An earlier draft of Chapter 2 reported throughput measured inside evaluation
conversations, where turn lengths varied and warm-up was included. Those figures
were not comparable and have been removed. This harness exists because of that.

## Results

Decode, tokens per second, median of three.

| Prompt tokens | Old, TP=4 | Old, PP=2×TP=2 | New, PP=2×TP=2 | New, TP=4 |
|---|---|---|---|---|
| 503 | 79.08 | 73.68 | 72.94 | 72.83 |
| 2,061 | 62.00 | 61.00 | 49.23 | 44.55 |
| 8,211 | 33.40 | 34.66 | 22.78 | 17.61 |
| 16,370 | 20.72 | 22.56 | 13.34 | 9.76 |

Time to first token, seconds.

| Prompt tokens | Old, TP=4 | Old, PP=2×TP=2 | New, PP=2×TP=2 | New, TP=4 |
|---|---|---|---|---|
| 503 | 0.13 | 0.14 | 0.13 | 0.13 |
| 2,061 | 0.45 | 0.46 | 0.46 | 0.44 |
| 8,211 | 1.73 | 1.16 | 1.18 | 1.74 |
| 16,370 | 3.55 | 2.16 | 2.22 | 3.55 |

## Peak decode is identical

At 503 tokens all four configurations land within 8 percent, three of them
within 1 percent. Nothing here touches the speed of a forward pass.

That is the number most benchmarks report. A 512-token benchmark would call these
two images equivalent.

## The newer image costs 41 to 53 percent at 16k

| Configuration | Old | New | Loss |
|---|---|---|---|
| PP=2×TP=2 | 22.56 | 13.34 | 41% |
| TP=4 | 20.72 | 9.76 | 53% |

Zero at short context, growing monotonically with sequence length. That is the
signature of the attention and KV path, not the matrix kernels.

Coding sessions, document work and tool loops live between eight and thirty
thousand tokens. That range decides whether a model is usable.

## Prefill is untouched, and parallelism alone sets it

TTFT at 16k is 3.55 seconds at TP=4 and 2.2 at PP=2×TP=2, identical across both
images to within 3 percent.

The two effects separate cleanly. Parallelism sets prefill cost and ignores the
stack. The stack sets long-context decode cost and ignores prefill.

## Going past the KV head floor is nearly free, on the old image

Qwen3-Coder-Next has two KV heads. At TP=4 the degree exceeds the head count, so
vLLM replicates the cache across ranks rather than splitting it. The expectation
was a real penalty at long context.

On the older image it costs 8 percent at 16k, and TP=4 is 7 percent faster at
short context. On the newer image the same change costs 27 percent.

So KV replication is not expensive on this hardware. It is expensive on this
software, and something in the newer stack made it three times worse.

This closes an open question from Chapter 1. An earlier draft asserted that the
tensor parallel degree must divide `num_key_value_heads`. It does not, and a
model was served at PP=5×TP=2 on that false premise. The correction stands; the
cost of the original decision was about 8 percent.

## Two upgrade traps

**The newer image cannot load this class of model.** Every compressed-tensors
W4A16 MoE fails on gfx1100 with `KeyError: 'intermediate_size_full'`, raised from
`fused_moe/routed_experts.py` into `compressed_tensors_moe_wna16.py`.
`_needs_intermediate_size_param` matches the quantization method by class name
against a list of three, and `CompressedTensorsWNA16RDNA3MoEMethod` is not on it.
The same tuple appears twelve lines later in the same file, and there the RDNA3
class is present. One of two sites was updated.

Upstream PR #55522 removed the dispatcher entirely on 12 September 2026. Both
images here predate it, so both carry the defect. Adding the class name to the
tuple is enough to load and serve.

**The newer image gives you a quarter of the KV cache you asked for.** CUDA graph
memory profiling became a default in vLLM 0.21.0. With
`--gpu-memory-utilization 0.92` the server reports an effective 0.7069 and the
cache falls from roughly 1.2 million tokens to 307,000.
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` restores the previous accounting.
Both images here were measured with the old accounting, so cache size is not a
variable in the results above.

## Limits

The two images differ in ROCm, PyTorch, Triton and vLLM at once, and AMD
publishes no crossed combination. Attributing the regression to a single layer
requires building from source.

The newer image was patched to load. The patch adds one string to a list
consulted once at model load and executes zero times per token; at 503 tokens the
patched newer image and the unpatched older one are 1.2 percent apart, and the
divergence appears only as context grows. The load log confirms the intended
kernel: `rocm_moe_rdna.py:40 Using CompressedTensorsWNA16RDNA3MoEMethod (native
RDNA3 HIP kernel)`.

One model, one quantization, one card count.

## What to do with this

Pin your image and record the tag next to any throughput figure you publish.

Benchmark at the context length you work at. 512-token benchmarks are most of
what gets published and they would have shown nothing here.

Do not assume newer is faster. The newer image carries four releases of fixes,
halves startup time, and costs 41 percent where it matters.

## Reproduction

`n02-bench`, both image tags, the patch and the raw JSONL are in `bench/`. Every
figure is a median of three after a discarded warm-up, with per-row spread
recorded.

Corrections welcome, particularly from anyone able to build a crossed image and
pin the regression to one layer.
